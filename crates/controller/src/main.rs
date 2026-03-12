/// OCRLLM Task Controller
/// =======================
/// Polls the SQLite task queue and drives each PDF through the full pipeline:
///
///   pending
///     └─► processing_ocr        → call OCR Engine (port 8000)
///           └─► processing_embedding → chunk + embed + upsert to Qdrant
///                 └─► completed
///                 │
///                 └─► failed (at any stage)

use std::time::Duration;
use sqlx::{sqlite::SqlitePoolOptions, SqlitePool};
use tokio::time::sleep;
use data_pipeline::chunk_document;
use protocol::{UnifiedDocument, EmbedRequest};

// ── Service endpoints ────────────────────────────────────────────────────────
const OCR_ENGINE_URL:   &str = "http://127.0.0.1:8000/api/v1/extract";
const EMBED_SERVICE_URL: &str = "http://127.0.0.1:8003/embed";
const QDRANT_URL:        &str = "http://127.0.0.1:6333";
const QDRANT_COLLECTION: &str = "ocrllm_docs";

/// Vector dimension emitted by multilingual-e5-small
const EMBED_DIM: u64 = 384;

struct AppState {
    db_pool: SqlitePool,
    http:    reqwest::Client,
}

pub async fn run_controller(db_url: &str) {
    let pool = SqlitePoolOptions::new()
        .max_connections(5)
        .connect(db_url)
        .await
        .expect("Failed to connect to SQLite");

    let state = AppState {
        db_pool: pool,
        http: reqwest::Client::builder()
            .timeout(Duration::from_secs(300))
            .build()
            .expect("Failed to build HTTP client"),
    };

    println!("OCRLLM Task Controller started.");
    ensure_qdrant_collection(&state).await;

    loop {
        if let Err(e) = process_one_task(&state).await {
            eprintln!("[controller] Error: {}", e);
        }
        sleep(Duration::from_secs(5)).await;
    }
}

/// Ensure the Qdrant collection exists before accepting tasks.
async fn ensure_qdrant_collection(state: &AppState) {
    let body = serde_json::json!({
        "vectors": {
            "size": EMBED_DIM,
            "distance": "Cosine"
        }
    });
    let url = format!("{}/collections/{}", QDRANT_URL, QDRANT_COLLECTION);
    match state.http.put(&url).json(&body).send().await {
        Ok(r) if r.status().is_success() || r.status().as_u16() == 409 => {
            println!("[controller] Qdrant collection '{}' ready.", QDRANT_COLLECTION);
        }
        Ok(r) => eprintln!("[controller] Qdrant collection setup warning: {}", r.status()),
        Err(e) => eprintln!("[controller] Qdrant unreachable on startup (will retry): {}", e),
    }
}

/// Pick one task from the queue and advance it through the pipeline.
async fn process_one_task(state: &AppState) -> Result<(), Box<dyn std::error::Error>> {
    // ── Stage 1: Pick up a pending task ──────────────────────────────────────
    let task = sqlx::query!(
        "SELECT id, pdf_path, primary_language FROM tasks WHERE status = 'pending' LIMIT 1"
    )
    .fetch_optional(&state.db_pool)
    .await?;

    let Some(record) = task else { return Ok(()); };
    let task_id = record.id.unwrap_or_default();
    let pdf_path = record.pdf_path;
    let primary_language = record.primary_language;
    println!("[controller] Picked task: {}", task_id);

    set_status(state, &task_id, "processing_ocr", None).await?;

    // ── Stage 2: OCR Extraction ───────────────────────────────────────────────
    let ocr_payload = serde_json::json!({
        "task_id":          task_id,
        "pdf_path":         pdf_path,
        "primary_language": primary_language,
    });

    let ocr_res = state.http
        .post(OCR_ENGINE_URL)
        .json(&ocr_payload)
        .send()
        .await;

    let ocr_json = match ocr_res {
        Ok(r) if r.status().is_success() => r.json::<serde_json::Value>().await?,
        Ok(r) => {
            let msg = format!("OCR HTTP {}: {}", r.status(), r.text().await.unwrap_or_default());
            return fail_task(state, &task_id, &msg).await;
        }
        Err(e) => return fail_task(state, &task_id, &e.to_string()).await,
    };

    if ocr_json.get("status").and_then(|s| s.as_str()) != Some("success") {
        let msg = ocr_json.get("message").and_then(|m| m.as_str()).unwrap_or("unknown OCR error");
        return fail_task(state, &task_id, msg).await;
    }

    println!("[controller] OCR complete for {}", task_id);
    set_status(state, &task_id, "processing_embedding", None).await?;

    // ── Stage 3: Load unified JSON, chunk ────────────────────────────────────
    let json_path = ocr_json["output_path"].as_str().unwrap_or("").to_string();
    let json_bytes = match tokio::fs::read(&json_path).await {
        Ok(b) => b,
        Err(e) => return fail_task(state, &task_id, &format!("Cannot read unified JSON: {}", e)).await,
    };

    let unified_doc: UnifiedDocument = match serde_json::from_slice(&json_bytes) {
        Ok(d) => d,
        Err(e) => return fail_task(state, &task_id, &format!("JSON parse error: {}", e)).await,
    };

    let chunks = chunk_document(&unified_doc);
    let chunk_count = chunks.len();
    println!("[controller] {} chunks produced for {}", chunk_count, task_id);

    if chunks.is_empty() {
        // Empty document: mark complete with 0 chunks
        return complete_task(state, &task_id, 0).await;
    }

    // ── Stage 4: Embed all chunk texts ───────────────────────────────────────
    let texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
    let embed_payload = EmbedRequest { texts };

    let embed_res = state.http
        .post(EMBED_SERVICE_URL)
        .json(&embed_payload)
        .send()
        .await;

    let embed_json = match embed_res {
        Ok(r) if r.status().is_success() => r.json::<serde_json::Value>().await?,
        Ok(r) => {
            let msg = format!("Embed HTTP {}: {}", r.status(), r.text().await.unwrap_or_default());
            return fail_task(state, &task_id, &msg).await;
        }
        Err(e) => return fail_task(state, &task_id, &e.to_string()).await,
    };

    let embeddings = embed_json["embeddings"]
        .as_array()
        .ok_or("Missing embeddings field")?;

    // ── Stage 5: Upsert points to Qdrant ────────────────────────────────────
    let points: Vec<serde_json::Value> = chunks
        .iter()
        .zip(embeddings.iter())
        .map(|(chunk, emb)| {
            serde_json::json!({
                // Qdrant requires unsigned integer IDs; use a hash of chunk_id
                "id":      chunk_id_to_u64(&chunk.chunk_id),
                "vector":  emb,
                "payload": {
                    "chunk_id":    chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "page_num":    chunk.page_num,
                    "text":        chunk.text,
                    "char_count":  chunk.char_count,
                }
            })
        })
        .collect();

    let upsert_url = format!("{}/collections/{}/points", QDRANT_URL, QDRANT_COLLECTION);
    let upsert_body = serde_json::json!({ "points": points });

    match state.http.put(&upsert_url).json(&upsert_body).send().await {
        Ok(r) if r.status().is_success() => {
            println!("[controller] Upserted {} vectors to Qdrant for {}", chunk_count, task_id);
        }
        Ok(r) => {
            let msg = format!("Qdrant upsert HTTP {}: {}", r.status(), r.text().await.unwrap_or_default());
            return fail_task(state, &task_id, &msg).await;
        }
        Err(e) => return fail_task(state, &task_id, &e.to_string()).await,
    }

    complete_task(state, &task_id, chunk_count).await
}

// ── Helpers ──────────────────────────────────────────────────────────────────

async fn set_status(
    state: &AppState,
    task_id: &str,
    status: &str,
    error: Option<&str>,
) -> Result<(), sqlx::Error> {
    sqlx::query!(
        "UPDATE tasks SET status = ?, error_message = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
        status,
        error,
        task_id
    )
    .execute(&state.db_pool)
    .await?;
    Ok(())
}

async fn fail_task(
    state: &AppState,
    task_id: &str,
    error: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("[controller] Task {} failed: {}", task_id, error);
    set_status(state, task_id, "failed", Some(error)).await?;
    Ok(())
}

async fn complete_task(
    state: &AppState,
    task_id: &str,
    chunk_count: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let n = chunk_count as i64;
    sqlx::query(
        "UPDATE tasks SET status = 'completed', chunk_count = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
    )
    .bind(n)
    .bind(task_id)
    .execute(&state.db_pool)
    .await?;
    println!("[controller] Task {} completed ({} chunks).", task_id, chunk_count);
    Ok(())
}

/// Deterministic u64 hash of a string chunk_id for Qdrant point ID.
/// Uses FNV-1a (no external dep needed).
fn chunk_id_to_u64(id: &str) -> u64 {
    let mut hash: u64 = 14695981039346656037;
    for byte in id.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(1099511628211);
    }
    hash
}

#[tokio::main]
async fn main() {
    let db_url = std::env::var("DATABASE_URL")
        .unwrap_or_else(|_| "sqlite://ocrllm.sqlite".to_string());
    run_controller(&db_url).await;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_id_to_u64_deterministic() {
        let id = "test_p1_c0";
        let h1 = chunk_id_to_u64(id);
        let h2 = chunk_id_to_u64(id);
        assert_eq!(h1, h2);
        assert_ne!(h1, 0);
    }

    #[test]
    fn test_chunk_id_to_u64_different() {
        let h1 = chunk_id_to_u64("a");
        let h2 = chunk_id_to_u64("b");
        assert_ne!(h1, h2);
    }
}

