use axum::{
    extract::{Multipart, Path, State},
    http::StatusCode,
    routing::{get, post},
    Router, Json,
};
use serde::{Deserialize, Serialize};
use sqlx::{sqlite::SqlitePoolOptions, SqlitePool, Row};
use std::sync::Arc;
use std::time::Duration;
use tokio::fs::{self, File};
use tokio::io::AsyncWriteExt;
use tower_http::{
    cors::{Any, CorsLayer},
    services::ServeDir,
};
use uuid::Uuid;
use protocol::{QueryRequest, QueryResponse, RankedChunk};

// ── External service endpoints ────────────────────────────────────────────────
const EMBED_SERVICE_URL:   &str = "http://127.0.0.1:8003/embed";
const RERANK_SERVICE_URL:  &str = "http://127.0.0.1:8004/rerank";
const MODEL_RUNNER_URL:    &str = "http://127.0.0.1:8002/v1/chat/completions";
const QDRANT_URL:          &str = "http://127.0.0.1:6333";
const QDRANT_COLLECTION:   &str = "ocrllm_docs";

// ── Shared state ─────────────────────────────────────────────────────────────

#[derive(Serialize)]
struct SubmitResponse {
    task_id: String,
    status: String,
    message: String,
}

#[derive(Serialize, Deserialize)]
struct TaskStatusResponse {
    task_id: String,
    status: String,
    pdf_path: String,
    primary_language: String,
    chunk_count: Option<i64>,
    error_message: Option<String>,
}

struct AppState {
    db_pool: SqlitePool,
    http: reqwest::Client,
}

// ── Server bootstrap ─────────────────────────────────────────────────────────

pub async fn run_server(db_url: &str) {
    let pool = SqlitePoolOptions::new()
        .max_connections(5)
        .connect(db_url)
        .await
        .expect("Failed to connect to SQLite");

    let shared_state = Arc::new(AppState {
        db_pool: pool,
        http: reqwest::Client::builder()
            .timeout(Duration::from_secs(120))
            .build()
            .expect("Failed to build HTTP client"),
    });

    let cors = CorsLayer::new().allow_origin(Any);
    fs::create_dir_all("storage/raw_pdfs").await.unwrap();

    let app = Router::new()
        .route("/health", get(|| async { "Gateway is healthy." }))
        .route("/api/v1/submit", post(submit_pdf))
        .route("/api/v1/tasks/{id}", get(get_task_status))
        .route("/api/v1/query", post(query_rag))
        .fallback_service(ServeDir::new("crates/gateway/static"))
        .layer(cors)
        .with_state(shared_state);

    let addr = "0.0.0.0:7070";
    println!("API Gateway & UI listening on http://{}", addr);
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

// ── Handlers ─────────────────────────────────────────────────────────────────

async fn submit_pdf(
    State(state): State<Arc<AppState>>,
    mut multipart: Multipart,
) -> Result<Json<SubmitResponse>, (StatusCode, String)> {
    let task_id = Uuid::new_v4().to_string();
    let mut primary_language = "ja".to_string();
    let mut saved_path = String::new();

    while let Some(field) = multipart.next_field().await.unwrap() {
        let name = field.name().unwrap_or("").to_string();

        if name == "primary_language" {
            let data = field.text().await.unwrap_or_default();
            if !data.is_empty() {
                primary_language = data;
            }
        } else if name == "file" {
            let file_name = field.file_name().unwrap_or("upload.pdf").to_string();
            let sanitized_name = format!("{}_{}", task_id, file_name);
            let path = format!("storage/raw_pdfs/{}", sanitized_name);

            let data = field.bytes().await.map_err(|e| {
                (StatusCode::INTERNAL_SERVER_ERROR, format!("Failed to read file: {}", e))
            })?;

            let mut file = File::create(&path).await.map_err(|e| {
                (StatusCode::INTERNAL_SERVER_ERROR, format!("Failed to create file: {}", e))
            })?;

            file.write_all(&data).await.map_err(|e| {
                (StatusCode::INTERNAL_SERVER_ERROR, format!("Failed to write file: {}", e))
            })?;

            // Use absolute path; replace Windows backslashes for cross-process compatibility
            saved_path = std::env::current_dir()
                .unwrap()
                .join(path)
                .to_string_lossy()
                .replace('\\', "/");
        }
    }

    if saved_path.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "No valid PDF file uploaded".to_string()));
    }

    let result = sqlx::query!(
        r#"INSERT INTO tasks (id, status, pdf_path, primary_language) VALUES ($1, 'pending', $2, $3)"#,
        task_id,
        saved_path,
        primary_language
    )
    .execute(&state.db_pool)
    .await;

    match result {
        Ok(_) => Ok(Json(SubmitResponse {
            task_id,
            status: "pending".to_string(),
            message: "File uploaded successfully. Processing will begin shortly.".to_string(),
        })),
        Err(e) => {
            eprintln!("Failed to insert task: {}", e);
            Err((StatusCode::INTERNAL_SERVER_ERROR, "Database error".to_string()))
        }
    }
}

async fn get_task_status(
    Path(id): Path<String>,
    State(state): State<Arc<AppState>>,
) -> Result<Json<TaskStatusResponse>, (StatusCode, String)> {
    let task = sqlx::query(
        "SELECT id, status, pdf_path, primary_language, chunk_count, error_message FROM tasks WHERE id = ?"
    )
    .bind(id)
    .fetch_optional(&state.db_pool)
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    if let Some(row) = task {
        Ok(Json(TaskStatusResponse {
            task_id:          row.get::<Option<String>, _>("id").unwrap_or_default(),
            status:           row.get("status"),
            pdf_path:         row.get::<Option<String>, _>("pdf_path").unwrap_or_default(),
            primary_language: row.get::<Option<String>, _>("primary_language").unwrap_or_default(),
            chunk_count:      row.get("chunk_count"),
            error_message:    row.get("error_message"),
        }))
    } else {
        Err((StatusCode::NOT_FOUND, "Task not found".to_string()))
    }
}

/// RAG Query handler — 4-step chain:
/// 1. Embed the question via Embed Service
/// 2. Search Qdrant for top-(top_k * 4) candidates
/// 3. Rerank via Reranker Service → top_k
/// 4. Call Qwen3.5 with context, return answer + sources
async fn query_rag(
    State(state): State<Arc<AppState>>,
    Json(req): Json<QueryRequest>,
) -> Result<Json<QueryResponse>, (StatusCode, String)> {
    let top_k = req.top_k.unwrap_or(5);
    let retrieval_k = top_k * 4; // over-fetch for reranker

    // ── Step 1: Embed the question ────────────────────────────────────────────
    let embed_payload = serde_json::json!({
        "texts": [req.question],
        "mode": "query"
    });

    let embed_res = state.http
        .post(EMBED_SERVICE_URL)
        .json(&embed_payload)
        .send()
        .await
        .map_err(|e| (StatusCode::SERVICE_UNAVAILABLE, format!("Embed service error: {}", e)))?;

    let embed_json: serde_json::Value = embed_res.json().await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let question_vector = embed_json["embeddings"][0].clone();

    // ── Step 2: Qdrant vector search ──────────────────────────────────────────
    let mut qdrant_search = serde_json::json!({
        "vector": question_vector,
        "limit": retrieval_k,
        "with_payload": true
    });

    // Optionally filter by doc_id
    if let Some(ref doc_id) = req.doc_id {
        qdrant_search["filter"] = serde_json::json!({
            "must": [{
                "key": "document_id",
                "match": { "value": doc_id }
            }]
        });
    }

    let qdrant_url = format!("{}/collections/{}/points/search", QDRANT_URL, QDRANT_COLLECTION);
    let qdrant_res = state.http
        .post(&qdrant_url)
        .json(&qdrant_search)
        .send()
        .await
        .map_err(|e| (StatusCode::SERVICE_UNAVAILABLE, format!("Qdrant error: {}", e)))?;

    let qdrant_json: serde_json::Value = qdrant_res.json().await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let qdrant_hits = qdrant_json["result"]
        .as_array()
        .cloned()
        .unwrap_or_default();

    if qdrant_hits.is_empty() {
        return Ok(Json(QueryResponse {
            answer: "No relevant documents found. Please ingest some PDFs first.".to_string(),
            sources: vec![],
        }));
    }

    // Build candidates list for the reranker
    let candidates: Vec<serde_json::Value> = qdrant_hits
        .iter()
        .map(|hit| {
            let payload = &hit["payload"];
            serde_json::json!({
                "chunk_id": payload["chunk_id"].as_str().unwrap_or(""),
                "text":     payload["text"].as_str().unwrap_or(""),
                "page_num": payload["page_num"].as_i64().unwrap_or(0),
            })
        })
        .collect();

    // ── Step 3: Rerank via Reranker Service ───────────────────────────────────
    let rerank_payload = serde_json::json!({
        "question":   req.question,
        "candidates": candidates,
        "top_k":      top_k,
    });

    let rerank_res = state.http
        .post(RERANK_SERVICE_URL)
        .json(&rerank_payload)
        .send()
        .await
        .map_err(|e| (StatusCode::SERVICE_UNAVAILABLE, format!("Reranker error: {}", e)))?;

    let rerank_json: serde_json::Value = rerank_res.json().await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let ranked_arr = rerank_json["ranked"].as_array().cloned().unwrap_or_default();

    let sources: Vec<RankedChunk> = ranked_arr
        .iter()
        .map(|r| RankedChunk {
            chunk_id: r["chunk_id"].as_str().unwrap_or("").to_string(),
            text:     r["text"].as_str().unwrap_or("").to_string(),
            page_num: r["page_num"].as_i64().unwrap_or(0) as i32,
            score:    r["score"].as_f64().unwrap_or(0.0) as f32,
        })
        .collect();

    // ── Step 4: Generate answer with Qwen3.5 ─────────────────────────────────
    let context: String = sources
        .iter()
        .enumerate()
        .map(|(i, chunk)| format!("[Source {} — Page {}]\n{}", i + 1, chunk.page_num, chunk.text))
        .collect::<Vec<_>>()
        .join("\n\n---\n\n");

    let system_prompt = "You are a helpful assistant for analysing Japanese, Chinese and English \
        documents. Answer the user's question using ONLY the provided source excerpts. \
        If the answer cannot be found in the sources, say so clearly. \
        Cite the source number in your answer when referencing specific content.";

    let user_message = format!(
        "Sources:\n{}\n\n---\n\nQuestion: {}",
        context, req.question
    );

    let llm_payload = serde_json::json!({
        "messages": [
            { "role": "system", "content": system_prompt },
            { "role": "user",   "content": user_message }
        ],
        "max_tokens": 1024,
        "temperature": 0.2
    });

    let llm_res = state.http
        .post(MODEL_RUNNER_URL)
        .json(&llm_payload)
        .send()
        .await
        .map_err(|e| (StatusCode::SERVICE_UNAVAILABLE, format!("Model-Runner error: {}", e)))?;

    let llm_json: serde_json::Value = llm_res.json().await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    // Handle both error and success responses from the model runner
    if let Some(err) = llm_json.get("error") {
        return Err((StatusCode::INTERNAL_SERVER_ERROR, err.to_string()));
    }

    let answer = llm_json["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("(no answer generated)")
        .to_string();

    Ok(Json(QueryResponse { answer, sources }))
}

#[tokio::main]
async fn main() {
    let db_url = std::env::var("DATABASE_URL")
        .unwrap_or_else(|_| "sqlite://ocrllm.sqlite".to_string());
    run_server(&db_url).await;
}
