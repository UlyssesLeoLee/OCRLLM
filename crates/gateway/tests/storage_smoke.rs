use sqlx::{sqlite::SqlitePoolOptions, Row};
use std::time::Duration;

#[tokio::test]
async fn test_sqlite_connectivity() {
    // Use an in-memory database for testing the logic
    let db_url = "sqlite::memory:";
    let pool = SqlitePoolOptions::new()
        .connect(db_url)
        .await
        .expect("Failed to connect to in-memory SQLite");

    // Initialize schema (matching the real one)
    sqlx::query(
        "CREATE TABLE tasks (
            id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            pdf_path TEXT NOT NULL,
            primary_language TEXT NOT NULL,
            chunk_count INTEGER,
            error_message TEXT,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )"
    )
    .execute(&pool)
    .await
    .expect("Failed to create table");

    // Insert a dummy task
    let task_id = "smoke-test-task";
    sqlx::query("INSERT INTO tasks (id, status, pdf_path, primary_language) VALUES (?, 'pending', 'dummy.pdf', 'ja')")
        .bind(task_id)
        .execute(&pool)
        .await
        .expect("Failed to insert task");

    // Retrieve the task
    let row = sqlx::query("SELECT status FROM tasks WHERE id = ?")
        .bind(task_id)
        .fetch_one(&pool)
        .await
        .expect("Failed to fetch task");

    let status: String = row.get("status");
    assert_eq!(status, "pending");
}

#[tokio::test]
async fn test_qdrant_health_smoke() {
    let qdrant_url = "http://127.0.0.1:6333/healthz";
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(2))
        .build()
        .unwrap();

    // If Qdrant is running, this should return 200 OK.
    let res = client.get(qdrant_url).send().await;
    if let Ok(r) = res {
        assert!(r.status().is_success(), "Qdrant health check failed with status: {}", r.status());
    } else {
        println!("INFO: Qdrant unreachable, skipping Qdrant smoke test. (Run 'docker-compose up qdrant' to test fully)");
    }
}
