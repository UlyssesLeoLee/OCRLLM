-- Schema for tracking PDF extraction tasks via SQLite
CREATE TABLE tasks (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL, -- pending, processing, completed, failed
    pdf_path TEXT NOT NULL,
    primary_language TEXT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    error_message TEXT
);
