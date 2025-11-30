\c postgres_db;

CREATE TABLE IF NOT EXISTS answer_cache (
    query TEXT PRIMARY KEY,
    response JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);