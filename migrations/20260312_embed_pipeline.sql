-- Add embedding pipeline columns to tasks table
-- Run this after the initial schema migration
ALTER TABLE tasks ADD COLUMN chunk_count INTEGER;
ALTER TABLE tasks ADD COLUMN embed_model TEXT DEFAULT 'intfloat/multilingual-e5-small';
