CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS conversation_turns (
  id BIGSERIAL PRIMARY KEY,
  session_id TEXT NOT NULL,
  role TEXT NOT NULL,
  content TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS conversation_turns_session_idx
  ON conversation_turns (session_id, created_at);

CREATE TABLE IF NOT EXISTS memories (
  id UUID PRIMARY KEY,
  text TEXT NOT NULL,
  category TEXT NOT NULL DEFAULT 'general',
  metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
  embedding VECTOR(1024) NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS memories_embedding_hnsw_idx
  ON memories USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS memories_category_idx
  ON memories (category);
