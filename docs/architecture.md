# Architecture

This monolith is split by component folders:

- `server/agent`: conversation state and orchestration.
- `server/models`: lazy local model adapters for Whisper ASR, Qwen embeddings, and Qwen TTS.
- `server/memory`: in-memory and Postgres/pgvector memory stores.
- `server/tools`: tool and skill registry.
- `server/voice`: voice session and barge-in state.
- `web/static`: browser UI.
- `prompts`: prompt registry and eval cases.
- `docker`: database schema and container files.

## Single GPU behavior

The `SingleGpuGate` serializes local model work. Heavy adapters are lazy-loaded, so ASR, embedding, and TTS work can be staged instead of trying to run all local models at once. For production use, add model unload hooks after each call or introduce a small worker queue per model.

## RAG flow

1. User text arrives from typed input or transcribed voice.
2. The orchestrator embeds the query and retrieves memories from the vector store.
3. Retrieved memories are reranked and inserted into the system prompt context.
4. DeepSeek generates a multi-turn response.
5. DeepSeek is asked whether the user text or tool result is durable memory.
6. Durable memory is embedded and stored with metadata.

## Voice flow

The UI uses browser microphone capture and sends audio chunks for ASR. The `/ws/voice/{session_id}` endpoint is ready for streaming events and barge-in messages. A full peer-connection WebRTC transport can be added around this without changing the agent orchestrator.
