# Architecture

This monolith is split by component folders:

- `server/agent`: conversation state and orchestration.
- `server/conversations.py`: database-backed conversation history listing and loading.
- `server/agent/decision.py` and `server/agent/router.py`: turn-level control policy.
- `server/skills`: stateful multi-turn workflows such as planning.
- `server/models`: lazy local model adapters for Whisper ASR, Qwen embeddings, and Qwen TTS.
- `server/memory`: in-memory and Postgres/pgvector memory stores.
- `server/tools`: tool and skill registry.
- `server/voice`: voice session and barge-in state.
- `web/static`: browser UI.
- `prompts`: prompt registry and eval cases.
- `docker`: database schema and container files.

## Single GPU behavior

The `SingleGpuGate` serializes local model work. Heavy adapters are lazy-loaded, so ASR, embedding, and TTS work can be staged instead of trying to run all local models at once. Set `AI_COMPANION_ENABLE_LOCAL_MODELS=1` when the Hugging Face models are downloaded and ready; otherwise the app uses deterministic local fallbacks for development. For production use, add model unload hooks after each call or introduce a small worker queue per model.

## RAG flow

1. User text arrives from typed input or transcribed voice.
2. The orchestrator embeds the query and retrieves memories from the vector store.
3. The router receives user text, history, memories, active task state, tool specs, and skill specs.
4. The router chooses answer, clarification, tool use, start skill, or continue skill.
5. Retrieved memories and any tool results are inserted into the system prompt context.
6. DeepSeek generates a multi-turn response, or a skill renders the next step.
7. DeepSeek is asked whether the user text or tool result is durable memory.
8. Durable memory is embedded and stored with metadata.

## Agent control flow

The companion now has a control layer around the previous chatbot flow. `AgentDecision` is the central turn decision object. Deterministic rules catch obvious current-info, memory, and planning cases; the router can fall back to a strict JSON LLM decision. Skills are separate from atomic tools: tools do one external action, while skills own multi-turn state. Active skill state lives on the in-memory conversation session for this first pass.

To add a tool, implement `Tool`, fill out `ToolSpec` metadata including `when_to_use`, and register it in `build_services`. To add a skill, implement `Skill`, expose a `SkillSpec`, register it in `SkillRegistry`, and teach the router when to start it.

## Voice flow

The UI uses browser microphone capture and sends audio chunks for ASR. The `/ws/voice/{session_id}` endpoint is ready for streaming events and barge-in messages. A full peer-connection WebRTC transport can be added around this without changing the agent orchestrator.

## Conversation history

Conversation turns are written to `conversation_turns` in Postgres when the configured database is available. The UI loads recent sessions from `/api/conversations`; selecting one calls `/api/conversations/{session_id}/load`, restores the visible history, and makes that session the active server-side context for continuing the conversation.
