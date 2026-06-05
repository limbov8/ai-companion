from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import re
import wave
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from server.agent.orchestrator import AgentOrchestrator
from server.config import load_config
from server.llm.deepseek import DeepSeekClient
from server.logging import ConversationLogger, configure_logging
from server.memory.store import InMemoryVectorStore
from server.models.asr import LocalAsrService
from server.models.embedding import EmbeddingService
from server.models.gpu import SingleGpuGate
from server.models.tts import QwenTtsService, TtsUnavailableError
from server.prompts.registry import PromptRegistry
from server.tools.registry import ToolRegistry
from server.tools.web_search import WebSearchTool
from server.voice.session import VoiceSessionManager

log = logging.getLogger(__name__)
SENTENCE_END_RE = re.compile(r"^(.+?[.!?。！？](?:\s+|$))", re.DOTALL)


class ChatRequest(BaseModel):
    session_id: str | None = None
    text: str


class ToolRunRequest(BaseModel):
    name: str
    args: dict[str, object] = {}


def build_services() -> dict[str, object]:
    config = load_config()
    gpu_gate = SingleGpuGate()
    tools = ToolRegistry()
    tools.register(WebSearchTool())
    embeddings = EmbeddingService(config.models.embedding_model_id, config.models.device, gpu_gate)
    return {
        "config": config,
        "sessions": VoiceSessionManager(),
        "logger": ConversationLogger(),
        "asr": LocalAsrService(
            config.models.asr_model_id,
            config.models.device,
            gpu_gate,
            language=config.models.asr_language,
        ),
        "tts": QwenTtsService(
            config.models.tts_model_id,
            config.models.device,
            gpu_gate,
            language=config.models.tts_language,
            languages=config.models.tts_languages,
            speed=config.models.tts_speed,
            speaker=config.models.tts_speaker,
            ref_audio=config.models.tts_ref_audio,
            ref_text=config.models.tts_ref_text,
            instruct=config.models.tts_instruct,
        ),
        "orchestrator": AgentOrchestrator(
            chat=DeepSeekClient(config.deepseek),
            embeddings=embeddings,
            memory_store=InMemoryVectorStore(),
            prompts=PromptRegistry(),
            tools=tools,
            memory_config=config.memory,
            conversation_config=config.conversation,
        ),
        "tools": tools,
        "prompts": PromptRegistry(),
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    app.state.services = build_services()
    if os.getenv("AI_COMPANION_PRELOAD_LOCAL_MODELS", "0") == "1":
        strict = os.getenv("AI_COMPANION_STRICT_LOCAL_MODELS", "0") == "1"
        await preload_local_models(app.state.services, strict=strict)
        if os.getenv("AI_COMPANION_SMOKE_TEST_LOCAL_MODELS", "0") == "1":
            await smoke_test_local_models(app.state.services)
    log.info("AI companion services ready")
    yield


async def preload_local_models(services: dict[str, object], *, strict: bool) -> None:
    log.info("Preloading local ASR, embedding, and TTS models on GPU")
    asr: LocalAsrService = services["asr"]
    orchestrator: AgentOrchestrator = services["orchestrator"]
    tts: QwenTtsService = services["tts"]
    results = {
        "asr": await asr.preload(strict=strict),
        "embedding": await orchestrator.embeddings.preload(strict=strict),
        "tts": await tts.preload(strict=strict),
    }
    if strict and not all(results.values()):
        raise RuntimeError(f"Local model preload failed: {results}")
    log.info("Local model preload results: %s", results)


async def smoke_test_local_models(services: dict[str, object]) -> None:
    log.info("Running local model smoke inference for ASR, embedding, and TTS")
    asr: LocalAsrService = services["asr"]
    orchestrator: AgentOrchestrator = services["orchestrator"]
    tts: QwenTtsService = services["tts"]

    embedding = await orchestrator.embeddings.embed("local model smoke test")
    transcript = await asr.transcribe(silence_wav_bytes(duration_seconds=0.35), "audio/wav")
    audio = await tts.synthesize("Hello. Local voice model is ready.")
    log.info(
        "Local model smoke results: embedding_dims=%s asr_chars=%s tts_bytes=%s",
        len(embedding),
        len(transcript),
        len(audio),
    )


def silence_wav_bytes(*, duration_seconds: float, sample_rate: int = 16_000) -> bytes:
    frame_count = int(sample_rate * duration_seconds)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x00\x00" * frame_count)
    return buffer.getvalue()


def pop_speakable_chunks(buffer: str) -> tuple[list[str], str]:
    chunks: list[str] = []
    pending = buffer
    while True:
        match = SENTENCE_END_RE.match(pending)
        if not match:
            break
        chunk = match.group(1).strip()
        if chunk:
            chunks.append(chunk)
        pending = pending[match.end() :]
    if len(pending) >= 120:
        split_at = max(pending.rfind(","), pending.rfind(";"), pending.rfind("，"), pending.rfind("；"))
        if split_at >= 60:
            chunks.append(pending[: split_at + 1].strip())
            pending = pending[split_at + 1 :]
    return chunks, pending


app = FastAPI(title="AI Companion", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="web/static"), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse("web/static/index.html")


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/chat")
async def chat(request: ChatRequest) -> dict[str, object]:
    services = app.state.services
    sessions: VoiceSessionManager = services["sessions"]
    logger: ConversationLogger = services["logger"]
    orchestrator: AgentOrchestrator = services["orchestrator"]
    session = sessions.get(request.session_id)
    logger.record(session.session_id, "user", request.text)
    response = await orchestrator.handle_text(request.text, session.messages(), source="text")
    session.add("user", request.text)
    session.add("assistant", response.text)
    logger.record(session.session_id, "assistant", response.text)
    return {
        "session_id": session.session_id,
        "text": response.text,
        "memory_stored": response.memory_stored,
        "memories_used": [item.text for item in response.memories_used],
    }


@app.post("/api/asr")
async def transcribe(
    audio: Annotated[UploadFile, File()],
    session_id: Annotated[str | None, Form()] = None,
) -> dict[str, object]:
    services = app.state.services
    asr: LocalAsrService = services["asr"]
    data = await audio.read()
    text = await asr.transcribe(data, audio.content_type or "audio/webm")
    log.info(
        "ASR speech-to-text result session_id=%s filename=%s content_type=%s bytes=%s text=%r",
        session_id,
        audio.filename,
        audio.content_type,
        len(data),
        text,
    )
    return {"session_id": session_id, "text": text}


@app.post("/api/tts")
async def synthesize(request: ChatRequest) -> Response:
    services = app.state.services
    sessions: VoiceSessionManager = services["sessions"]
    tts: QwenTtsService = services["tts"]
    session = sessions.get(request.session_id)
    generation = session.speaking_generation
    try:
        audio = await tts.synthesize(
            request.text,
            interrupt_token="cancelled" if generation != session.speaking_generation else None,
        )
    except TtsUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return Response(audio, media_type="audio/wav")


@app.post("/api/barge-in/{session_id}")
async def barge_in(session_id: str) -> dict[str, object]:
    services = app.state.services
    sessions: VoiceSessionManager = services["sessions"]
    return {"session_id": session_id, "generation": sessions.barge_in(session_id)}


@app.get("/api/prompts")
async def prompts() -> dict[str, object]:
    registry: PromptRegistry = app.state.services["prompts"]
    return {"prompts": [template.__dict__ for template in registry.list()]}


@app.get("/api/evals")
async def evals() -> JSONResponse:
    return JSONResponse(content={"path": "prompts/evals.yml", "status": "registered"})


@app.get("/api/tools")
async def tools() -> dict[str, object]:
    registry: ToolRegistry = app.state.services["tools"]
    return {"tools": [spec.__dict__ for spec in registry.list_specs()]}


@app.post("/api/tools/run")
async def run_tool(request: ToolRunRequest) -> dict[str, object]:
    services = app.state.services
    registry: ToolRegistry = services["tools"]
    orchestrator: AgentOrchestrator = services["orchestrator"]
    result = await registry.run(request.name, **request.args)
    stored = await orchestrator.maybe_store_tool_result(request.name, result)
    return {"result": result, "memory_stored": stored}


@app.websocket("/ws/voice/{session_id}")
async def voice_socket(websocket: WebSocket, session_id: str) -> None:
    await websocket.accept()
    services = app.state.services
    sessions: VoiceSessionManager = services["sessions"]
    session = sessions.get(session_id)
    audio_chunks: list[bytes] = []
    audio_content_type = "audio/webm"
    try:
        while True:
            payload = await websocket.receive_json()
            event_type = payload.get("type")
            if event_type == "barge_in":
                await websocket.send_json({"type": "barge_in", "generation": session.barge_in()})
            elif event_type == "audio_start":
                audio_chunks = []
                audio_content_type = str(payload.get("content_type") or "audio/webm")
                await websocket.send_json({"type": "audio_started"})
            elif event_type == "audio_chunk":
                audio = base64.b64decode(str(payload.get("audio", "")))
                audio_chunks.append(audio)
                await websocket.send_json({"type": "audio_received", "bytes": len(audio)})
            elif event_type == "audio_end":
                data = b"".join(audio_chunks)
                text = await services["asr"].transcribe(data, audio_content_type)
                log.info(
                    "ASR streaming speech-to-text result session_id=%s content_type=%s bytes=%s text=%r",
                    session_id,
                    audio_content_type,
                    len(data),
                    text,
                )
                audio_chunks = []
                await websocket.send_json({"type": "transcript", "text": text})
            elif event_type == "text":
                user_text = str(payload.get("text", "")).strip()
                await stream_voice_response(websocket, services, session, user_text)
            else:
                await websocket.send_json({"type": "error", "message": "unknown event"})
    except WebSocketDisconnect:
        log.info("voice websocket disconnected: %s", session_id)


async def stream_voice_response(
    websocket: WebSocket,
    services: dict[str, object],
    session: object,
    user_text: str,
) -> None:
    orchestrator: AgentOrchestrator = services["orchestrator"]
    logger: ConversationLogger = services["logger"]
    tts: QwenTtsService = services["tts"]
    send_lock = asyncio.Lock()
    speech_queue: asyncio.Queue[str | None] = asyncio.Queue()
    generation = session.speaking_generation

    async def send_json(payload: dict[str, object]) -> None:
        async with send_lock:
            await websocket.send_json(payload)

    async def tts_worker() -> None:
        while True:
            chunk = await speech_queue.get()
            if chunk is None:
                return
            if generation != session.speaking_generation:
                return
            try:
                audio = await tts.synthesize(
                    chunk,
                    interrupt_token="cancelled" if generation != session.speaking_generation else None,
                )
            except TtsUnavailableError as exc:
                await send_json({"type": "tts_error", "message": str(exc), "text": chunk})
                continue
            if audio and generation == session.speaking_generation:
                await send_json(
                    {
                        "type": "assistant_audio",
                        "audio": base64.b64encode(audio).decode("ascii"),
                        "mime_type": "audio/wav",
                        "text": chunk,
                    }
                )

    logger.record(session.session_id, "user", user_text)
    await send_json({"type": "assistant_start"})
    worker = asyncio.create_task(tts_worker())
    answer = ""
    tts_buffer = ""
    final_event: dict[str, object] = {}
    try:
        async for event in orchestrator.handle_text_stream(user_text, session.messages(), source="voice"):
            if event.get("type") == "text_delta":
                delta = str(event.get("text") or "")
                answer += delta
                tts_buffer += delta
                await send_json({"type": "assistant_delta", "text": delta})
                chunks, tts_buffer = pop_speakable_chunks(tts_buffer)
                for chunk in chunks:
                    await speech_queue.put(chunk)
            elif event.get("type") == "done":
                final_event = event
        if tts_buffer.strip():
            await speech_queue.put(tts_buffer.strip())
        await speech_queue.put(None)
        await worker
        session.add("user", user_text)
        session.add("assistant", answer)
        logger.record(session.session_id, "assistant", answer)
        await send_json(
            {
                "type": "assistant_done",
                "text": answer,
                "memory_stored": bool(final_event.get("memory_stored")),
                "memories_used": final_event.get("memories_used") or [],
            }
        )
    finally:
        if not worker.done():
            worker.cancel()
