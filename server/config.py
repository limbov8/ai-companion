from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DeepSeekConfig:
    api_key: str | None
    base_url: str
    conversation_model: str
    utility_model: str
    thinking: str
    reasoning_effort: str


@dataclass(frozen=True)
class ModelConfig:
    asr_model_id: str
    embedding_model_id: str
    tts_model_id: str
    device: str
    lazy_tts: bool


@dataclass(frozen=True)
class DatabaseConfig:
    dsn: str


@dataclass(frozen=True)
class MemoryConfig:
    top_k: int
    rerank_top_k: int
    similarity_floor: float


@dataclass(frozen=True)
class ConversationConfig:
    proactive_topic_interval_minutes: int
    allow_random_topics: bool


@dataclass(frozen=True)
class AppConfig:
    deepseek: DeepSeekConfig
    models: ModelConfig
    database: DatabaseConfig
    memory: MemoryConfig
    conversation: ConversationConfig


def load_config(path: str | Path = "config.yml") -> AppConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    deepseek = raw["deepseek"]
    models = raw["models"]
    db = raw["database"]
    memory = raw["memory"]
    conversation = raw["conversation"]

    return AppConfig(
        deepseek=DeepSeekConfig(
            api_key=os.getenv(deepseek.get("api_key_env", "DEEPSEEK_API_KEY")),
            base_url=deepseek["base_url"].rstrip("/"),
            conversation_model=deepseek["models"]["conversation"],
            utility_model=deepseek["models"]["other"],
            thinking=deepseek.get("thinking", "disabled"),
            reasoning_effort=deepseek.get("reasoning_effort", "medium"),
        ),
        models=ModelConfig(
            asr_model_id=models["asr"]["model_id"],
            embedding_model_id=models["embedding"]["model_id"],
            tts_model_id=models["tts"]["model_id"],
            device=models["asr"].get("device", "cuda"),
            lazy_tts=bool(models["tts"].get("lazy_load", True)),
        ),
        database=DatabaseConfig(dsn=os.getenv(db.get("dsn_env", "DATABASE_URL"), db["default_dsn"])),
        memory=MemoryConfig(
            top_k=int(memory["top_k"]),
            rerank_top_k=int(memory["rerank_top_k"]),
            similarity_floor=float(memory["similarity_floor"]),
        ),
        conversation=ConversationConfig(
            proactive_topic_interval_minutes=int(conversation["proactive_topic_interval_minutes"]),
            allow_random_topics=bool(conversation["allow_random_topics"]),
        ),
    )


def deep_merge(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    merged = dict(left)
    for key, value in right.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
