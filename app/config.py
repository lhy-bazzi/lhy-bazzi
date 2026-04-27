"""
UniAI Python AI Service — Configuration Management

Supports:
- YAML file loading (configs/config.yaml)
- Environment-specific overrides (configs/config.{env}.yaml)
- Environment variable overrides (highest priority)
- Global singleton via get_settings()
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource

# Project root: uni-ai-python/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"


# ---------------------------------------------------------------------------
# Config sub-models
# ---------------------------------------------------------------------------

class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8100
    workers: int = 4


class DatabaseConfig(BaseModel):
    url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/uni_ai"
    pool_size: int = 20
    max_overflow: int = 10
    pool_timeout: int = 30
    pool_recycle: int = 1800
    auto_create_tables: bool = True


class MinioConfig(BaseModel):
    endpoint: str = "localhost:9000"
    access_key: str = "minioadmin"
    secret_key: str = "minioadmin"
    bucket: str = "uni-ai-docs"
    secure: bool = False


class RedisConfig(BaseModel):
    url: str = "redis://localhost:6379/0"
    prefix: str = "uni_ai:"


class MilvusConfig(BaseModel):
    host: str = "localhost"
    port: int = 19530
    collection: str = "knowledge_chunks"


class ESConfig(BaseModel):
    hosts: list[str] = Field(default_factory=lambda: ["http://localhost:9200"])
    index_prefix: str = "uni_ai_"
    analyzer: str = "ik_max_word"


class MQConfig(BaseModel):
    url: str = "amqp://guest:guest@localhost:5672/"
    parse_queue: str = "doc_parse_queue"


class LLMModelEntry(BaseModel):
    name: str
    api_key: str = ""
    api_base: str | None = None


class LLMConfig(BaseModel):
    provider: str = "litellm"
    default_model: str = "deepseek/deepseek-chat"
    models: list[LLMModelEntry] = Field(default_factory=list)
    temperature: float = 0.1
    max_tokens: int = 4096


class DashScopeConfig(BaseModel):
    api_key: str = ""
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    default_model: str = "qwen-max"
    embedding_model: str = "text-embedding-v3"


class EmbeddingConfig(BaseModel):
    model: str = "text-embedding-v3"
    device: str = "cpu"
    batch_size: int = 32
    dimension: int = 1024
    normalize: bool = True
    cache_ttl: int = 604800  # 7 days in seconds


class RerankerConfig(BaseModel):
    model: str = "gte-rerank"
    device: str = "cpu"
    top_k: int = 10


class PDFParsingConfig(BaseModel):
    primary_engine: str = "pymupdf"
    fallback_engine: str = ""
    quality_threshold: float = 0.6
    max_pages: int = 500
    enable_ocr: bool = True
    enable_table: bool = True
    enable_formula: bool = True


class ChunkParsingConfig(BaseModel):
    size: int = 512
    overlap: int = 64
    min_size: int = 128
    max_size: int = 1024
    parent_size: int = 2048


class ParsingConfig(BaseModel):
    pdf: PDFParsingConfig = Field(default_factory=PDFParsingConfig)
    chunk: ChunkParsingConfig = Field(default_factory=ChunkParsingConfig)


class RetrievalConfig(BaseModel):
    mode: str = "hybrid"
    vector_weight: float = 0.4
    sparse_weight: float = 0.3
    bm25_weight: float = 0.3
    top_k: int = 10
    rerank: bool = True
    enable_hyde: bool = True


class QAConfig(BaseModel):
    default_mode: str = "auto"
    max_iterations: int = 3
    stream: bool = True
    max_chat_history: int = 10


# ---------------------------------------------------------------------------
# Main Settings
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    """Application settings — assembled from YAML + env vars."""

    app_name: str = "uni-ai-python"
    app_version: str = "1.0.0"
    debug: bool = False

    server: ServerConfig = Field(default_factory=ServerConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    minio: MinioConfig = Field(default_factory=MinioConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    milvus: MilvusConfig = Field(default_factory=MilvusConfig)
    elasticsearch: ESConfig = Field(default_factory=ESConfig)
    mq: MQConfig = Field(default_factory=MQConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    dashscope: DashScopeConfig = Field(default_factory=DashScopeConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    parsing: ParsingConfig = Field(default_factory=ParsingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    qa: QAConfig = Field(default_factory=QAConfig)

    model_config = {
        "env_prefix": "UNI_AI_",
        "env_nested_delimiter": "__",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Prioritize environment variables over YAML kwargs.

        `get_settings()` injects YAML via init kwargs. Without this override,
        those kwargs would mask .env / environment overrides.
        """
        return (
            env_settings,
            dotenv_settings,
            init_settings,
            file_secret_settings,
        )


# ---------------------------------------------------------------------------
# YAML loading & merging
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base dict."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml_config() -> dict:
    """Load and merge YAML configuration files."""
    config: dict = {}

    # Base config
    base_path = CONFIGS_DIR / "config.yaml"
    if base_path.exists():
        with open(base_path, encoding="utf-8") as f:
            base = yaml.safe_load(f) or {}
            config = _deep_merge(config, base)

    # Environment-specific overlay
    env = os.getenv("UNI_AI_ENV", "dev")
    env_path = CONFIGS_DIR / f"config.{env}.yaml"
    if env_path.exists():
        with open(env_path, encoding="utf-8") as f:
            env_config = yaml.safe_load(f) or {}
            config = _deep_merge(config, env_config)

    return config


@lru_cache
def get_settings() -> Settings:
    """
    Global singleton for application settings.

    Priority (highest to lowest):
    1. Environment variables (UNI_AI__*)
    2. config.{env}.yaml overrides
    3. config.yaml defaults
    4. Pydantic field defaults
    """
    yaml_data = _load_yaml_config()
    return Settings(**yaml_data)
