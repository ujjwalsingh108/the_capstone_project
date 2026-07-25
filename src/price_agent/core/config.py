from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Intelligent Price Prediction Agent"
    app_env: str = "development"
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    openai_api_key: str = ""
    hf_token: str = ""
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "price_agent_documents"
    model_name: str = "base-reasoning-model"
    embedding_model: str = "text-embedding-3-small"
    rag_top_k: int = 5
    cloud_provider: str = "local"
    artifact_dir: str = "artifacts"
    data_dir: str = "data"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
