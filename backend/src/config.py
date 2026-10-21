"""Central runtime configuration, loaded from environment variables / .env."""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Vision / LLM providers
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    vision_provider: str = "mock"  # mock | anthropic | openai | ollama
    vision_model: str = "claude-sonnet-5"

    # Ollama (free, local, open-source vision models - no API key required).
    # Install: https://ollama.com, then `ollama pull moondream` (or another
    # vision-capable model) before setting VISION_PROVIDER=ollama.
    ollama_base_url: str = "http://localhost:11434"
    ollama_vision_model: str = "moondream"

    # Vector DB. If chroma_host is set, connect to a standalone Chroma
    # server (docker-compose service) via HttpClient; otherwise fall back
    # to an embedded PersistentClient writing to chroma_persist_dir - handy
    # for `streamlit run` / `pytest` outside Docker.
    chroma_host: str | None = None
    chroma_port: int = 8000
    chroma_persist_dir: str = "./vector_store"
    chroma_collection_name: str = "spare_parts_catalog"
    embedding_model: str = "all-MiniLM-L6-v2"

    # Backend
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000
    upload_dir: str = "./uploads"
    max_upload_mb: int = 500

    # Frontend
    backend_base_url: str = "http://localhost:8000"
    streamlit_port: int = 8501

    # Verification checkpoint thresholds
    blur_variance_threshold: float = 100.0
    brightness_min: float = 25.0
    brightness_max: float = 230.0
    min_resolution_width: int = 480
    ocr_confidence_threshold: float = 0.80
    rag_similarity_threshold: float = 0.70
    max_structured_output_retries: int = 3
    keyframe_sample_interval_sec: float = 1.0
    max_keyframes: int = 24

    @property
    def upload_path(self) -> Path:
        p = Path(self.upload_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def chroma_path(self) -> Path:
        p = Path(self.chroma_persist_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p


settings = Settings()
