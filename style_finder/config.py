"""
Runtime configuration loaded from environment variables.

Secrets never live in source. Copy `.env.example` to `.env` and fill in values.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return default if raw is None else float(raw)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return default if raw is None else int(raw)


def _env_csv_floats(name: str, default: tuple[float, float, float]) -> tuple[float, float, float]:
    raw = os.getenv(name)
    if not raw:
        return default
    values = tuple(float(part.strip()) for part in raw.split(","))
    if len(values) != 3:
        raise ValueError(f"{name} must contain exactly 3 comma-separated floats")
    return values  # type: ignore[return-value]


@dataclass(frozen=True)
class Settings:
    """Immutable application settings resolved at process start."""

    llama_model_id: str = os.getenv(
        "LLAMA_MODEL_ID",
        "meta-llama/llama-4-maverick-17b-128e-instruct-fp8",
    )
    watsonx_project_id: str = os.getenv("WATSONX_PROJECT_ID", "skills-network")
    watsonx_region: str = os.getenv("WATSONX_REGION", "us-south")
    watsonx_api_key: str | None = os.getenv("WATSONX_APIKEY") or os.getenv("WATSONX_API_KEY")

    llm_temperature: float = _env_float("LLM_TEMPERATURE", 0.2)
    llm_top_p: float = _env_float("LLM_TOP_P", 0.6)
    llm_max_tokens: int = _env_int("LLM_MAX_TOKENS", 2000)

    image_size: tuple[int, int] = field(
        default_factory=lambda: (
            _env_int("IMAGE_WIDTH", 224),
            _env_int("IMAGE_HEIGHT", 224),
        )
    )
    normalization_mean: tuple[float, float, float] = field(
        default_factory=lambda: _env_csv_floats(
            "NORMALIZATION_MEAN", (0.485, 0.456, 0.406)
        )
    )
    normalization_std: tuple[float, float, float] = field(
        default_factory=lambda: _env_csv_floats(
            "NORMALIZATION_STD", (0.229, 0.224, 0.225)
        )
    )

    similarity_threshold: float = _env_float("SIMILARITY_THRESHOLD", 0.8)
    default_alternatives_count: int = _env_int("DEFAULT_ALTERNATIVES_COUNT", 5)

    dataset_path: Path = Path(
        os.getenv("DATASET_PATH", str(PROJECT_ROOT / "data" / "swift-style-embeddings.pkl"))
    )
    chroma_path: Path = Path(
        os.getenv("CHROMA_PATH", str(PROJECT_ROOT / "data" / "chroma"))
    )
    chroma_collection: str = os.getenv("CHROMA_COLLECTION", "fashion_outfits")
    examples_dir: Path = Path(os.getenv("EXAMPLES_DIR", str(PROJECT_ROOT / "examples")))

    server_name: str = os.getenv("SERVER_NAME", "127.0.0.1")
    server_port: int = _env_int("SERVER_PORT", 5000)
    share: bool = _env_bool("GRADIO_SHARE", False)

    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    @property
    def watsonx_url(self) -> str:
        return f"https://{self.watsonx_region}.ml.cloud.ibm.com"

    def require_api_key(self) -> str:
        if not self.watsonx_api_key:
            raise RuntimeError(
                "Missing WATSONX_APIKEY. Copy .env.example to .env and set your IBM watsonx.ai key."
            )
        return self.watsonx_api_key

    def example_images(self) -> list[str]:
        if not self.examples_dir.exists():
            return []
        return sorted(
            str(path)
            for path in self.examples_dir.iterdir()
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        )


settings = Settings()
