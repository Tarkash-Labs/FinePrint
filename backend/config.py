import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

_ENV_PATH = Path(__file__).resolve().with_name(".env")
load_dotenv(dotenv_path=_ENV_PATH)

def _parse_csv(value: str) -> tuple[str, ...]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    return tuple(items) if items else ("*",)

@dataclass(frozen=True)
class Settings:
    api_key: str = os.getenv("GEMMA_API_KEY", "")
    e4b_model: str = os.getenv("GEMMA_E4B_MODEL", "gemma-4-26b-a4b-it")
    moe_model: str = os.getenv("GEMMA_MOE_MODEL", "gemma-4-31b-it")
    dense_model: str = os.getenv("GEMMA_DENSE_MODEL", "gemma-4-31b-it")
    max_contract_chars: int = int(os.getenv("MAX_CONTRACT_CHARS", "120000"))
    summary_trigger_chars: int = int(os.getenv("SUMMARY_TRIGGER_CHARS", "20000"))
    summary_chunk_chars: int = int(os.getenv("SUMMARY_CHUNK_CHARS", "8000"))
    max_upload_bytes: int = int(os.getenv("MAX_UPLOAD_BYTES", "10485760"))
    cors_allow_origins: tuple[str, ...] = _parse_csv(os.getenv("CORS_ALLOW_ORIGINS", "*"))
    cors_allow_credentials: bool = os.getenv("CORS_ALLOW_CREDENTIALS", "false").lower() == "true"

settings = Settings()