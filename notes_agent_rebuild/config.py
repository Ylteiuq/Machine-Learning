import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent


@dataclass
class Settings:
    workspace: Path = BASE_DIR / "workspace"
    database: Path = BASE_DIR / "data" / "agent.db"
    api_url: str = ""
    api_key: str = ""
    model: str = ""
    context_window: int = 128000

    @classmethod
    def load(cls):
        load_dotenv(BASE_DIR / ".env")
        return cls(
            api_url=os.getenv("LLM_API_URL", ""),
            api_key=os.getenv("LLM_API_KEY", ""),
            model=os.getenv("LLM_MODEL", ""),
            context_window=int(os.getenv("CONTEXT_WINDOW", "128000"))
        )
