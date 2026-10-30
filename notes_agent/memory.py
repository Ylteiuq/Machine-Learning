import json
from pathlib import Path

SESSION_FILE = Path(__file__).resolve().parent / "session.json"


def save_messages(messages: list[dict]) -> None:
    """保存完整会话历史。"""

    with open(SESSION_FILE, "w", encoding="utf-8") as file:
        json.dump(messages, file, ensure_ascii=False, indent=2)


def load_messages() -> list[dict]:
    """加载会话历史；首次运行时返回空列表。"""

    if not SESSION_FILE.exists():
        return []

    with open(SESSION_FILE, "r", encoding="utf-8") as file:
        memory = json.load(file)

    return memory
