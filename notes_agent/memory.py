import json
from pathlib import Path

SESSION_FILE = Path(__file__).resolve().parent / "session.json"
PREFERENCES_FILE = Path(__file__).resolve().parent / "preferences.json"


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


def load_preferences() -> dict:
    """读取已保存的用户偏好；首次运行时返回空字典。"""
    try:
        with open(PREFERENCES_FILE, "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        return {}


def remember_preference(key: str, value: str) -> dict:
    """保存一项偏好，同名偏好覆盖原值。"""
    preferences = load_preferences()
    preferences[key] = value

    with open(PREFERENCES_FILE, "w", encoding="utf-8") as file:
        json.dump(preferences, file, ensure_ascii=False, indent=2)

    return {key: value}
