import json
from pathlib import Path

SESSION_FILE = Path(__file__).resolve().parent / "session.json"
PREFERENCES_FILE = Path(__file__).resolve().parent / "preferences.json"


def save_messages(messages: list[dict], summary: str = "", summarized_until: int = 0) -> None:
    """一起保存完整历史、摘要及其覆盖位置。"""

    session_data = {
        "messages": messages,
        "summary": summary,
        "summarized_until": summarized_until
    }

    temporary_file = SESSION_FILE.with_suffix(".tmp")
    with open(temporary_file, "w", encoding="utf-8") as file:
        json.dump(session_data, file, ensure_ascii=False, indent=2)
    temporary_file.replace(SESSION_FILE)


def load_messages() -> dict:
    """加载会话；兼容此前只保存消息列表的文件。"""

    if not SESSION_FILE.exists():
        return {"messages": [], "summary": "", "summarized_until": 0}

    with open(SESSION_FILE, "r", encoding="utf-8") as file:
        memory = json.load(file)

    if isinstance(memory, list):
        return {"messages": memory, "summary": "", "summarized_until": 0}
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
