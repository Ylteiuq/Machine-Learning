import memory

import json
from pathlib import Path

NOTES_FILE = Path(__file__).resolve().parent / "notes.json"

def load_notes() -> list[dict]:
    with open(NOTES_FILE, "r", encoding="utf-8") as file:
        return json.load(file)

def search_notes(query: str) -> list[dict]:
    query = query.strip().lower()

    result = []

    Notes = load_notes()

    for note in Notes:
        if query in note["title"].lower() or query in note["content"].lower():
            result.append({"id": note["id"], "title": note["title"]})

    return result

def read_notes(note_id: str) -> dict:
    Notes = load_notes()
    for note in Notes:
        if note["id"] == note_id:
            return note.copy()

    raise ValueError(f"未查询到对应的id: {note_id}")

def save_notes(title: str, content: str, *, execution_id: str) -> dict:
    Notes = load_notes()

    for note in Notes:
        if note.get("execution_id") == execution_id:
            return note

    num_id = 'n' + str(max((int(note["id"][1:]) for note in Notes), default=0) + 1)

    Notes.append({
        "id": num_id,
        "title": title,
        "content": content,
        "execution_id": execution_id,
    })

    # 笔记和执行标识一起提交，恢复后才能识别已经完成的写入。
    temporary_file = NOTES_FILE.with_suffix(".tmp")
    with open(temporary_file, "w", encoding="utf-8") as file:
        json.dump(Notes, file, ensure_ascii=False, indent=2)
    temporary_file.replace(NOTES_FILE)

    return Notes[-1]

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_notes",
            "description": "按关键词搜索本地笔记的标题和正文，只返回编号和标题；需要正文时继续调用 read_notes。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词，例如 Attention",
                    }
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_notes",
            "description": "根据搜索结果中的笔记编号读取一篇本地笔记，返回编号、标题和完整正文。",
            "parameters": {
                "type": "object",
                "properties": {
                    "note_id": {
                        "type": "string",
                        "description": "搜索对应id，如 n1",
                    }
                },
                "required": ["note_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remember_preference",
            "description": "当用户明确要求记住某项偏好时使用；保存偏好，同名偏好覆盖原值。",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "偏好名称，例如回答长度",
                    },
                    "value": {
                        "type": "string",
                        "description": "偏好内容，例如尽量简短",
                    },
                },
                "required": ["key", "value"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_notes",
            "description": "写入新的笔记",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "标题的内容", 
                    },
                    "content": {
                        "type": "string",
                        "description": "具体的笔记内容"
                    }
                },
                "required": ["title", "content"],
                "additionalProperties": False,
            },
        },
    }
]

TOOL_HANDLERS = {
    "search_notes": search_notes,
    "read_notes": read_notes,
    "remember_preference": memory.remember_preference,
    "save_notes": save_notes,
}
