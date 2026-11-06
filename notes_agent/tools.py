import memory
import notes_sqlite

TOOL_PARALLEL = {"search_notes", "read_notes"}

def load_notes() -> list[dict]:
    return notes_sqlite.list_notes()

def search_notes(query: str, top_k: int = 5) -> list[dict]:
    return notes_sqlite.search_notes(query, top_k)

def read_notes(note_id: str) -> dict:
    return notes_sqlite.get_note_by_id(note_id)

def save_notes(title: str, content: str, *, execution_id: str) -> dict:
    return notes_sqlite.create_note(title, content, execution_id)

def update_notes(note_id: str, title: str, content: str) -> dict:
    return notes_sqlite.update_note(note_id, title, content)

def delete_notes(note_id: str) -> bool:
    return notes_sqlite.delete_note(note_id)

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_notes",
            "description": "搜索笔记标题和正文，多个关键词用空格分隔，命中任一词即可。返回编号、标题和排序分数（短词补充结果为 null）；需要正文时调用 read_notes。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词，例如 Transformer Attention",
                    },
                    "top_k": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "最多返回多少条，默认 5",
                    },
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
    },
    {
        "type": "function",
        "function": {
            "name": "update_notes",
            "description": "按编号替换已有笔记的完整标题和正文。用户要求修改时使用；需要保留原有内容时先读取笔记，再提交完整的新内容。",
            "parameters": {
                "type": "object",
                "properties": {
                    "note_id": {
                        "type": "string",
                        "description": "笔记的id"
                    },
                    "title": {
                        "type": "string",
                        "description": "笔记的标题"
                    },
                    "content": {
                        "type": "string",
                        "description": "笔记的内容"
                    }
                },
                "required": ["note_id", "title", "content"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_notes",
            "description": "用户明确要求删除时，按编号永久删除一篇笔记；成功返回 true，编号不存在时返回错误。",
            "parameters": {
                "type": "object",
                "properties": {
                    "note_id": {
                        "type": "string",
                        "description": "笔记的id"
                    }
                },
                "required": ["note_id"],
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
    "update_notes": update_notes,
    "delete_notes": delete_notes,
}
