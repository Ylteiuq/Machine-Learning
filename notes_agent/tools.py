Notes = [
    {
        "id": "n1",
        "title": "Attention",
        "content": "Q 表示查询，K 表示键，V 表示值。"
    },
    {
        "id": "n2",
        "title": "Transformer结构",
        "content": "Transformer 使用多头 Attention 和前馈网络。"
    },
    {
        "id": "n3",
        "title": "Python异常处理",
        "content": "使用try和except处理异常"
    }
]

def search_notes(query: str) -> list[dict]:
    query = query.strip().lower()

    result = []

    for note in Notes:
        if query in note["title"].lower() or query in note["content"].lower():
            result.append({"id": note["id"], "title": note["title"]})

    return result

def read_notes(note_id: str) -> dict:
    for note in Notes:
        if note["id"] == note_id:
            return note.copy()

    raise ValueError(f"未查询到对应的id: {note_id}")

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
    }
]

TOOL_HANDLERS = {
    "search_notes": search_notes,
    "read_notes": read_notes,
}
