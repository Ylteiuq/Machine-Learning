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
    query = query.lower()

    result = []
    
    for note in Notes:
        if note["title"].lower().find(query) or note["content"].lower().find(query) is not -1:
            result.append(note)

    return result

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_notes",
            "description": "按关键词搜索本地笔记的标题和正文，返回匹配笔记的编号和标题。",
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
    }
]

TOOL_HANDLERS = {
    "search_notes": search_notes,
}
