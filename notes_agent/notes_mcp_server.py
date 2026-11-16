"""通过 MCP 暴露笔记搜索和读取，业务逻辑复用现有模块。"""

from mcp.server import MCPServer

from retrieval import search_notes as search_local_notes
from notes_sqlite import get_note_by_id, make_sqlite


mcp = MCPServer("学习笔记服务")


@mcp.tool()
def search_notes(query: str, top_k: int = 5) -> list[dict]:
    """搜索笔记，多个关键词用空格分隔。返回编号、标题和分数；正文用 read_notes 读取。"""
    return search_local_notes(query, top_k)


@mcp.tool()
def read_notes(note_id: str) -> dict:
    """根据笔记编号（例如 n1）读取标题和完整正文。"""
    return get_note_by_id(note_id)


def main() -> None:
    make_sqlite()
    # stdio 的标准输出用于协议通信，不在这里 print 普通日志。
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
