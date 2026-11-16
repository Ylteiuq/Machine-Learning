"""把异步 MCP 客户端接到当前同步 Agent，连接在整个对话期间复用。"""

import json
import sys
from contextlib import contextmanager
from pathlib import Path

from anyio.from_thread import start_blocking_portal
from mcp import Client, StdioServerParameters


class NotesMCPClient:
    def __init__(self, portal, client):
        self.portal = portal
        self.client = client
        discovered = portal.call(client.list_tools)
        self.tool_names = {tool.name for tool in discovered.tools}
        # MCP 的 input_schema 转成聊天模型 Function Calling 的 parameters。
        self.tool_schemas = [
            {"type": "function", "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.input_schema,
            }}
            for tool in discovered.tools
        ]

    def call_tool(self, name: str, arguments: dict):
        result = self.portal.call(self.client.call_tool, name, arguments)
        texts = [block.text for block in result.content if block.type == "text"]
        if result.is_error:
            raise ValueError("\n".join(texts) or f"MCP 工具执行失败：{name}")

        if result.structured_content is not None:
            data = result.structured_content
            # SDK 将搜索函数返回的列表包装成 {"result": [...]}。
            return data["result"] if set(data) == {"result"} else data

        # 当前 read_notes 返回的字典由 SDK 编码在文本内容中。
        if len(texts) == 1:
            try:
                return json.loads(texts[0])
            except json.JSONDecodeError:
                return texts[0]
        return texts


@contextmanager
def connect_notes_server(server: StdioServerParameters = None):
    if server is None:
        server = StdioServerParameters(
            command=sys.executable,
            args=[str(Path(__file__).with_name("notes_mcp_server.py"))],
        )
    # portal 在后台线程运行事件循环；主线程和工具线程通过 call 提交请求。
    with start_blocking_portal() as portal:
        with portal.wrap_async_context_manager(Client(server)) as client:
            yield NotesMCPClient(portal, client)
