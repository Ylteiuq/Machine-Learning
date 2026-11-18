import json
import sys
from contextlib import contextmanager
from pathlib import Path

from anyio.from_thread import start_blocking_portal
from mcp import Client, StdioServerParameters

from .registry import Tool


class MCPConnection:
    def __init__(self, portal, client):
        self.portal = portal
        self.client = client

    def register_tools(self, registry, parallel_names=()) -> None:
        discovered = self.portal.call(self.client.list_tools)
        for tool in discovered.tools:
            registry.register(Tool(
                tool.name, tool.description or "", tool.input_schema,
                lambda arguments, context, name=tool.name: self.call_tool(name, arguments),
                parallel=tool.name in parallel_names, source="mcp",
            ))

    def call_tool(self, name: str, arguments: dict):
        result = self.portal.call(self.client.call_tool, name, arguments)
        texts = [block.text for block in result.content if block.type == "text"]
        if result.is_error:
            raise ValueError("\n".join(texts) or f"MCP 工具执行失败：{name}")
        if result.structured_content is not None:
            data = result.structured_content
            return data["result"] if set(data) == {"result"} else data
        if len(texts) == 1:
            try:
                return json.loads(texts[0])
            except json.JSONDecodeError:
                return texts[0]
        return texts


@contextmanager
def connect_mcp(server: StdioServerParameters):
    with start_blocking_portal() as portal:
        with portal.wrap_async_context_manager(Client(server)) as client:
            yield MCPConnection(portal, client)


def file_server(workspace: Path) -> StdioServerParameters:
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "notes_agent_rebuild.files_mcp_server", "--workspace", str(workspace.resolve())],
        cwd=str(Path(__file__).resolve().parents[2]),
    )
