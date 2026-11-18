"""文件工具的 MCP 入口；与本地调用共用 FileTools，不维护第二份业务实现。"""

import argparse
from functools import wraps
from pathlib import Path

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from .tools.files import FileTools


def mcp_handler(function):
    @wraps(function)
    def call(**arguments):
        try:
            return function(**arguments)
        except (ValueError, OSError) as error:
            raise ToolError(str(error)) from error
    return call


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    files = FileTools(args.workspace)
    server = MCPServer("文件工具服务")
    for function in (files.list_files, files.read_file, files.search_files, files.write_file):
        server.add_tool(mcp_handler(function))
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
