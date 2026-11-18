from pathlib import Path

from .registry import Tool


class FileTools:
    def __init__(self, workspace: Path):
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)

    def resolve(self, path: str) -> Path:
        target = (self.workspace / path).resolve()
        if not target.is_relative_to(self.workspace):
            raise ValueError("文件路径必须位于工作目录内")
        return target

    def list_files(self, pattern: str = "**/*", limit: int = 100) -> list[str]:
        """列出工作目录内的文件路径，支持 glob，例如 **/*.py；默认不列出隐藏文件。"""
        if limit < 1:
            raise ValueError("limit 必须大于 0")
        paths = []
        for path in sorted(self.workspace.glob(pattern)):
            relative = path.relative_to(self.workspace)
            if any(part.startswith('.') for part in relative.parts):
                continue
            if path.is_file() and path.resolve().is_relative_to(self.workspace):
                paths.append(relative.as_posix())
                if len(paths) >= limit:
                    break
        return paths

    def read_file(self, path: str, start_line: int = 1, max_lines: int = 200) -> dict:
        """读取 UTF-8 文本。行号从 1 开始；长文件可以按行继续读取。"""
        if start_line < 1 or max_lines < 1:
            raise ValueError("start_line 和 max_lines 必须大于 0")
        lines = self.resolve(path).read_text(encoding="utf-8").splitlines(keepends=True)
        selected = lines[start_line - 1:start_line - 1 + max_lines]
        end = min(start_line - 1 + len(selected), len(lines))
        return {"path": path, "content": ''.join(selected), "start_line": start_line,
                "total_lines": len(lines), "next_line": end + 1 if end < len(lines) else None}

    def write_file(self, path: str, content: str) -> dict:
        """写入完整 UTF-8 文本，已有文件会被覆盖；修改已有文件前先读取内容。"""
        target = self.resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return {"path": path, "characters": len(content)}

    def search_files(self, query: str, pattern: str = "**/*", limit: int = 30) -> list[dict]:
        """在文本文件中按子串查找内容，忽略大小写；返回路径、行号和命中行。"""
        if not query.strip() or limit < 1:
            raise ValueError("query 不能为空，limit 必须大于 0")
        results = []
        for path in self.list_files(pattern, limit=10000):
            try:
                lines = self.resolve(path).read_text(encoding="utf-8").splitlines()
            except UnicodeDecodeError:
                continue
            for number, line in enumerate(lines, start=1):
                if query.casefold() in line.casefold():
                    results.append({"path": path, "line": number, "text": line})
                    if len(results) >= limit:
                        return results
        return results


def file_tools(workspace: Path) -> list[Tool]:
    files = FileTools(workspace)
    definitions = [
        (files.list_files, True, [], {"pattern": {"type": "string"}, "limit": {"type": "integer", "minimum": 1}}),
        (files.read_file, True, ["path"], {
            "path": {"type": "string"}, "start_line": {"type": "integer", "minimum": 1},
            "max_lines": {"type": "integer", "minimum": 1},
        }),
        (files.write_file, False, ["path", "content"], {
            "path": {"type": "string"}, "content": {"type": "string"},
        }),
        (files.search_files, True, ["query"], {
            "query": {"type": "string", "minLength": 1}, "pattern": {"type": "string"},
            "limit": {"type": "integer", "minimum": 1},
        }),
    ]
    tools = []
    for function, parallel, required, properties in definitions:
        tools.append(Tool(
            function.__name__, function.__doc__,
            {"type": "object", "properties": properties, "required": required, "additionalProperties": False},
            lambda arguments, context, function=function: function(**arguments), parallel,
        ))
    return tools
