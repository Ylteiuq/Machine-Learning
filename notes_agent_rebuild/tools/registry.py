import json
from dataclasses import dataclass
from typing import Callable

from jsonschema import ValidationError, validate

from ..agent.state import ToolContext


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    handler: Callable
    parallel: bool = False
    source: str = "local"

    def schema(self) -> dict:
        return {"type": "function", "function": {
            "name": self.name, "description": self.description, "parameters": self.parameters,
        }}


class ToolRegistry:
    """工具描述、执行入口和并行属性保存在同一个注册项中。"""

    def __init__(self):
        self.tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self.tools:
            raise ValueError(f"工具重复注册：{tool.name}")
        self.tools[tool.name] = tool

    def schemas(self, allowed: list[str] | None = None) -> list[dict]:
        return [tool.schema() for tool in self.tools.values()
                if allowed is None or tool.name in allowed]

    def batch(self, calls: list[dict]) -> list[dict]:
        batch = []
        for call in calls:
            tool = self.tools.get(call["function"]["name"])
            if tool is None or not tool.parallel:
                break
            batch.append(call)
        return batch or calls[:1]

    def execute(self, call: dict, context: ToolContext, allowed=None) -> dict:
        name = call["function"]["name"]
        try:
            if allowed is not None and name not in allowed:
                raise ValueError(f"当前 Agent 不允许调用：{name}")
            if name not in self.tools:
                raise ValueError(f"未知工具：{name}")
            tool = self.tools[name]
            arguments = json.loads(call["function"]["arguments"])
            validate(arguments, tool.parameters)
            result = {"ok": True, "data": tool.handler(arguments, context)}
        except ValidationError as error:
            result = {"ok": False, "error": error.message}
        except (ValueError, OSError) as error:
            result = {"ok": False, "error": str(error)}
        return {"role": "tool", "tool_call_id": call["id"],
                "content": json.dumps(result, ensure_ascii=False)}
