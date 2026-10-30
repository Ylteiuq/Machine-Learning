import os
import json
import urllib.request
from enum import Enum, auto
from pathlib import Path

from dotenv import load_dotenv

import tools

load_dotenv(Path(__file__).resolve().parent / ".env")

def request_llm(messages: list[dict], tools=None) -> dict:
    api_key = os.getenv("LLM_API_KEY")

    if not api_key:
        raise RuntimeError("缺少 LLM_API_KEY,请在终端设置api_key")

    api_url = os.environ["LLM_API_URL"]
    model = os.environ["LLM_MODEL"]

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "thinking": {"type": "disabled"},
        "max_tokens": 512,
    }

    # 先补齐请求体，再编码；编码后的 body 不会随 payload 的修改而变化。
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    body = json.dumps(payload).encode("utf-8")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    request = urllib.request.Request(
        api_url,
        data=body,
        headers=headers,
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)

def find_tool_schemas(tool_name: str, tool_schemas: list[dict]) -> dict:
    """返回工具原始的参数描述，保持与发送给模型的字段名一致。"""
    for tool in tool_schemas:
        if tool_name == tool["function"]["name"]:
            return tool["function"]["parameters"]

    raise ValueError(f"找不到对应tool: {tool_name}")

def execute_tool(tool_call: dict) -> dict:
    """执行一次工具请求，返回可追加到 messages 的工具消息。"""

    tool_id = tool_call["id"]
    tool_name = tool_call["function"]["name"]
    tool_arguments = tool_call["function"]["arguments"]

    try:
        if tool_name not in tools.TOOL_HANDLERS:
            raise ValueError(f"未知工具：{tool_name}")
        if not isinstance(tool_arguments, str):
            raise ValueError("工具参数必须是 JSON 字符串")

        parameters = find_tool_schemas(tool_name, tools.TOOL_SCHEMAS)
        arguments = json.loads(tool_arguments)
        if not isinstance(arguments, dict):
            raise ValueError("工具参数必须解析为一个对象")

        required = parameters.get("required", [])
        properties = parameters.get("properties", {})
        missing_fields = set(required) - set(arguments)
        if missing_fields:
            raise ValueError(f"缺少必填参数：{', '.join(sorted(missing_fields))}")

        if parameters.get("additionalProperties", True) is False:
            extra_fields = set(arguments) - set(properties)
            if extra_fields:
                raise ValueError(f"存在不允许的参数：{', '.join(sorted(extra_fields))}")

        # 这里只校验当前两个工具用到的字符串参数，并非完整的 JSON Schema 校验器。
        for field, value in arguments.items():
            if properties.get(field, {}).get("type") == "string":
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"参数 {field} 必须是非空字符串")

        # 实际调用也放在 try 中，read_notes 抛出的业务错误才能回传给模型。
        func = tools.TOOL_HANDLERS[tool_name]
        result = {
            "ok": True,
            "data": func(**arguments),
        }
    except json.JSONDecodeError:
        result = {"ok": False, "error": "工具参数不是有效的 JSON"}
    except ValueError as error:
        result = {"ok": False, "error": str(error)}

    # 成功与失败都作为工具结果回传，并关联到原始调用编号。
    return {
        "role": "tool",
        "tool_call_id": tool_id,
        "content": json.dumps(result, ensure_ascii=False),
    }

class AgentStatus(Enum):
    COMPLETE = auto()
    MAX_STEPS = auto()
    INCOMPLETE = auto()

def run_agent(messages: list[dict], max_steps: int = 6) -> tuple[AgentStatus, list[dict]]:
    """维护传入的消息历史；max_steps 限制模型调用次数，而非工具调用次数。"""
    if type(max_steps) is not int or max_steps < 1:
        raise ValueError("max_steps 必须是正整数")

    try:
        for step in range(1, max_steps + 1):
            response = request_llm(messages, tools.TOOL_SCHEMAS)
            choice = response["choices"][0]
            message = choice["message"]
            finish_reason = choice["finish_reason"]

            messages.append(message)

            print(f"第 {step} 次模型调用：", finish_reason, response.get("usage", {}))

            # 被截断或异常终止的响应不能算完成，也不能执行其中可能残缺的工具请求。
            if finish_reason not in {"stop", "tool_calls"}:
                print(f"模型未正常完成生成：{finish_reason}")
                return AgentStatus.INCOMPLETE, messages

            tool_calls = message.get("tool_calls") or []

            if not tool_calls:
                if finish_reason == "stop":
                    return AgentStatus.COMPLETE, messages
                print("模型声明要调用工具，但没有返回工具请求")
                return AgentStatus.INCOMPLETE, messages

            if finish_reason != "tool_calls":
                print("模型的停止原因与工具请求不一致")
                return AgentStatus.INCOMPLETE, messages

            for tool_call in tool_calls:
                print("调用工具：", tool_call["function"]["name"], tool_call["function"]["arguments"])
                messages.append(execute_tool(tool_call))

        return AgentStatus.MAX_STEPS, messages
    except Exception as error:
        # 工具的预期业务错误已转成 tool 消息；其余运行失败在任务边界终止。
        print(f"Agent 运行失败（{type(error).__name__}）：{error}")
        return AgentStatus.INCOMPLETE, messages

def main() -> None:
    messages = [
        {
            "role": "system",
            "content": (
                "你是学习笔记助手，使用简洁的中文回答。先搜索本地笔记获取编号；"
                "回答笔记的具体内容前，使用 read_notes 读取正文。"
                "依据工具结果回答并标注笔记编号；没有找到时明确说明。"
            )
        },
        {
            "role": "user",
            "content": "找出关于 Attention 的笔记，读取正文，解释 Q、K、V 分别是什么，并标注笔记编号。"
        }
    ]
    status, messages = run_agent(messages, 6)

    if status == AgentStatus.COMPLETE:
        print("Agent 正常完成")
    elif status == AgentStatus.INCOMPLETE:
        print("Agent 生成未完成或运行失败")
    elif status == AgentStatus.MAX_STEPS:
        print("达到最大模型调用次数，Agent 已停止，任务尚未完成")

    for message in messages:
        if message.get("tool_calls"):
            continue  # 工具请求已在执行时打印。
        print(message["role"], message["content"])

if __name__ == "__main__":
    main()
