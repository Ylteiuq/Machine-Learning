import os
import json
import urllib.request
from enum import Enum, auto
from pathlib import Path
import uuid
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv

import tools
import memory
import context
import checkpoint

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
        "max_tokens": 2048,
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

def execute_tool(tool_call: dict, run_id: str) -> dict:
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
        if tool_name == "save_notes":
            # 内部执行标识由运行时注入，不由模型提供。
            arguments["execution_id"] = f"{run_id}:{tool_id}"
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

def execute_tool_batch(tool_calls: list[dict], run_id: str) -> list[dict]:
    """执行调度好的批次：单个工具直接执行，多个读取工具最多两路并行。"""
    if len(tool_calls) == 1:
        return [execute_tool(tool_calls[0], run_id)]

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(execute_tool, call, run_id) for call in tool_calls]
        # 先全部提交，再按请求顺序收集；结果顺序不受完成先后影响。
        return [future.result() for future in futures]


class AgentStatus(Enum):
    COMPLETE = auto()
    MAX_STEPS = auto()
    INCOMPLETE = auto()

def step(state: dict) -> None:
    """原地推进一次模型请求或一个工具批次；终止后不再推进。"""
    messages = state["messages"]
    if state["phase"] == "model":
        state["model_calls"] += 1
        response = request_llm(context.build_context(messages), tools.TOOL_SCHEMAS)
        choice = response["choices"][0]
        message = choice["message"]
        finish_reason = choice["finish_reason"]

        messages.append(message)
        print(f"第 {state['model_calls']} 次模型调用：", finish_reason, response.get("usage", {}))

        # 被截断或异常终止的响应不能算完成，也不能执行其中可能残缺的工具请求。
        if finish_reason not in {"stop", "tool_calls"}:
            print(f"模型未正常完成生成：{finish_reason}")
            state["phase"] = "failed"
            return

        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            if finish_reason == "stop":
                state["phase"] = "done"
                return
            print("模型声明要调用工具，但没有返回工具请求")
            state["phase"] = "failed"
            return

        if finish_reason != "tool_calls":
            print("模型的停止原因与工具请求不一致")
            state["phase"] = "failed"
            return

        state["phase"] = "tools"
        # 队列会被 pop；复制列表，避免同时删除历史中的工具请求。
        state["pending_tools"] = tool_calls.copy()

    elif state["phase"] == "tools":
        batch = []
        for tool_call in state["pending_tools"]:
            if tool_call["function"]["name"] not in tools.TOOL_PARALLEL:
                break
            batch.append(tool_call)
        if not batch:
            batch = state["pending_tools"][:1]

        for tool_call in batch:
            print("调用工具：", tool_call["function"]["name"], tool_call["function"]["arguments"])
        results = execute_tool_batch(batch, state["run_id"])
        # 工作线程不修改状态；整批结果返回后由主线程统一提交。
        messages.extend(results)
        del state["pending_tools"][:len(batch)]

        if not state["pending_tools"]:
            state["phase"] = "model"


def run_agent(messages: list[dict], max_steps: int = 6, resume: bool = False) -> tuple[AgentStatus, list[dict]]:
    """维护传入的消息历史；max_steps 限制模型调用次数，而非工具调用次数。"""
    if type(max_steps) is not int or max_steps < 1:
        raise ValueError("max_steps 必须是正整数")

    if resume:
        state = checkpoint.load_checkpoint()
        messages = state["messages"]
    else:
        state = {
            "run_id": uuid.uuid4().hex,
            "messages": messages,
            "phase": "model",
            "model_calls": 0,
            "pending_tools": [],
        }
        checkpoint.save_checkpoint(state)

    try:
        while state["phase"] not in {"done", "failed"}:
            # 预算只阻止下一次模型请求；已收到的工具请求仍执行完。
            if state["phase"] == "model" and state["model_calls"] >= max_steps:
                return AgentStatus.MAX_STEPS, messages
            step(state)
            checkpoint.save_checkpoint(state)

        if state["phase"] == "done":
            return AgentStatus.COMPLETE, messages
        return AgentStatus.INCOMPLETE, messages
    except Exception as error:
        # 工具的预期业务错误已转成 tool 消息；其余运行失败在任务边界终止。
        print(f"Agent 运行失败（{type(error).__name__}）：{error}")
        return AgentStatus.INCOMPLETE, messages

def main() -> None:
    messages = memory.load_messages()
    if not messages:
        messages = [
            {
                "role": "system",
                "content": (
                    "你是学习笔记助手，使用简洁的中文回答。先搜索本地笔记获取编号；"
                    "回答笔记的具体内容前，使用 read_notes 读取正文。"
                    "依据工具结果回答并标注笔记编号；没有找到时明确说明。"
                )
            }
        ]

    while True:
        question = input("请输入问题（输入 exit 退出）：").strip()
        if not question:
            continue
        if question == "exit":
            break
        if question == "/resume":
            status, messages = run_agent(messages, 10, resume=True)
        else:
            message = {
                "role": "user",
                "content": question
            }
            messages.append(message)

            status, messages = run_agent(messages, 6)

        if status == AgentStatus.COMPLETE:
            print(messages[-1]["role"], messages[-1]["content"])
            memory.save_messages(messages)
        elif status == AgentStatus.INCOMPLETE:
            print("Agent 生成未完成或运行失败")
            break
        elif status == AgentStatus.MAX_STEPS:
            print("达到最大模型调用次数，Agent 已停止，任务尚未完成")
            break

if __name__ == "__main__":
    main()
