import os
import json
import urllib.request

import tools

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

        arguments = json.loads(tool_arguments)
        if not isinstance(arguments, dict):
            raise ValueError("工具参数必须解析为一个对象")
        # 当前只有 search_notes；后续增加工具时再扩展各自的参数校验。
        if set(arguments) != {"query"}:
            raise ValueError("参数必须且只能包含 query 字段")
        query = arguments["query"]
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query 必须是非空字符串")
    except json.JSONDecodeError:
        result = {"ok": False, "error": "工具参数不是有效的 JSON"}
    except ValueError as error:
        result = {"ok": False, "error": str(error)}
    else:
        func = tools.TOOL_HANDLERS[tool_name]
        result = {
            "ok": True,
            "data": func(**arguments),
        }

    # 成功与失败都作为工具结果回传，并关联到原始调用编号。
    return {
        "role": "tool",
        "tool_call_id": tool_id,
        "content": json.dumps(result, ensure_ascii=False),
    }

def main() -> None:
    messages = [
        {
            "role": "system",
            "content": "你是学习笔记助手，使用简洁的中文回答。查询本地笔记时先调用工具，依据工具结果回答；没有找到时明确说明。"
        },
        {
            "role": "user",
            "content": "搜索 Attention 笔记"
        }
    ]

    first_response = request_llm(messages, tools.TOOL_SCHEMAS)
    first_message = first_response["choices"][0]["message"]

    # 保留完整的 tool_calls，后续工具结果才能与调用请求配对。
    messages.append(first_message)

    tool_calls = first_message.get("tool_calls") or []

    if not tool_calls:
        for message in messages:
            print(message["role"], message["content"])
        print(first_response["choices"][0]["finish_reason"], first_response["usage"])

        return

    for tool_call in tool_calls:
        print("调用工具：", tool_call["function"]["name"], tool_call["function"]["arguments"])
        messages.append(execute_tool(tool_call))

    second_response = request_llm(messages)
    second_message = second_response["choices"][0]["message"]

    messages.append(second_message)

    for message in messages:
        if message.get("tool_calls"):
            continue  # 工具请求已在执行时打印。
        print(message["role"], message["content"])

    print(first_response["choices"][0]["finish_reason"], first_response["usage"])
    print(second_response["choices"][0]["finish_reason"], second_response["usage"])

if __name__ == "__main__":
    main()
