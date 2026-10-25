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

    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)

def execute_tool(tool_call: dict) -> dict:
    """执行一次工具请求，返回可追加到messages到工具消息"""

    tool_id = tool_call["id"]
    tool_name = tool_call["function"]["name"]
    tool_arguments = tool_call["function"]["arguments"]

    func = tools.TOOL_HANDLERS[tool_name]

    tool_arguments = json.loads(tool_arguments)

    if tool_arguments.get("query") is None:
        result = {
            "ok": False,
            "data": "未包含 \"query\" 字段"
        }

        return {
            "role": "tool",
            "tool_call_id": tool_call["id"],
            "content": json.dumps(result, ensure_ascii=False)
        }

    result = {
        "ok": True,
        "data": func(**tool_arguments)
    }

    res = {
        "role": "tool",
        "tool_call_id": tool_call["id"],
        "content": json.dumps(result, ensure_ascii=False)
    }

    return res

def main() -> None:
    messages = [
        {
            "role": "system",
            "content": "你是学习笔记助手，使用简介的中文回答，必要的时候可以调用工具。"
        },
        {
            "role": "user",
            "content": "搜索 Attention 笔记"
        }
    ]

    first_response = request_llm(messages, tools.TOOL_SCHEMAS)
    first_message = first_response["choices"][0]["message"]

    messages.append({first_message})

    tool_calls = first_message.get("tool_calls") or []

    if not tool_calls:
        for message in messages:
                print(message["content"])

        return

    for tool_call in tool_calls:
        messages.append(execute_tool(tool_call))

    second_response = request_llm(messages)
    second_message = second_response["choices"][0]["message"]

    messages.append({
        "role": second_message["role"],
        "content": second_message["content"]
    })

    for message in messages:
        print(message["content"])

    print(first_response["choices"][0]["finish_reason"], first_response["usage"])
    print(second_response["choices"][0]["finish_reason"], second_response["usage"])

if __name__ == "__main__":
    main()
