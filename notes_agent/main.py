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

def find_tool_schemas(tool_name: str, tools: None) -> dict:
    for tool in tools:
        if tool_name == tool["function"]["name"]:
            parameters = tool["function"]["parameters"]

            return {
                "requires": parameters.get("required",[]),
                "properties": parameters.get("properties", {}),
                "addtional_properties": parameters.get("additionalProperties", True)
            }

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

        tool_find = find_tool_schemas(tool_name, tools.TOOL_SCHEMAS)

        arguments = json.loads(tool_arguments)
        for filed in tool_find["tool_arguments"]:
            if filed not in arguments:
                raise ValueError(f"缺少必填参数 {filed}")

            value = arguments[filed]

            if value is None or (isinstance(value, str) and not value.strip()):
                raise ValueError(f"参数 {filed} 不能为空")

        if tool_find["additional_properties"] is False:
            extra_fields = set(arguments) - set(tool_find["properties"])

            if extra_fields:
                raise ValueError(f"存在不允许的参数： {', '.join(sorted(extra_fields))}")
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

def run_agent(messages: list[dict], max_steps: int = 6) -> dict:
    for i in range(max_steps):
        response = request_llm(messages, tools.TOOL_SCHEMAS)
        message = response["choice"][0]["message"]

        messages.append(message)

        print(response["chioce"][0]["finish_reason"], response["usage"])

        tool_calls = message.get("tool_calls") or []

        if not tool_calls:
            return messages

        for tool_call in tool_calls:
            print("调用工具：", tool_call["function"]["name"], tool_call["function"]["arguments"])
            messages.append(execute_tool(tool_call))

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

    messages = run_agent(messages, 6)

    for message in messages:
        if message.get("tool_calls"):
            continue  # 工具请求已在执行时打印。
        print(message["role"], message["content"])

if __name__ == "__main__":
    main()
