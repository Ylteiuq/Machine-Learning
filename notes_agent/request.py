import os
import urllib.request
import json

def request_llm(messages: list[dict], tools=None, stream=False, on_text=None) -> dict:
    api_key = os.getenv("LLM_API_KEY")

    if not api_key:
        raise RuntimeError("缺少 LLM_API_KEY,请在终端设置api_key")

    api_url = os.environ["LLM_API_URL"]
    model = os.environ["LLM_MODEL"]

    payload = {
        "model": model,
        "messages": messages,
        "stream": stream,
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
        if not stream:
            return json.load(response)

        content = ""
        tool_calls = {}
        finish_reason = None
        usage = {}

        for line in response:
            line = line.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break

            chunk = json.loads(data)
            if chunk.get("usage") is not None:
                usage = chunk["usage"]
            if not chunk.get("choices"):
                continue

            choice = chunk["choices"][0]
            delta = choice.get("delta", {})
            text = delta.get("content") or ""
            if text:
                content += text
                if on_text is not None:
                    on_text(text)

            # 同一个工具调用会分多次到达，用 index 找到它，再拼接参数。
            for fragment in delta.get("tool_calls") or []:
                index = fragment["index"]
                if index not in tool_calls:
                    tool_calls[index] = {
                        "id": "",
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    }
                call = tool_calls[index]
                if fragment.get("id"):
                    call["id"] = fragment["id"]
                function = fragment.get("function") or {}
                call["function"]["name"] += function.get("name") or ""
                call["function"]["arguments"] += function.get("arguments") or ""

            if choice.get("finish_reason") is not None:
                finish_reason = choice["finish_reason"]

        if finish_reason is None:
            raise RuntimeError("流式响应未返回结束原因")

        # 对 Agent Loop 仍返回一条完整消息，工具参数由 execute_tool 解析。
        message = {"role": "assistant", "content": content}
        if tool_calls:
            message["tool_calls"] = [tool_calls[i] for i in sorted(tool_calls)]
        return {
            "choices": [{"message": message, "finish_reason": finish_reason}],
            "usage": usage,
        }
