import os
import urllib.request
import json

def request_llm(messages: list[dict], tools=None, request=False, on_text=None) -> dict:
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