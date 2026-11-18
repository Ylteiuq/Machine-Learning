"""保留原来的 urllib 请求和流式拼接，连接配置由实例保存。"""

import urllib.request
import json
from math import ceil


class LLMClient:
    def __init__(self, api_url: str, api_key: str, model: str, context_window: int):
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        self.context_window = context_window
        self.max_output_tokens = 2048
        if self.context_window <= self.max_output_tokens:
            raise ValueError("CONTEXT_WINDOW 必须大于预留的 2048 个输出 token")

    @property
    def input_budget(self) -> int:
        return self.context_window - self.max_output_tokens

    @staticmethod
    def estimate_input_tokens(messages, tools=None) -> int:
        """按聊天内容和工具描述近似计数，用于请求前判断预算。"""
        prompt = {"messages": messages}
        if tools:
            prompt.update(tools=tools, tool_choice="auto")
        text = json.dumps(prompt, ensure_ascii=False, separators=(",", ":"))
        ascii_chars = sum(char.isascii() for char in text)
        # 未绑定特定 tokenizer：ASCII 每 3 个字符约 1 token，其余每个字符按 1.5 估算。
        return ceil(ascii_chars / 3 + (len(text) - ascii_chars) * 1.5)

    def request(self, messages: list[dict], tools=None, stream=False, on_text=None) -> dict:
        api_key = self.api_key

        if not api_key:
            raise RuntimeError("缺少 LLM_API_KEY,请在终端设置api_key")

        api_url = self.api_url
        model = self.model

        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "thinking": {"type": "disabled"},
            "max_tokens": self.max_output_tokens,
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
