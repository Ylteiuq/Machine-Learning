import json

import memory


def build_context(messages: list[dict], max_turns: int = 3) -> list[dict]:
    """保留开头的 system 消息和最近几轮；约定 max_turns 为正整数。"""
    user_indices = []
    for i, message in enumerate(messages):
        if message["role"] == "user":
            user_indices.append(i)

    if len(user_indices) <= max_turns:
        context = messages.copy()

    else:
        # 从 user 消息开始截取，保留这一轮后续的全部工具请求和结果。
        start = user_indices[-max_turns]

        context = [messages[0]]

        context.extend(messages[start:])

    preferences = memory.load_preferences()

    if preferences:
        context[0] = context[0].copy()
        context[0]["content"] += (
            "\n\n以下是已保存的用户偏好，用于调整回答方式；用户当前明确要求优先："
            + json.dumps(preferences, ensure_ascii=False, indent=2)
        )

    return context
