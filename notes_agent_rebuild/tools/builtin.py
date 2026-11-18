from .registry import Tool


def memory_tool(database) -> Tool:
    return Tool(
        "remember_preference", "用户明确要求记住某项偏好时保存，同名偏好覆盖旧值。",
        {"type": "object", "properties": {
            "key": {"type": "string", "minLength": 1},
            "value": {"type": "string"},
        }, "required": ["key", "value"], "additionalProperties": False},
        lambda arguments, context: database.remember(**arguments),
    )
