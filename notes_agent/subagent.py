"""前台委派：子 Agent 使用独立历史和 checkpoint，执行完再交回报告。"""

import json
import uuid

import checkpoint


TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "delegate_task",
        "description": "将可独立完成的查找、阅读或分析任务交给子 Agent，等待其返回报告。简单问题直接处理。",
        "parameters": {
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "明确的任务目标和需要返回的结果"},
                "context": {"type": "string", "description": "必要背景、约束和已知信息；无背景时可省略"},
            },
            "required": ["task"],
            "additionalProperties": False,
        },
    },
}

SYSTEM_PROMPT = (
    "你是负责完成委派任务的子 Agent。使用搜索和读取工具获取依据，"
    "回答笔记内容前必须读取正文。只处理收到的任务，不向用户提问。"
    "最后向主 Agent 返回简洁报告，包含结论、笔记编号以及未完成事项；不要编造结果。"
)


def delegate_task(task: str, context: str = "", *, parent_run_id: str,
                  tool_call_id: str, runner, mcp_client=None) -> dict:
    # 同一工具调用恢复时复用子任务记录，避免父运行恢复后重新开始委派。
    task_id = uuid.uuid5(uuid.NAMESPACE_URL, f"{parent_run_id}:{tool_call_id}").hex
    path = checkpoint.CHECKPOINT.parent / "runs" / f"{task_id}.json"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps({"task": task, "context": context}, ensure_ascii=False)},
    ]
    print(f"开始子任务：{task}")
    status, messages = runner(
        messages, max_steps=4, resume=path.exists(), mcp_client=mcp_client,
        checkpoint_path=path, parent_run_id=parent_run_id,
        allowed_tools={"search_notes", "read_notes"},
    )
    state = checkpoint.load_checkpoint(path)
    print(f"子任务结束：{status.name}，模型调用 {state['model_calls']} 次")
    return {
        "run_id": state["run_id"],
        "status": status.name,
        "report": messages[-1].get("content", "") if status.name == "COMPLETE" else "子任务未完成，请根据状态调整任务。",
        "model_calls": state["model_calls"],
    }
