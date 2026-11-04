"""使用真实模型评估两个任务；所有工具数据写入临时目录。"""

import io
import json
import os
import re
import tempfile
from contextlib import ExitStack, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import main as agent
import tools

INITIAL_NOTES = [
    {"id": "n1", "title": "Attention", "content": "Q 表示查询，K 表示键，V 表示值。"},
    {"id": "n2", "title": "Python异常处理", "content": "使用 try 和 except 处理异常。"},
]
CASES = [
    {
        "id": "search_and_read",
        "prompt": "查找并读取 Attention 笔记，解释 Q、K、V 的含义，并标注笔记编号。",
        "expected": INITIAL_NOTES[0],
    },
    {
        "id": "save_note",
        "prompt": "请新增一篇笔记，标题为“学习计划”，正文原样保存为“每天学习 Python 三十分钟。”。",
        "expected": {"title": "学习计划", "content": "每天学习 Python 三十分钟。"},
    },
]
REPORT_FILE = Path(__file__).resolve().parent / "evaluation_report.json"


def tool_records(messages: list[dict]) -> list[dict]:
    """通过调用 ID 将工具请求与结果配对，不依赖工具返回顺序。"""
    calls = {}
    records = []
    for index, message in enumerate(messages):
        for call in message.get("tool_calls") or []:
            calls[call["id"]] = call
        if message["role"] == "tool":
            call = calls[message["tool_call_id"]]
            try:
                arguments = json.loads(call["function"]["arguments"])
            except (json.JSONDecodeError, TypeError):
                arguments = call["function"]["arguments"]
            records.append({
                "tool_call_id": call["id"],
                "name": call["function"]["name"],
                "arguments": arguments,
                "result": json.loads(message["content"]),
                "message_index": index,
            })
    return records


def final_answer(messages: list[dict]) -> str:
    if messages and messages[-1]["role"] == "assistant" and not messages[-1].get("tool_calls"):
        return messages[-1].get("content") or ""
    return ""


def ensure_notes(messages: list[dict], argument: dict) -> bool:
    """检查目标笔记先被搜索到、再被完整读取，并在最终回答中引用。"""
    found = False
    read = False
    for record in tool_records(messages):
        result = record["result"]
        if not result["ok"]:
            continue  # 一次失败不代表整个任务失败，允许模型修正后重试。
        if record["name"] == "search_notes":
            found = found or any(note["id"] == argument["id"] for note in result["data"])
        elif record["name"] == "read_notes" and found:
            if record["arguments"].get("note_id") == argument["id"] and result["data"] == argument:
                read = True
    # 避免把 n10 中的 n1 当成有效引用。
    citation = re.search(r"(?<![A-Za-z0-9_])" + re.escape(argument["id"]) + r"(?![A-Za-z0-9_])", final_answer(messages))
    return read and citation is not None


def check_task(case: dict, messages: list[dict], notes_after: list[dict]) -> tuple[bool, str]:
    if case["id"] == "search_and_read":
        if notes_after != INITIAL_NOTES:
            return False, "读取任务修改了笔记文件"
        passed = ensure_notes(messages, case["expected"])
        return passed, "" if passed else "未完成目标笔记的搜索、读取或最终编号引用"

    original_ids = {note["id"] for note in INITIAL_NOTES}
    originals = [note for note in notes_after if note["id"] in original_ids]
    added = [note for note in notes_after if note["id"] not in original_ids]
    if originals != INITIAL_NOTES:
        return False, "原有笔记被修改或删除"
    if len(added) != 1:
        return False, f"应新增一篇笔记，实际新增 {len(added)} 篇"
    if any(added[0].get(key) != value for key, value in case["expected"].items()):
        return False, "新增笔记的标题或正文与要求不符"
    return True, ""


def run_case(case: dict) -> dict:
    messages = [{"role": "system", "content": agent.SYSTEM_PROMPT},
                {"role": "user", "content": case["prompt"]}]
    output = io.StringIO()
    with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
        root = Path(directory)
        for module, attribute, filename in [
            (tools, "NOTES_FILE", "notes.json"),
            (agent.memory, "PREFERENCES_FILE", "preferences.json"),
            (agent.memory, "SESSION_FILE", "session.json"),
            (agent.checkpoint, "CHECKPOINT", "checkpoint.json"),
        ]:
            stack.enter_context(patch.object(module, attribute, root / filename))
        tools.NOTES_FILE.write_text(json.dumps(INITIAL_NOTES, ensure_ascii=False), encoding="utf-8")
        with redirect_stdout(output):
            status, messages = agent.run_agent(messages, max_steps=6)
        notes_after = tools.load_notes()
        passed, reason = check_task(case, messages, notes_after)
        if status != agent.AgentStatus.COMPLETE:
            passed = False
            reason = f"运行未正常完成：{status.name}" + (f"；{reason}" if reason else "")

    return {
        "case_id": case["id"], "task": case["prompt"],
        "runtime_status": status.name, "task_passed": passed, "failure_reason": reason,
        "final_answer": final_answer(messages), "tool_calls": tool_records(messages),
        "notes_before": INITIAL_NOTES, "notes_after": notes_after,
        "messages": messages, "runtime_log": output.getvalue(),
    }


def main() -> None:
    results = []
    for case in CASES:
        print(f"正在评估：{case['id']}", flush=True)
        result = run_case(case)
        results.append(result)
        print(f"  运行：{result['runtime_status']}；任务：{'通过' if result['task_passed'] else '未通过'}", flush=True)
        if result["failure_reason"]:
            print(f"  原因：{result['failure_reason']}", flush=True)
    passed = sum(result["task_passed"] for result in results)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": os.environ.get("LLM_MODEL"),
        "total": len(results), "passed": passed, "success_rate": passed / len(results),
        "scope": "规则检查工具使用、引用编号和保存结果；回答解释的语义质量需要人工检查。",
        "results": results,
    }
    REPORT_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"任务通过 {passed}/{len(results)}；报告：{REPORT_FILE}")


if __name__ == "__main__":
    main()
