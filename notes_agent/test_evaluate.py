"""验证评估器不会把口头声称成功误判为任务成功。"""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import evaluate
from test_agent import model_response, tool_call


class EvaluationTests(unittest.TestCase):
    def test_real_tools_in_isolated_cases(self):
        responses = [
            model_response(calls=[tool_call("search_notes", {"query": "Attention"}, "search")], reason="tool_calls"),
            model_response(calls=[tool_call("read_notes", {"note_id": "n1"}, "read")], reason="tool_calls"),
            model_response("笔记n1中，Q 是查询，K 是键，V 是值。"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            original_path = Path(directory) / "notes.json"
            original_path.write_text("[]", encoding="utf-8")
            with patch.object(evaluate.tools, "NOTES_FILE", original_path):
                with patch.object(evaluate.agent, "request_llm", side_effect=responses):
                    read = evaluate.run_case(evaluate.CASES[0])
                with patch.object(evaluate.agent, "request_llm", side_effect=[
                    model_response(calls=[tool_call("save_notes", evaluate.CASES[1]["expected"], "save")], reason="tool_calls"),
                    model_response("保存完成。"),
                ]):
                    saved = evaluate.run_case(evaluate.CASES[1])
                self.assertEqual(evaluate.tools.NOTES_FILE, original_path)
                self.assertEqual(original_path.read_text(encoding="utf-8"), "[]")
        self.assertTrue(read["task_passed"])
        self.assertTrue(saved["task_passed"])
        self.assertEqual(len(saved["notes_after"]), len(evaluate.INITIAL_NOTES) + 1)
        self.assertEqual([r["name"] for r in read["tool_calls"]], ["search_notes", "read_notes"])
        history = copy.deepcopy(read["messages"])
        history[-1]["content"] = "根据 n10 回答。"
        self.assertFalse(evaluate.ensure_notes(history, evaluate.CASES[0]["expected"]))

    def test_claimed_success_without_actions_fails(self):
        for case in evaluate.CASES:
            with self.subTest(case=case["id"]), patch.object(
                evaluate.agent, "request_llm", return_value=model_response("已读取 n1，并保存了笔记。")
            ):
                result = evaluate.run_case(case)
            self.assertEqual(result["runtime_status"], "COMPLETE")
            self.assertFalse(result["task_passed"])
            self.assertTrue(result["failure_reason"])

    def test_failed_attempt_can_be_followed_by_success(self):
        bad_call = tool_call("read_notes", {})
        bad_call["function"]["arguments"] = "invalid json"
        responses = [
            model_response(calls=[bad_call], reason="tool_calls"),
            model_response(calls=[tool_call("search_notes", {"query": "Attention"}, "search")], reason="tool_calls"),
            model_response(calls=[tool_call("read_notes", {"note_id": "n1"}, "read")], reason="tool_calls"),
            model_response("根据 n1，Q 是查询，K 是键，V 是值。"),
        ]
        with patch.object(evaluate.agent, "request_llm", side_effect=responses):
            result = evaluate.run_case(evaluate.CASES[0])
        self.assertTrue(result["task_passed"])
        self.assertFalse(result["tool_calls"][0]["result"]["ok"])

    def test_wrong_content_and_duplicate_saves_fail(self):
        expected = evaluate.CASES[1]["expected"]
        for added in [
            [{"id": "n3", **expected, "content": "错误正文"}],
            [{"id": "n3", **expected}, {"id": "n4", **expected}],
        ]:
            passed, reason = evaluate.check_task(evaluate.CASES[1], [], evaluate.INITIAL_NOTES + added)
            self.assertFalse(passed)
            self.assertTrue(reason)

    def test_runtime_failure_is_recorded(self):
        with patch.object(evaluate.agent, "request_llm", side_effect=TimeoutError("模拟请求超时")):
            result = evaluate.run_case(evaluate.CASES[0])
        self.assertEqual(result["runtime_status"], "INCOMPLETE")
        self.assertFalse(result["task_passed"])
        self.assertIn("模拟请求超时", result["runtime_log"])


if __name__ == "__main__":
    unittest.main()
