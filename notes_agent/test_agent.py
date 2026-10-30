"""离线验证工具协议与 Agent Loop，不调用真实模型服务。"""

import copy
import inspect
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import main as agent
import tools


def tool_call(name, arguments, call_id="call_test"):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def model_response(content=None, calls=None, reason="stop"):
    message = {"role": "assistant", "content": content}
    if calls is not None:
        message["tool_calls"] = calls
    return {
        "choices": [{"message": message, "finish_reason": reason}],
        "usage": {"total_tokens": 25},
    }


class AgentTests(unittest.TestCase):
    def run_script(self, responses, max_steps=6):
        """按给定顺序返回模型响应，记录每次请求时的历史快照。"""
        script = iter(responses)
        snapshots = []
        messages = [{"role": "user", "content": "读取 Attention 笔记并总结"}]

        def fake_request(history, schemas):
            self.assertEqual(schemas, tools.TOOL_SCHEMAS)
            snapshots.append(copy.deepcopy(history))
            response = next(script)
            if isinstance(response, Exception):
                raise response
            return copy.deepcopy(response)

        with patch.object(agent, "request_llm", side_effect=fake_request):
            with redirect_stdout(io.StringIO()):
                status, returned_messages = agent.run_agent(messages, max_steps)
        self.assertIs(returned_messages, messages)
        return status, messages, snapshots

    def test_schema_matches_python_parameters(self):
        for schema in tools.TOOL_SCHEMAS:
            function = schema["function"]
            with self.subTest(tool=function["name"]):
                parameters = function["parameters"]
                handler = tools.TOOL_HANDLERS[function["name"]]
                self.assertEqual(
                    set(parameters["properties"]),
                    set(inspect.signature(handler).parameters),
                )
                self.assertLessEqual(set(parameters["required"]), set(parameters["properties"]))

    def test_search_and_read(self):
        self.assertEqual(
            tools.search_notes(" aTtEnTiOn "),
            [{"id": "n1", "title": "Attention"}, {"id": "n2", "title": "Transformer结构"}],
        )
        self.assertEqual(tools.search_notes("不存在的关键词"), [])
        self.assertIn("Q 表示查询", tools.read_notes("n1")["content"])
        with self.assertRaises(ValueError):
            tools.read_notes("missing")

    def test_invalid_arguments_become_tool_results(self):
        for name, field in [("search_notes", "query"), ("read_notes", "note_id")]:
            for arguments in [{}, [], None, {field: None}, {field: False}, {field: 1},
                              {field: ""}, {field: "  "}, {field: "n1", "extra": 1}]:
                with self.subTest(tool=name, arguments=arguments):
                    with patch.dict(tools.TOOL_HANDLERS, {name: unittest.mock.Mock()}) as handlers:
                        result = agent.execute_tool(tool_call(name, arguments))
                        handlers[name].assert_not_called()
                    self.assertEqual(result["role"], "tool")
                    self.assertEqual(result["tool_call_id"], "call_test")
                    payload = json.loads(result["content"])
                    self.assertFalse(payload["ok"])
                    self.assertIsInstance(payload["error"], str)

    def test_unknown_tool_bad_json_and_missing_note(self):
        invalid_json = tool_call("search_notes", {})
        invalid_json["function"]["arguments"] = "not-json"
        non_string = tool_call("search_notes", {})
        non_string["function"]["arguments"] = {"query": "Attention"}
        for call in [tool_call("unknown", {}), invalid_json, non_string,
                     tool_call("read_notes", {"note_id": "missing"})]:
            with self.subTest(call=call):
                result = agent.execute_tool(call)
                self.assertEqual(result["tool_call_id"], call["id"])
                self.assertFalse(json.loads(result["content"])["ok"])

    def test_search_read_answer(self):
        search = tool_call("search_notes", {"query": "Attention"}, "search")
        reads = [tool_call("read_notes", {"note_id": "n1"}, "read_1"),
                 tool_call("read_notes", {"note_id": "n2"}, "read_2")]
        status, history, snapshots = self.run_script([
            model_response(calls=[search], reason="tool_calls"),
            model_response(calls=reads, reason="tool_calls"),
            model_response("根据 n1，Q 表示查询，K 表示键，V 表示值。"),
        ])
        self.assertEqual(status, agent.AgentStatus.COMPLETE)
        self.assertEqual([len(snapshot) for snapshot in snapshots], [1, 3, 6])
        self.assertEqual(snapshots[1][1]["tool_calls"], [search])
        results = [message for message in history if message["role"] == "tool"]
        self.assertEqual([result["tool_call_id"] for result in results], ["search", "read_1", "read_2"])
        self.assertIn("content", json.loads(results[1]["content"])["data"])
        self.assertEqual(history[-1]["role"], "assistant")

    def test_missing_note_can_be_corrected_next_step(self):
        status, history, snapshots = self.run_script([
            model_response(calls=[tool_call("read_notes", {"note_id": "missing"})], reason="tool_calls"),
            model_response(calls=[tool_call("read_notes", {"note_id": "n1"}, "retry")], reason="tool_calls"),
            model_response("根据 n1 给出回答"),
        ])
        self.assertEqual(status, agent.AgentStatus.COMPLETE)
        self.assertFalse(json.loads(snapshots[1][-1]["content"])["ok"])
        self.assertTrue(json.loads(snapshots[2][-1]["content"])["ok"])

    def test_normal_answer_on_last_allowed_step_completes(self):
        status, history, snapshots = self.run_script([model_response("直接回答")], max_steps=1)
        self.assertEqual(status, agent.AgentStatus.COMPLETE)
        self.assertEqual(len(snapshots), 1)
        self.assertEqual(history[-1]["content"], "直接回答")

    def test_step_limit_keeps_paired_tool_history(self):
        response = model_response(calls=[tool_call("search_notes", {"query": "Attention"})], reason="tool_calls")
        for limit in [1, 2]:
            with self.subTest(limit=limit):
                status, history, snapshots = self.run_script([response] * limit, max_steps=limit)
                self.assertEqual(status, agent.AgentStatus.MAX_STEPS)
                self.assertEqual(len(snapshots), limit)
                self.assertEqual([message["role"] for message in history], ["user"] + ["assistant", "tool"] * limit)

    def test_incomplete_responses_never_execute_tools(self):
        call = tool_call("search_notes", {"query": "Attention"})
        for response in [model_response("截断的回答", reason="length"),
                         model_response(calls=[call], reason="length"),
                         model_response(reason="content_filter"),
                         model_response(calls=[], reason="tool_calls"),
                         model_response(calls=[call], reason="stop")]:
            with self.subTest(response=response), patch.object(agent, "execute_tool") as execute:
                status, history, snapshots = self.run_script([response])
                self.assertEqual(status, agent.AgentStatus.INCOMPLETE)
                self.assertEqual(len(snapshots), 1)
                execute.assert_not_called()

    def test_request_failure_stops_without_fabricated_answer(self):
        response = model_response(calls=[tool_call("search_notes", {"query": "Attention"})], reason="tool_calls")
        status, history, snapshots = self.run_script([response, TimeoutError("模拟超时")])
        self.assertEqual(status, agent.AgentStatus.INCOMPLETE)
        self.assertEqual(len(snapshots), 2)
        self.assertEqual(history[-1]["role"], "tool")

    def test_invalid_step_limit(self):
        for limit in [0, -1, True, 1.5]:
            with self.subTest(limit=limit), patch.object(agent, "request_llm") as request:
                with self.assertRaises(ValueError):
                    agent.run_agent([], max_steps=limit)
                request.assert_not_called()

    def test_http_payload_and_script_entry(self):
        responses = iter([
            model_response(calls=[tool_call("search_notes", {"query": "Attention"})], reason="tool_calls"),
            model_response(calls=[tool_call("read_notes", {"note_id": "n1"})], reason="tool_calls"),
            model_response("根据 n1，Q 表示查询。"),
            model_response("Q 就是用来查找信息的查询。"),
        ])
        payloads = []

        def fake_urlopen(request, timeout):
            self.assertEqual(timeout, 60)
            self.assertEqual(request.get_method(), "POST")
            self.assertEqual(request.get_header("Authorization"), "Bearer local-test-key")
            payloads.append(json.loads(request.data))
            return io.BytesIO(json.dumps(next(responses)).encode())

        settings = {"LLM_API_KEY": "local-test-key", "LLM_API_URL": "https://example.invalid/chat/completions",
                    "LLM_MODEL": "local-test-model"}
        output = io.StringIO()
        with patch.dict(os.environ, settings, clear=True), patch("urllib.request.urlopen", side_effect=fake_urlopen):
            with tempfile.TemporaryDirectory() as directory, patch.object(
                agent.memory, "SESSION_FILE", Path(directory) / "session.json"
            ):
                self.assertEqual(agent.memory.load_messages(), [])
                with redirect_stdout(output), patch("builtins.input", side_effect=[
                    "   ", "查找并读取 Attention 笔记", "exit", "把刚才的解释缩短成一句话", "exit"
                ]):
                    agent.main()
                    saved_messages = agent.memory.load_messages()
                    self.assertEqual(saved_messages, payloads[-1]["messages"] + [
                        {"role": "assistant", "content": "根据 n1，Q 表示查询。"}
                    ])
                    agent.main()
                self.assertEqual(agent.memory.load_messages()[-1]["content"], "Q 就是用来查找信息的查询。")
        self.assertEqual(len(payloads), 4)
        self.assertTrue(all(payload["tools"] == tools.TOOL_SCHEMAS for payload in payloads))
        self.assertEqual([len(payload["messages"]) for payload in payloads], [2, 4, 6, 8])
        self.assertEqual(payloads[-1]["messages"][-2]["content"], "根据 n1，Q 表示查询。")
        self.assertIn("根据 n1", output.getvalue())
        self.assertIn("Q 就是用来查找信息的查询。", output.getvalue())


if __name__ == "__main__":
    unittest.main()
