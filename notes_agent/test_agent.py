"""离线验证工具协议与 Agent Loop，不调用真实模型服务。"""

import copy
import inspect
import io
import json
import os
import tempfile
import threading
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
    def test_parallel_reads_keep_result_order_and_do_not_cross_writes(self):
        calls = [tool_call("read_notes", {"note_id": "n1"}, "read_1"),
                 tool_call("read_notes", {"note_id": "n2"}, "read_2"),
                 tool_call("save_notes", {"title": "测试", "content": "正文"}, "write_1"),
                 tool_call("search_notes", {"query": "测试"}, "read_3"),
                 tool_call("remember_preference", {"key": "长度", "value": "简短"}, "write_2")]
        state = {"run_id": "run-a", "phase": "tools", "model_calls": 1,
                 "pending_tools": calls.copy(), "messages": [
                     {"role": "assistant", "content": None, "tool_calls": copy.deepcopy(calls)}
                 ]}
        original = copy.deepcopy(state)
        barrier = threading.Barrier(2)
        second_finished = threading.Event()
        finished = []
        main_thread = threading.get_ident()

        def execute(call, run_id):
            self.assertEqual(run_id, "run-a")
            call_id = call["id"]
            if call_id in {"read_1", "read_2"}:
                self.assertNotEqual(threading.get_ident(), main_thread)
                barrier.wait(timeout=3)  # 串行执行无法通过此处。
                if call_id == "read_1":
                    self.assertTrue(second_finished.wait(timeout=3))
                self.assertEqual(state, original)  # 工作线程执行期间没有提交半批结果。
                finished.append(call_id)
                if call_id == "read_2":
                    second_finished.set()
            else:
                expected = {"write_1": ["read_2", "read_1"],
                            "read_3": ["read_2", "read_1", "write_1"],
                            "write_2": ["read_2", "read_1", "write_1", "read_3"]}
                self.assertEqual(finished, expected[call_id])
                self.assertEqual(threading.get_ident(), main_thread)
                finished.append(call_id)
            return {"role": "tool", "tool_call_id": call_id, "content": "结果"}

        with patch.object(agent, "execute_tool", side_effect=execute), redirect_stdout(io.StringIO()):
            for remaining in [3, 2, 1, 0]:
                agent.step(state)
                self.assertEqual(len(state["pending_tools"]), remaining)
        self.assertEqual(state["phase"], "model")
        self.assertEqual(state["model_calls"], 1)
        self.assertEqual([m["tool_call_id"] for m in state["messages"][1:]], [c["id"] for c in calls])
        self.assertEqual(state["messages"][0], original["messages"][0])

    def test_failed_read_batch_does_not_commit_partial_state(self):
        calls = [tool_call("read_notes", {"note_id": "n1"}, "read_1"),
                 tool_call("read_notes", {"note_id": "n2"}, "read_2")]
        state = {"run_id": "run-a", "phase": "tools", "model_calls": 1,
                 "pending_tools": calls.copy(), "messages": [
                     {"role": "assistant", "content": None, "tool_calls": copy.deepcopy(calls)}
                 ]}
        original = copy.deepcopy(state)
        barrier = threading.Barrier(2)
        finished = threading.Event()

        def execute(call, run_id):
            barrier.wait(timeout=3)
            if call["id"] == "read_1":
                raise OSError("模拟读取失败")
            finished.set()
            return {"role": "tool", "tool_call_id": call["id"], "content": "结果"}

        with patch.object(agent, "execute_tool", side_effect=execute), redirect_stdout(io.StringIO()):
            with self.assertRaises(OSError):
                agent.step(state)
        self.assertTrue(finished.is_set())
        self.assertEqual(state, original)

    def test_save_notes_deduplicates_by_execution_not_content(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            tools, "NOTES_FILE", Path(directory) / "notes.json"
        ):
            tools.NOTES_FILE.write_text("[]", encoding="utf-8")
            call = tool_call("save_notes", {"title": "测试", "content": "正文"}, "save_1")
            first = agent.execute_tool(call, "run-a")
            self.assertTrue(json.loads(first["content"])["ok"])
            self.assertEqual(agent.execute_tool(call, "run-a"), first)
            self.assertEqual(len(tools.load_notes()), 1)
            agent.execute_tool(call, "run-b")
            agent.execute_tool(tool_call("save_notes", {"title": "测试", "content": "正文"}, "save_2"), "run-a")
            self.assertEqual([note["id"] for note in tools.load_notes()], ["n1", "n2", "n3"])
            forged = tool_call("save_notes", {"title": "测试", "content": "正文", "execution_id": "forged"})
            self.assertFalse(json.loads(agent.execute_tool(forged, "run-a")["content"])["ok"])
            self.assertEqual(len(tools.load_notes()), 3)

    def test_resume_after_note_written_but_checkpoint_not_saved(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            tools, "NOTES_FILE", Path(directory) / "notes.json"
        ), redirect_stdout(io.StringIO()):
            tools.NOTES_FILE.write_text("[]", encoding="utf-8")
            call = tool_call("save_notes", {"title": "测试", "content": "正文"}, "save_1")
            state = {"run_id": "run-a", "phase": "tools", "model_calls": 1,
                     "pending_tools": [call], "messages": [
                         {"role": "user", "content": "保存笔记"},
                         {"role": "assistant", "content": None, "tool_calls": [call]},
                     ]}
            agent.checkpoint.save_checkpoint(state)
            with patch.object(agent.checkpoint, "save_checkpoint", side_effect=OSError("模拟保存中断")), \
                    patch.object(agent, "request_llm") as request:
                status, _ = agent.run_agent([], resume=True)
                self.assertEqual(status, agent.AgentStatus.INCOMPLETE)
                request.assert_not_called()
            self.assertEqual(len(tools.load_notes()), 1)
            saved_note = tools.load_notes()[0]
            self.assertEqual(agent.checkpoint.load_checkpoint(), state)

            with patch.object(agent, "request_llm", return_value=model_response("保存完成")):
                status, history = agent.run_agent([], resume=True)
            self.assertEqual(status, agent.AgentStatus.COMPLETE)
            self.assertEqual(tools.load_notes(), [saved_note])
            self.assertEqual(json.loads(history[-2]["content"])["data"], saved_note)
            saved_state = agent.checkpoint.load_checkpoint()
            self.assertEqual(saved_state["run_id"], "run-a")
            self.assertEqual(saved_state["phase"], "done")
            self.assertEqual(saved_state["model_calls"], 2)

    def test_failed_replacement_preserves_notes_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            tools, "NOTES_FILE", Path(directory) / "notes.json"
        ):
            tools.NOTES_FILE.write_text("[]", encoding="utf-8")
            original = {"run_id": "run-a", "phase": "model"}
            agent.checkpoint.save_checkpoint(original)
            with patch.object(Path, "replace", side_effect=OSError("模拟替换失败")):
                with self.assertRaises(OSError):
                    tools.save_notes("测试", "正文", execution_id="run-a:save_1")
                with self.assertRaises(OSError):
                    agent.checkpoint.save_checkpoint({"phase": "done"})
            self.assertEqual(tools.load_notes(), [])
            self.assertEqual(agent.checkpoint.load_checkpoint(), original)

    def test_step_runs_read_batch_without_repeating_or_changing_calls(self):
        calls = [tool_call("read_notes", {"note_id": "n1"}, "read_1"),
                 tool_call("read_notes", {"note_id": "n2"}, "read_2")]
        expected_calls = copy.deepcopy(calls)
        state = {"run_id": "test-run", "messages": [{"role": "user", "content": "读取两篇笔记"}],
                 "phase": "model", "model_calls": 0, "pending_tools": []}
        responses = [model_response(calls=calls, reason="tool_calls"), model_response("读取完成")]
        with patch.object(agent, "request_llm", side_effect=responses) as request, \
                patch.object(agent, "execute_tool", wraps=agent.execute_tool) as execute, \
                redirect_stdout(io.StringIO()):
            agent.step(state)
            self.assertEqual(state["phase"], "tools")
            self.assertEqual(state["model_calls"], 1)
            execute.assert_not_called()

            agent.step(state)
            self.assertEqual(state["phase"], "model")
            self.assertEqual(state["pending_tools"], [])
            self.assertEqual(state["model_calls"], 1)
            self.assertEqual(request.call_count, 1)
            self.assertCountEqual([call.args[0]["id"] for call in execute.call_args_list], ["read_1", "read_2"])
            self.assertEqual([message["tool_call_id"] for message in state["messages"][-2:]], ["read_1", "read_2"])
            self.assertEqual(state["messages"][1]["tool_calls"], expected_calls)

            agent.step(state)
            self.assertEqual(state["phase"], "done")
            self.assertEqual(state["model_calls"], 2)
            finished = copy.deepcopy(state)
            agent.step(state)
            self.assertEqual(state, finished)
            self.assertEqual(request.call_count, 2)
            self.assertEqual(execute.call_count, 2)

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        preferences_file = patch.object(
            agent.memory, "PREFERENCES_FILE", Path(directory.name) / "preferences.json"
        )
        preferences_file.start()
        self.addCleanup(preferences_file.stop)
        checkpoint_file = patch.object(agent.checkpoint, "CHECKPOINT", Path(directory.name) / "checkpoint.json")
        checkpoint_file.start()
        self.addCleanup(checkpoint_file.stop)

    def test_context_includes_current_preferences_without_changing_history(self):
        for turns in [1, 4]:
            with self.subTest(turns=turns):
                messages = [{"role": "system", "content": "学习笔记助手"}]
                for turn in range(turns):
                    messages.extend([
                        {"role": "user", "content": f"问题 {turn}"},
                        {"role": "assistant", "content": f"回答 {turn}"},
                    ])
                original = copy.deepcopy(messages)
                agent.memory.remember_preference("回答长度", "尽量简短")
                first = agent.context.build_context(messages)
                second = agent.context.build_context(messages)
                self.assertEqual(first, second)
                self.assertEqual(first[0]["content"].count("尽量简短"), 1)
                self.assertEqual(len(first), 1 + 2 * min(turns, 3))

                agent.memory.remember_preference("回答长度", "详细解释")
                updated = agent.context.build_context(messages)
                self.assertIn("详细解释", updated[0]["content"])
                self.assertNotIn("尽量简短", updated[0]["content"])
                self.assertEqual(messages, original)

    def test_preference_tool_saves_and_updates_without_losing_other_keys(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            agent.memory, "PREFERENCES_FILE", Path(directory) / "preferences.json"
        ):
            self.assertEqual(agent.memory.load_preferences(), {})
            for key, value in [("回答长度", "尽量简短"), ("解释方式", "先举例"), ("回答长度", "详细解释")]:
                result = agent.execute_tool(tool_call("remember_preference", {"key": key, "value": value}), "test-run")
                self.assertEqual(json.loads(result["content"]), {"ok": True, "data": {key: value}})
                self.assertEqual(agent.memory.load_preferences()[key], value)
            self.assertEqual(agent.memory.load_preferences(), {"回答长度": "详细解释", "解释方式": "先举例"})

    def test_context_keeps_whole_turns_without_changing_history(self):
        messages = [{"role": "system", "content": "学习笔记助手"}]
        for turn in range(1, 4):
            messages.extend([
                {"role": "user", "content": f"第 {turn} 个问题"},
                {"role": "assistant", "content": f"第 {turn} 个回答"},
            ])

        context = agent.context.build_context(messages)
        self.assertEqual(context, messages)
        self.assertIsNot(context, messages)

        call = tool_call("read_notes", {"note_id": "n1"})
        messages.extend([
            {"role": "user", "content": "读取 Attention 笔记"},
            {"role": "assistant", "content": None, "tool_calls": [call]},
            agent.execute_tool(call, "test-run"),
        ])
        original = copy.deepcopy(messages)
        context = agent.context.build_context(messages)
        self.assertEqual(context, [messages[0]] + messages[3:])
        self.assertEqual(messages, original)

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
                    set(inspect.signature(handler).parameters) - ({"execution_id"} if function["name"] == "save_notes" else set()),
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
                        result = agent.execute_tool(tool_call(name, arguments), "test-run")
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
                result = agent.execute_tool(call, "test-run")
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
