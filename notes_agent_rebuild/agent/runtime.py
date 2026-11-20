import uuid
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from .context import ContextManager
from .state import RunState, RunStatus, ToolContext
from ..storage.runs import RunStore
from ..storage.sessions import SessionStore


class AgentRuntime:
    """实例持有公共依赖；每次 run 的状态是局部变量，可同时运行不同子任务。"""

    def __init__(self, model, registry, database, on_event=None):
        self.model = model
        self.registry = registry
        self.database = database
        self.sessions = SessionStore(database)
        self.runs = RunStore(database)
        self.context = ContextManager(model, database, self.sessions)
        self.on_event = on_event

    def emit(self, run, text):
        if self.on_event is not None:
            self.on_event(run.id, text)

    def create_run(self, session_id: str, task: str, *, max_steps: int = 6,
                   parent_run_id=None, call_id=None, allowed_tools=None) -> RunState:
        if max_steps < 1:
            raise ValueError("max_steps 必须大于 0")
        session = self.sessions.get(session_id)
        if session.kind == "legacy":
            raise ValueError("旧会话是只读归档，请新建通用 Agent 会话")
        if session.active_run_id and not self.runs.get(session.active_run_id).status.finished:
            raise ValueError("当前任务尚未结束，请先 /resume 或 /new")
        run = RunState(uuid.uuid4().hex, session.id, parent_run_id, call_id,
                       max_steps=max_steps, allowed_tools=sorted(allowed_tools) if allowed_tools is not None else None)
        if len(session.messages) == 1:
            session.title = task[:48]
        session.messages.append({"role": "user", "content": task})
        session.active_run_id = run.id
        self.save(session, run)
        return run

    def save(self, session, run) -> None:
        # 消息与运行进度在同一个事务中提交，恢复时不再比较两份历史。
        with self.database.connect() as conn:
            self.sessions.save(session, conn)
            self.runs.save(run, conn)

    def run(self, run_id: str, on_text=None, *, cancel_requested=None) -> RunState:
        run = self.runs.get(run_id)
        session = self.sessions.get(run.session_id)
        if run.status == RunStatus.ARCHIVED:
            raise ValueError("旧运行仅供查看，不能用新工具继续执行")
        if run.status.finished:
            return run
        run.status, run.error = RunStatus.RUNNING, ""
        try:
            while True:
                if run.phase == "done":
                    run.status = RunStatus.COMPLETE
                    break
                if cancel_requested and cancel_requested():
                    # 补齐模型发出的工具调用，与取消状态一起提交。
                    for call in run.pending_tools:
                        session.messages.append({
                            "role": "tool", "tool_call_id": call["id"],
                            "content": '{"ok": false, "error": "用户已取消，此工具未执行"}',
                        })
                    run.pending_tools.clear()
                    run.phase, run.status = "done", RunStatus.CANCELLED
                    break
                if run.phase == "model" and run.model_calls >= run.max_steps:
                    run.status = RunStatus.LIMIT
                    break
                self.step(session, run, on_text, cancel_requested=cancel_requested)
                self.save(session, run)
        except Exception as error:
            run.status = RunStatus.FAILED
            run.error = f"{type(error).__name__}: {error}"
            self.emit(run, run.error)
        self.save(session, run)
        return run

    def resume(self, run_id: str, additional_steps: int = 6, on_text=None, *, cancel_requested=None) -> RunState:
        run = self.runs.get(run_id)
        if run.status not in {RunStatus.CANCELLED, RunStatus.LIMIT, RunStatus.FAILED}:
            return self.run(run_id, on_text, cancel_requested=cancel_requested)
        session = self.sessions.get(run.session_id)
        if session.active_run_id != run.id:
            raise ValueError("会话已有新任务，只能恢复当前任务")
        if run.status == RunStatus.CANCELLED:
            last = session.messages[-1]
            # 旧记录可能在完整答复落库后才被标记为取消。
            if last["role"] == "assistant" and not last.get("tool_calls"):
                run.status = RunStatus.COMPLETE
                self.save(session, run)
                return run
            if last["role"] != "user":
                session.messages.append({"role": "user", "content": "继续刚才的任务，参考已有工具结果完成剩余工作。"})
            run.status, run.phase = RunStatus.RUNNING, "model"
        run.max_steps = max(run.max_steps, run.model_calls + additional_steps)
        self.save(session, run)
        return self.run(run_id, on_text, cancel_requested=cancel_requested)

    def request_for_run(self, session, run, messages, tools=None, on_text=None):
        run.model_calls += 1
        self.save(session, run)
        response = self.model.request(messages, tools, stream=on_text is not None, on_text=on_text)
        usage = response.get("usage") or {}
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            run.usage[key] = run.usage.get(key, 0) + (usage.get(key) or 0)
        return response

    def step(self, session, run, on_text=None, *, cancel_requested=None) -> None:
        if run.phase == "model":
            context = self.context.build(session)
            tools_schemas = self.registry.schemas(run.allowed_tools)
            send = partial(self.request_for_run, session, run)
            budget = self.model.input_budget
            estimate = self.model.estimate_input_tokens(context, tools_schemas)
            # 能直接回答时，把最后一次模型调用留给任务本身。
            spare_call = run.max_steps - run.model_calls > 1
            if estimate >= budget * 0.9 and (spare_call or estimate > budget):
                try:
                    compacted = self.context.compact(
                        session, request=send, target_savings=estimate - int(budget * 0.7)
                    )
                except (OSError, RuntimeError, ValueError) as error:
                    if estimate > budget:
                        raise
                    self.emit(run, f"自动摘要未完成，沿用原上下文：{error}")
                else:
                    if compacted:
                        context = self.context.build(session)
                        remaining = self.model.estimate_input_tokens(context, tools_schemas)
                        self.emit(run, f"已压缩历史：预计输入 {estimate} → {remaining} token，预算 {budget}")
                        estimate = remaining
            # 自动摘要也是一次模型调用，结束后先处理取消，再考虑发下一次请求。
            if cancel_requested and cancel_requested():
                return
            if estimate > budget:
                raise ValueError(f"保留的上下文和工具描述约需 {estimate} 个输入 token，预算为 {budget}。请缩短输入或新建会话")
            if run.model_calls >= run.max_steps:
                return  # 摘要用完最后一次请求，交给外层循环暂停，/resume 可继续。
            response = send(context, tools_schemas, on_text=on_text)
            choice = response["choices"][0]
            message, reason = choice["message"], choice["finish_reason"]
            self.emit(run, f"模型调用 {run.model_calls}，结束原因 {reason}")
            calls = message.get("tool_calls") or []
            if reason not in {"stop", "tool_calls"} or bool(calls) != (reason == "tool_calls"):
                raise RuntimeError(f"模型响应不完整或结束原因与工具调用不一致：{reason}")
            session.messages.append(message)
            run.pending_tools = calls.copy()
            run.phase = "tools" if calls else "done"
        elif run.phase == "tools":
            batch = self.registry.batch(run.pending_tools)
            results = self.execute_batch(run, batch)
            session.messages.extend(results)
            del run.pending_tools[:len(batch)]
            if not run.pending_tools:
                run.phase = "model"
        else:
            raise ValueError(f"未知运行阶段：{run.phase}")

    def execute_batch(self, run, calls) -> list[dict]:
        def execute(call):
            name = call["function"]["name"]
            tool = self.registry.tools.get(name)
            self.emit(run, f"调用 {name}（{tool.source if tool else 'unknown'}）")
            context = ToolContext(run.id, run.session_id, call["id"])
            return self.registry.execute(call, context, run.allowed_tools)

        if len(calls) == 1:
            return [execute(calls[0])]
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(execute, call) for call in calls]
            return [future.result() for future in futures]
