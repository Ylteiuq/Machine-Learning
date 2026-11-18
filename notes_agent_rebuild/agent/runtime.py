import uuid
from concurrent.futures import ThreadPoolExecutor

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
        if session.active_run_id and self.runs.get(session.active_run_id).status != RunStatus.COMPLETE:
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

    def run(self, run_id: str, on_text=None) -> RunState:
        run = self.runs.get(run_id)
        session = self.sessions.get(run.session_id)
        if run.status == RunStatus.ARCHIVED:
            raise ValueError("旧运行仅供查看，不能用新工具继续执行")
        if run.status == RunStatus.COMPLETE:
            return run
        run.status, run.error = RunStatus.RUNNING, ""
        try:
            while run.phase != "done":
                if run.phase == "model" and run.model_calls >= run.max_steps:
                    run.status = RunStatus.LIMIT
                    break
                self.step(session, run, on_text)
                self.save(session, run)
            if run.phase == "done":
                run.status = RunStatus.COMPLETE
        except Exception as error:
            run.status = RunStatus.FAILED
            run.error = f"{type(error).__name__}: {error}"
            self.emit(run, run.error)
        self.save(session, run)
        return run

    def resume(self, run_id: str, additional_steps: int = 6, on_text=None) -> RunState:
        run = self.runs.get(run_id)
        if run.status in {RunStatus.LIMIT, RunStatus.FAILED}:
            run.max_steps = max(run.max_steps, run.model_calls + additional_steps)
            with self.database.connect() as conn:
                self.runs.save(run, conn)
        return self.run(run_id, on_text)

    def step(self, session, run, on_text=None) -> None:
        if run.phase == "model":
            run.model_calls += 1
            self.save(session, run)
            response = self.model.request(
                self.context.build(session), self.registry.schemas(run.allowed_tools),
                stream=on_text is not None, on_text=on_text,
            )
            choice = response["choices"][0]
            message, reason = choice["message"], choice["finish_reason"]
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                run.usage[key] = run.usage.get(key, 0) + (response.get("usage") or {}).get(key, 0)
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
