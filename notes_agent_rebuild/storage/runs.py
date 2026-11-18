import json

from ..agent.state import RunState, RunStatus


class RunStore:
    def __init__(self, database):
        self.database = database

    def save(self, run: RunState, conn) -> None:
        conn.execute("""
            INSERT INTO runs (id, session_id, parent_run_id, call_id, phase, status,
                model_calls, max_steps, allowed_tools, pending_tools, error, usage)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET phase=excluded.phase, status=excluded.status,
                model_calls=excluded.model_calls, max_steps=excluded.max_steps,
                pending_tools=excluded.pending_tools, error=excluded.error, usage=excluded.usage
        """, (run.id, run.session_id, run.parent_run_id, run.call_id, run.phase, run.status.value,
              run.model_calls, run.max_steps, json.dumps(run.allowed_tools),
              json.dumps(run.pending_tools, ensure_ascii=False), run.error, json.dumps(run.usage)))

    def get(self, run_id: str) -> RunState:
        with self.database.connect() as conn:
            row = conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        if row is None:
            raise ValueError(f"找不到运行：{run_id}")
        return RunState(
            id=row["id"], session_id=row["session_id"], parent_run_id=row["parent_run_id"],
            call_id=row["call_id"], phase=row["phase"], status=RunStatus(row["status"]),
            model_calls=row["model_calls"], max_steps=row["max_steps"],
            allowed_tools=json.loads(row["allowed_tools"]), pending_tools=json.loads(row["pending_tools"]),
            error=row["error"], usage=json.loads(row["usage"]),
        )

    def child(self, parent_run_id: str, call_id: str) -> RunState | None:
        with self.database.connect() as conn:
            row = conn.execute("SELECT id FROM runs WHERE parent_run_id=? AND call_id=?",
                               (parent_run_id, call_id)).fetchone()
        return self.get(row["id"]) if row else None

    def totals(self, run_id: str) -> dict:
        with self.database.connect() as conn:
            rows = conn.execute("""
                WITH RECURSIVE tree(id) AS (
                    SELECT id FROM runs WHERE id=?
                    UNION ALL
                    SELECT runs.id FROM runs JOIN tree ON runs.parent_run_id=tree.id
                )
                SELECT runs.id, model_calls, usage FROM runs JOIN tree ON runs.id=tree.id
            """, (run_id,)).fetchall()
        if not rows:
            raise ValueError(f"找不到运行：{run_id}")

        total_calls = child_calls = total_tokens = child_tokens = 0
        for row in rows:
            calls = row["model_calls"]
            tokens = json.loads(row["usage"]).get("total_tokens", 0)
            total_calls += calls
            total_tokens += tokens
            if row["id"] != run_id:
                child_calls += calls
                child_tokens += tokens
        return {
            "model_calls": total_calls,
            "main_calls": total_calls - child_calls,
            "child_calls": child_calls,
            "child_runs": len(rows) - 1,
            "total_tokens": total_tokens,
            "child_tokens": child_tokens,
        }
