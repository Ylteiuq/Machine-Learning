import json
import uuid

from ..agent.state import Session


class SessionStore:
    def __init__(self, database):
        self.database = database

    def create(self, system_prompt: str, title: str = "新会话", kind: str = "main",
               workspace: str = "") -> Session:
        session = Session(uuid.uuid4().hex, title,
                          [{"role": "system", "content": system_prompt}], kind, workspace=workspace)
        with self.database.connect() as conn:
            self.save(session, conn)
        return session

    def save(self, session: Session, conn) -> None:
        conn.execute("""
            INSERT INTO sessions (id, title, kind, summary, summarized_until, active_run_id, workspace)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET title=excluded.title, summary=excluded.summary,
                summarized_until=excluded.summarized_until, active_run_id=excluded.active_run_id
        """, (session.id, session.title, session.kind, session.summary,
              session.summarized_until, session.active_run_id, session.workspace))
        count = conn.execute("SELECT COUNT(*) FROM messages WHERE session_id=?", (session.id,)).fetchone()[0]
        # 历史只追加；摘要仅更新 sessions，不另存一份完整历史。
        conn.executemany("INSERT INTO messages (session_id, position, payload) VALUES (?, ?, ?)", [
            (session.id, position, json.dumps(message, ensure_ascii=False))
            for position, message in enumerate(session.messages[count:], start=count)
        ])

    def get(self, session_id: str) -> Session:
        with self.database.connect() as conn:
            row = conn.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
            if row is None:
                raise ValueError(f"找不到会话：{session_id}")
            messages = [json.loads(item[0]) for item in conn.execute(
                "SELECT payload FROM messages WHERE session_id=? ORDER BY position", (session_id,))]
        return Session(row["id"], row["title"], messages, row["kind"], row["summary"],
                       row["summarized_until"], row["active_run_id"], row["workspace"])

    def latest(self, workspace: str) -> Session | None:
        with self.database.connect() as conn:
            row = conn.execute("SELECT id FROM sessions WHERE kind='main' AND workspace=? ORDER BY rowid DESC LIMIT 1",
                               (workspace,)).fetchone()
        return self.get(row["id"]) if row else None

    def list(self) -> list[dict]:
        with self.database.connect() as conn:
            return [dict(row) for row in conn.execute(
                "SELECT id, title, kind, active_run_id FROM sessions ORDER BY rowid DESC")]
