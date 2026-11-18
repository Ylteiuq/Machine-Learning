import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path


class Database:
    """每次操作使用独立连接；并行 Agent 只在短事务中写入数据库。"""

    def __init__(self, path: Path):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    summary TEXT NOT NULL DEFAULT '',
                    summarized_until INTEGER NOT NULL DEFAULT 0,
                    active_run_id TEXT,
                    workspace TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS messages (
                    session_id TEXT NOT NULL REFERENCES sessions(id),
                    position INTEGER NOT NULL,
                    payload TEXT NOT NULL,
                    PRIMARY KEY (session_id, position)
                );
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES sessions(id),
                    parent_run_id TEXT,
                    call_id TEXT,
                    phase TEXT NOT NULL,
                    status TEXT NOT NULL,
                    model_calls INTEGER NOT NULL,
                    max_steps INTEGER NOT NULL,
                    allowed_tools TEXT NOT NULL,
                    pending_tools TEXT NOT NULL,
                    error TEXT NOT NULL,
                    usage TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE (parent_run_id, call_id)
                );
                CREATE TABLE IF NOT EXISTS preferences (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS imports (
                    source TEXT PRIMARY KEY,
                    details TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                PRAGMA user_version = 1;
            """)

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def preferences(self) -> dict:
        with self.connect() as conn:
            return {row["key"]: json.loads(row["value"])
                    for row in conn.execute("SELECT key, value FROM preferences")}

    def remember(self, key: str, value: str) -> dict:
        with self.connect() as conn:
            conn.execute("""
                INSERT INTO preferences (key, value) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """, (key, json.dumps(value, ensure_ascii=False)))
        return {key: value}
