"""显式的一次性迁移；新版启动过程不会读取旧项目或 JSON 状态文件。"""

import argparse
import json
import shutil
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime
from pathlib import Path

from .agent.state import RunState, RunStatus, Session
from .config import BASE_DIR
from .storage.database import Database
from .storage.runs import RunStore
from .storage.sessions import SessionStore


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def migrate(source: Path, database: Database, workspace: Path) -> dict:
    source, workspace = source.resolve(), workspace.resolve()
    if not source.is_dir():
        raise ValueError(f"旧项目目录不存在：{source}")
    with database.connect() as conn:
        previous = conn.execute("SELECT details FROM imports WHERE source=?", (str(source),)).fetchone()
    if previous:
        return {**json.loads(previous[0]), "already_imported": True}

    backup = database.path.parent / "legacy_backup" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    backup.mkdir(parents=True)
    # SQLite backup 会包含 WAL 中已经提交的数据。
    with closing(sqlite3.connect(database.path)) as src, closing(sqlite3.connect(backup / "agent_before.db")) as dst:
        src.backup(dst)
    json_files = [source / name for name in ("notes.json", "session.json", "preferences.json", "checkpoint.json")]
    json_files += sorted((source / "runs").glob("*.json"))
    for path in json_files:
        if path.exists():
            target = backup / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)

    notes_db = source / "notes.db"
    notes = []
    if notes_db.exists():
        with closing(sqlite3.connect(notes_db.as_uri() + "?mode=ro", uri=True)) as src:
            src.row_factory = sqlite3.Row
            with closing(sqlite3.connect(backup / "notes.db")) as dst:
                src.backup(dst)
            notes = [dict(row) for row in src.execute("SELECT id, title, content FROM notes ORDER BY id")]
    elif (source / "notes.json").exists():
        # 只有旧数据库不存在时才导入 JSON，避免恢复已从数据库删除的旧笔记。
        notes = read_json(source / "notes.json")

    exported = []
    for note in notes:
        note_id = str(note["id"])
        number = int(note_id[1:] if note_id.startswith("n") else note_id)
        path = workspace / "legacy_notes" / f"n{number}.md"
        content = f"# {note['title']}\n\n{note['content']}\n"
        if path.exists() and path.read_text(encoding="utf-8") != content:
            raise ValueError(f"导出位置已有不同内容，未覆盖：{path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        exported.append(str(path.relative_to(workspace)))

    sessions, runs = SessionStore(database), RunStore(database)
    counts = {"notes": len(exported), "sessions": 0, "runs": 0, "preferences": 0,
              "exported": exported, "backup": str(backup)}
    with database.connect() as conn:
        preferences = source / "preferences.json"
        if preferences.exists():
            for key, value in read_json(preferences).items():
                cursor = conn.execute("INSERT OR IGNORE INTO preferences (key, value) VALUES (?, ?)",
                                      (key, json.dumps(value, ensure_ascii=False)))
                counts["preferences"] += cursor.rowcount
        session_file = source / "session.json"
        if session_file.exists():
            data = read_json(session_file)
            data = {"messages": data} if isinstance(data, list) else data
            session = Session(uuid.uuid4().hex, "旧项目对话", data["messages"], kind="legacy",
                              summary=data.get("summary", ""), summarized_until=data.get("summarized_until", 0))
            sessions.save(session, conn)
            counts["sessions"] += 1
        # 会话文件与 checkpoint 分别归档，保留分叉历史，不猜测哪份覆盖哪份。
        paths = [source / "checkpoint.json", *sorted((source / "runs").glob("*.json"))]
        for path in paths:
            if not path.exists():
                continue
            data = read_json(path)
            run_id = data.get("run_id", uuid.uuid4().hex)
            session = Session(uuid.uuid4().hex, f"旧运行：{run_id[:8]}", data["messages"], kind="legacy",
                              summary=data.get("summary", ""), summarized_until=data.get("summarized_until", 0),
                              active_run_id=run_id)
            run = RunState(
                run_id, session.id, parent_run_id=data.get("parent_run_id"),
                phase=data.get("phase", "done"), status=RunStatus.ARCHIVED,
                model_calls=data.get("model_calls", 0), max_steps=data.get("max_steps", 6),
                allowed_tools=data.get("allowed_tools"), pending_tools=data.get("pending_tools", []),
                error="旧笔记工具运行已归档，不在新版文件工具上重放。",
            )
            sessions.save(session, conn)
            runs.save(run, conn)
            counts["sessions"] += 1
            counts["runs"] += 1
        conn.execute("INSERT INTO imports (source, details) VALUES (?, ?)",
                     (str(source), json.dumps(counts, ensure_ascii=False)))
    return counts


def main():
    parser = argparse.ArgumentParser(description="迁移旧笔记 Agent 数据")
    parser.add_argument("--source", type=Path, default=BASE_DIR.parent / "notes_agent")
    parser.add_argument("--db", type=Path, default=BASE_DIR / "data" / "agent.db")
    parser.add_argument("--workspace", type=Path, default=BASE_DIR / "workspace")
    args = parser.parse_args()
    print(json.dumps(migrate(args.source, Database(args.db), args.workspace), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
