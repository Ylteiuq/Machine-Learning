import json
import sqlite3
from contextlib import closing
from pathlib import Path

file_name = Path(__file__).resolve().parent / "notes.db"
JSON_FILE = Path(__file__).resolve().parent / "notes.json"


def make_sqlite() -> bool:
    """建表并一次性导入旧笔记，保留 n1、n2 等编号。"""
    with closing(sqlite3.connect(file_name)) as conn, conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                execution_id TEXT UNIQUE
            )
        """)
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts
            USING fts5(
                title,
                content,
                content='notes',
                content_rowid='id',
                tokenize='trigram'
            )
        """)
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS notes_ai
            AFTER INSERT ON notes
            BEGIN
                INSERT INTO notes_fts(rowid, title, content)
                VALUES (new.id, new.title, new.content);
            END;
        """)
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS notes_ad
            AFTER DELETE ON notes
            BEGIN
                INSERT INTO notes_fts(
                    notes_fts, rowid, title, content
                )
                VALUES (
                    'delete', old.id, old.title, old.content
                );
            END;
        """)
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS notes_au
            AFTER UPDATE ON notes
            BEGIN
                INSERT INTO notes_fts(
                    notes_fts, rowid, title, content
                )
                VALUES (
                    'delete', old.id, old.title, old.content
                );

                INSERT INTO notes_fts(rowid, title, content)
                VALUES (new.id, new.title, new.content);
            END;
        """)
        if conn.execute("PRAGMA user_version").fetchone()[0] == 0:
            if JSON_FILE.exists():
                notes = json.loads(JSON_FILE.read_text(encoding="utf-8"))
                for note in notes:
                    conn.execute("""
                        INSERT INTO notes (id, title, content, execution_id)
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT(id) DO NOTHING
                    """, (int(note["id"][1:]), note["title"], note["content"], note.get("execution_id")))
            conn.execute("PRAGMA user_version = 1")

        if conn.execute("PRAGMA user_version").fetchone()[0] < 2:
            conn.execute("""
                INSERT INTO notes_fts(notes_fts)
                VALUES('rebuild')
            """)
            conn.execute("PRAGMA user_version = 2")
    return True


def _note_dict(row: sqlite3.Row) -> dict:
    note = dict(row)
    note["id"] = f"n{note['id']}"
    if note["execution_id"] is None:
        del note["execution_id"]
    return note


def create_note(title: str, content: str, execution_id: str) -> dict:
    with closing(sqlite3.connect(file_name)) as conn, conn:
        conn.row_factory = sqlite3.Row
        # 写入和读取结果在同一事务中完成，重复执行返回第一次保存的笔记。
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("""
            INSERT INTO notes (title, content, execution_id)
            VALUES (?, ?, ?)
            ON CONFLICT(execution_id) DO NOTHING
        """, (title, content, execution_id))
        row = conn.execute(
            "SELECT * FROM notes WHERE execution_id = ?", (execution_id,)
        ).fetchone()
        return _note_dict(row)


def get_note_by_id(note_id: str) -> dict:
    if not note_id.startswith("n") or not note_id[1:].isdigit():
        raise ValueError(f"无效的笔记编号: {note_id}")
    with closing(sqlite3.connect(file_name)) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM notes WHERE id = ?", (int(note_id[1:]),)).fetchone()
        if row is None:
            raise ValueError(f"未查询到对应的id: {note_id}")
        return _note_dict(row)


def update_note(note_id: str, title: str, content: str, execetion_id: str) -> dict:
    with closing(sqlite3.connect(file_name)) as conn:
        with conn:
            conn.row_factory = sqlite3.Row
            conn.execute("BEGIN IMMEDIATE")
            cursor = conn.execute("""
                UPDATE notes
                SET title = ?, content = ?
                WHERE id = ?
            """, (title, content, int(note_id[1:])))
            if cursor.rowcount == 0:
                raise "没有笔记被更改"
            row = conn.execute(
                "SELECT * FROM notes WHERE id = ?", (note_id,)
            ).fetchone()
            return _note_dict(row)


def delete_note():
    ...


def list_notes() -> list[dict]:
    with closing(sqlite3.connect(file_name)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM notes ORDER BY id").fetchall()
        return [_note_dict(row) for row in rows]

def search_notes(query: str, top_k: int = 5) -> list[dict]:
    """长词按 BM25 排序，短词用子串查询补充；返回可直接读取的笔记编号。"""
    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k 必须是正整数")
    terms = list(dict.fromkeys(query.strip().lower().split()))
    if not terms:
        return []
    long_terms = [term for term in terms if len(term) >= 3]
    short_terms = [term for term in terms if len(term) < 3]

    with closing(sqlite3.connect(file_name)) as conn:
        conn.row_factory = sqlite3.Row
        rows = []
        if long_terms:
            # 每个词单独转义，再用 OR 组合，避免把整段查询当成一个短语。
            match_query = ' OR '.join('"' + term.replace('"', '""') + '"' for term in long_terms)
            rows = conn.execute("""
                SELECT notes.id, notes.title, bm25(notes_fts) AS score
                FROM notes_fts
                JOIN notes ON notes.id = notes_fts.rowid
                WHERE notes_fts MATCH ?
                ORDER BY score, notes.id
                LIMIT ?
            """, (match_query, top_k)).fetchall()

        if short_terms and len(rows) < top_k:
            # trigram 的 MATCH 无法命中不足三个字符的词。
            conditions = []
            parameters = []
            for term in short_terms:
                conditions.append("(instr(lower(title), ?) > 0 OR instr(lower(content), ?) > 0)")
                parameters.extend([term, term])
            short_rows = conn.execute(
                "SELECT id, title, NULL AS score FROM notes WHERE "
                + " OR ".join(conditions) + " ORDER BY id LIMIT ?",
                (*parameters, top_k),
            ).fetchall()
            seen = {row["id"] for row in rows}
            rows.extend(row for row in short_rows if row["id"] not in seen)

        return [
            {"id": f"n{row['id']}", "title": row["title"], "score": row["score"]}
            for row in rows[:top_k]
        ]
