import argparse
from pathlib import Path
from threading import Lock

from .agent.state import RunStatus
from .app import create_runtime, system_prompt
from .config import Settings


class Console:
    def __init__(self):
        self.lock = Lock()
        self.line_open = False
        self.has_text = False

    def text(self, fragment):
        with self.lock:
            print(fragment, end="", flush=True)
            self.line_open = self.has_text = True

    def event(self, run_id, message):
        with self.lock:
            if self.line_open:
                print()
                self.line_open = False
            print(f"[{run_id[:8]}] {message}")

    def result(self, runtime, run):
        if run.status == RunStatus.COMPLETE and not self.has_text:
            print(runtime.sessions.get(run.session_id).messages[-1].get("content", ""))
        totals = runtime.runs.totals(run.id)
        self.event(run.id, f"状态 {run.status.value}；总模型请求 {totals['model_calls']} 次；子任务 {totals['child_runs']} 个")
        if run.error:
            print(run.error)


HELP = "/new 新会话 | /sessions 会话列表 | /use 编号 切换会话 | /history [编号] 查看历史\n/resume 继续未完成任务 | /compact 压缩历史 | /tools 工具列表 | /usage 使用统计 | exit 退出"


def chat(runtime, settings, args, console):
    session = runtime.sessions.get(args.session) if args.session else runtime.sessions.latest(str(settings.workspace))
    if session is None:
        session = runtime.sessions.create(system_prompt(settings.workspace), workspace=str(settings.workspace))

    def select(session_id):
        selected = runtime.sessions.get(session_id)
        if selected.kind == "child":
            raise ValueError("子会话通过 /history 编号查看；请在主会话继续任务")
        if selected.kind != "legacy" and selected.workspace != str(settings.workspace):
            raise ValueError("该会话属于其他工作目录，请用对应的 --workspace 启动")
        return selected

    session = select(session.id)
    if not args.task:
        print(f"工作目录：{settings.workspace}\n会话：{session.id}\n{HELP}")
    while True:
        try:
            question = args.task or input("你：").strip()
            if question in {"exit", "/exit"}:
                return 0
            if not question:
                continue
            if question == "/new":
                session = runtime.sessions.create(system_prompt(settings.workspace), workspace=str(settings.workspace))
                print(f"新会话：{session.id}")
            elif question == "/sessions":
                for row in runtime.sessions.list():
                    print(row["id"], row["kind"], row["title"])
            elif question.startswith("/use "):
                session = select(question.split(maxsplit=1)[1])
                print(f"当前会话：{session.id} {session.title}")
            elif question == "/history" or question.startswith("/history "):
                parts = question.split(maxsplit=1)
                history_id = parts[1] if len(parts) == 2 else session.id
                for message in runtime.sessions.get(history_id).messages:
                    print(message)
            elif question == "/tools":
                for tool in runtime.registry.tools.values():
                    print(tool.name, tool.source, "并行" if tool.parallel else "串行")
            elif question == "/compact":
                session = runtime.sessions.get(session.id)
                if session.kind == "legacy":
                    raise ValueError("归档会话只供查看")
                if session.active_run_id and runtime.runs.get(session.active_run_id).status != RunStatus.COMPLETE:
                    raise ValueError("请先完成当前运行再压缩")
                print("摘要已保存" if runtime.context.compact(session) else "没有需要压缩的新历史")
            elif question == "/usage":
                session = runtime.sessions.get(session.id)
                if session.active_run_id is None:
                    print("当前会话暂无运行记录")
                else:
                    totals = runtime.runs.totals(session.active_run_id)
                    print(f"共 {totals['model_calls']} 次，主 Agent {totals['main_calls']} 次，"
                          f"子 Agent {totals['child_calls']} 次")
                    print(f"Token 合计 {totals['total_tokens']}，其中子 Agent {totals['child_tokens']}")
            elif question == "/help":
                print(HELP)
            else:
                console.has_text = False
                session = runtime.sessions.get(session.id)
                if question == "/resume":
                    if not session.active_run_id:
                        raise ValueError("当前会话没有可恢复的运行")
                    run = runtime.resume(session.active_run_id, args.max_steps, console.text)
                else:
                    run = runtime.create_run(session.id, question, max_steps=args.max_steps)
                    run = runtime.run(run.id, console.text)
                console.result(runtime, run)
                if args.task:
                    return 0 if run.status == RunStatus.COMPLETE else 1
            if args.task:
                return 0
        except (EOFError, KeyboardInterrupt):
            print("\n已退出；运行进度保存在 SQLite 中。")
            return 0
        except (ValueError, OSError, RuntimeError) as error:
            print(error)
            if args.task:
                return 1


def main():
    settings = Settings.load()
    parser = argparse.ArgumentParser(description="通用文件 Agent")
    parser.add_argument("--workspace", type=Path, default=settings.workspace)
    parser.add_argument("--db", type=Path, default=settings.database)
    parser.add_argument("--mcp", action="store_true", help="通过 MCP 调用文件工具")
    parser.add_argument("--session", help="使用指定会话")
    parser.add_argument("--task", help="执行一个任务后退出")
    parser.add_argument("--max-steps", type=int, default=6)
    args = parser.parse_args()
    settings.workspace, settings.database = args.workspace.resolve(), args.db.resolve()
    console = Console()
    with create_runtime(settings, args.mcp, console.event) as runtime:
        exit_code = chat(runtime, settings, args, console)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
