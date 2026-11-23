import argparse
import os
import select
import signal
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from queue import Empty, Queue
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


HELP = "/new 新会话 | /sessions 会话列表 | /use 编号 切换会话 | /history [编号] 查看历史\n/resume 继续未完成任务 | /steer 内容 补充运行指令 | /compact 压缩历史 | /tools 工具列表 | /usage 使用统计 | Ctrl+C 取消运行 | exit 退出"


@contextmanager
def cancel_on_interrupt():
    requested = False

    def handle_interrupt(signum, frame):
        nonlocal requested
        if not requested:
            requested = True
            os.write(2, "\n已请求取消，等待当前步骤结束。\n".encode())

    previous = signal.signal(signal.SIGINT, handle_interrupt)
    try:
        yield lambda: requested
    finally:
        signal.signal(signal.SIGINT, previous)


def read_command(prompt=""):
    if sys.stdin.isatty():
        return input(prompt).strip()
    # 管道输入逐行读取，避免提前缓冲后续命令，导致 select 看不到它们。
    line = sys.stdin.buffer.raw.readline()
    if not line:
        raise EOFError
    return line.decode(sys.stdin.encoding).strip()


def run_interactive(execute, run_id, console, cancel_requested):
    steering = Queue()
    next_command = None
    with ThreadPoolExecutor(max_workers=1) as worker:
        future = worker.submit(execute, cancel_requested=cancel_requested, steering=steering)
        try:
            while not future.done():
                if not select.select([sys.stdin], [], [], 0.1)[0]:
                    continue
                try:
                    command = read_command()
                except EOFError:
                    break
                if not command:
                    continue
                parts = command.split(maxsplit=1)
                if parts[0] == "/steer":
                    if len(parts) == 2:
                        steering.put(parts[1])
                        console.event(run_id, "补充指令已排队")
                    else:
                        console.event(run_id, "用法：/steer 补充要求")
                elif future.done() or not sys.stdin.isatty():
                    next_command = command
                    break
                else:
                    console.event(run_id, "运行中可输入 /steer 补充要求，或按 Ctrl+C 取消")
            return future.result(), next_command
        finally:
            # 最后一个检查点可能已过去，让用户知道哪些指令来不及应用。
            while future.done():
                try:
                    unused = steering.get_nowait()
                except Empty:
                    break
                console.event(run_id, f"本次运行未采用这条指令：{unused}")


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
    next_command = None
    while True:
        try:
            question = args.task or next_command or read_command("你：")
            next_command = None
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
                if session.active_run_id and not runtime.runs.get(session.active_run_id).status.finished:
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
            elif question.split(maxsplit=1)[0] == "/steer":
                print("/steer 用于正在执行的任务。")
            else:
                console.has_text = False
                session = runtime.sessions.get(session.id)
                with cancel_on_interrupt() as cancel_requested:
                    if question == "/resume":
                        if not session.active_run_id:
                            raise ValueError("当前会话没有可恢复的运行")
                        run_id = session.active_run_id
                        execute = partial(runtime.resume, run_id, args.max_steps, console.text)
                    else:
                        run = runtime.create_run(session.id, question, max_steps=args.max_steps)
                        run_id = run.id
                        execute = partial(runtime.run, run_id, console.text)
                    if args.task:
                        run = execute(cancel_requested=cancel_requested)
                    else:
                        run, next_command = run_interactive(execute, run_id, console, cancel_requested)
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
