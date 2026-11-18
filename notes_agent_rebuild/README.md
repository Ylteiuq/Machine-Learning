# 通用文件 Agent

目录沿用已有的 `notes_agent_rebuild` 骨架，运行逻辑已经改为通用文件 Agent。
使用 Python 3.11、原生 HTTP 请求和 SQLite；MCP 只负责工具通信。

## 启动

在仓库根目录、`agent` Python 环境中执行：

```bash
python -m notes_agent_rebuild
python -m notes_agent_rebuild --mcp
python -m notes_agent_rebuild --workspace /absolute/path/to/workspace --mcp
```

默认工作目录是本目录的 `workspace/`，数据库是 `data/agent.db`。
文件路径相对于工作目录解析。默认直接调用本地文件工具；`--mcp` 显式改为调用文件服务。
两种模式复用同一个 `FileTools`，不会在 MCP 失败时悄悄回退到本地执行。
需要安装依赖时使用 `python -m pip install -r notes_agent_rebuild/requirements.txt`。
模型配置放在本目录的 `.env`，字段见 `.env.example`，启动时不会读取旧项目的配置。

执行一个任务后退出：

```bash
python -m notes_agent_rebuild --mcp --task "列出工作目录中的 Markdown 文件"
```

终端命令：`/new`、`/sessions`、`/use 会话编号`、`/history [会话编号]`、`/resume`、`/compact`、`/tools`、`exit`。
启动时恢复当前工作目录最近的主会话，但不会自动执行未完成任务；使用 `/resume` 继续。
也可以通过 `--session 编号` 指定主会话。

## 建议阅读顺序

1. `app.py`：集中组装模型、数据库、工具注册表和运行时。
2. `agent/state.py`：会话、运行、工具调用上下文三个数据结构。
3. `agent/runtime.py`：`create_run → run → step → execute_batch`，主子 Agent 共用。
4. `tools/registry.py`：schema、handler、parallel、source 集中在一个 Tool 中。
5. `agent/subagent.py`：委派工具调用同一个 Runtime，建立独立会话和运行。
6. `storage/`：消息和执行状态在一个事务中提交，恢复时只访问 SQLite。
7. `tools/mcp_client.py` 与 `files_mcp_server.py`：相同文件工具的跨进程调用。

`main.py` 只处理终端交互。工具实现不导入 main，Runtime 不识别具体工具名称。
委派工具通过 `app.py` 获取 Runtime 实例，注册成普通工具；这是依赖注入，不是模块互相导入。
Runtime 持有公共依赖，每次运行的 Session、RunState 是局部变量，所以不同子任务可以并发使用同一实例。

## 工具和调度

- `list_files`：glob 列出文件，默认跳过隐藏路径。
- `read_file`：读取 UTF-8 文本，可指定行范围，返回下一段起始行。
- `search_files`：在文本文件中做不区分大小写的子串搜索，最多扫描 10000 个文件。
- `write_file`：写入完整内容，覆盖已有文件；相同内容重放仍是替换，不做追加。
- `remember_preference`：明确要求记忆时保存偏好。
- `delegate_task`：创建只读子任务，返回状态、报告和模型调用数。

注册工具需要提供 `handler(arguments, context)`。context 包含 run_id、session_id 和 call_id。
增加工具时注册一个 Tool 即可，不需要在 Runtime 中添加名称分支。

模型同一轮返回的连续并行工具会组成批次，线程池最多两路，结果按原请求顺序回填。
委派工具允许并行；主 Agent 等待整个批次完成。子 Agent 只能列举、搜索、读取，不能再委派或写文件。
主运行默认最多 6 次模型请求，子运行最多 4 次；这是每个运行的预算，不是全树总预算。
`/resume` 给当前运行增加请求空间；最终统计汇总父子运行的实际调用次数。

## 数据与恢复

SQLite 表：

- `sessions`：上下文归属、摘要、压缩游标、工作目录、当前运行。
- `messages`：按会话和位置存储消息，只追加新的消息。
- `runs`：父子关系、阶段、状态、工具范围、待执行队列和调用统计。
- `preferences`：长期偏好。
- `imports`：显式迁移记录。

消息和工具参数中的嵌套结构用 JSON 字段表示；不存在 JSON 文件和数据库之间的运行期同步。
摘要只改变摘要与游标，完整历史仍在 messages。主、子 Agent 的消息属于不同会话。
同一个父运行的同一个委派 call_id 对应已有子运行，重放时复用其结果或继续其进度。

运行恢复属于步骤级恢复。工具执行与外部文件写入不是一个 SQLite 事务：
若在文件写完、运行进度提交前进程退出，该工具可能重放；目前的完整文件替换可以重放。
这里没有实现任意外部写操作的“恰好一次”保证，也没有实现后台 Agent 的取消与干预。

## 旧数据迁移

```bash
python -m notes_agent_rebuild.migrate_legacy
```

迁移时优先从旧 `notes.db` 导出；仅当旧数据库不存在时读取旧 `notes.json`。
笔记按原编号导出到 `workspace/legacy_notes/n编号.md`。
旧 session、checkpoint、子运行各自归档到新数据库，保留各份历史，避免覆盖分叉记录。
旧运行涉及已经移除的笔记工具，因此只供查看，不能在新的文件工具上直接重放。
旧偏好导入数据库，同名的新偏好优先。源文件和数据库在 `data/legacy_backup/` 留有备份。
重复执行根据 imports 表跳过，正常启动不运行迁移。

迁移旧会话不等于继续旧业务；使用 `/new` 创建通用文件 Agent 会话，使用 `/history 编号` 查阅归档。
