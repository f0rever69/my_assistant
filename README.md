# my_assistant

面向个人本地使用的 Agent Harness，提供终端交互、单次问答和 Python 嵌入式调用。

## 目录结构

```text
my_assistant/
├── config.yaml
├── extensions_config.json
├── skills/
└── packages/
    ├── harness/       # Agent、工具、中间件、记忆、沙箱和 TUI
    └── extension-api/ # 扩展接口契约
```

## 安装

```powershell
uv sync
```

可选依赖：

```powershell
uv sync --extra postgres
uv sync --extra redis
uv sync --extra browser
```

## 运行

```powershell
# 交互式终端
uv run my_assistant

# 单次问答
uv run my_assistant --print "用一句话介绍你自己"

# JSON 事件流
uv run my_assistant --json "你好"

# 恢复最近会话
uv run my_assistant --continue

# 将 Agent 文件操作限制在指定项目目录
uv run my_assistant --workspace D:\projects\my-project
```

使用 `--workspace` 时，Agent 还可以用 FTS5 搜索工作区文本文件；工具会合并模型生成的多组关键词查询，再返回标题、命中片段和可供 `read_file` 使用的虚拟路径。中文默认使用 jieba 分词，并叠加字符二元词提高相近表述的召回；若 jieba 不可用，则退回字符二元词检索。

终端对话、工具回执、进度和状态文字支持鼠标拖选，再按 `Ctrl+C` 复制；输入框的已选文字也可直接复制。有选中文字时，复制不会中断任务；没有选中时，`Ctrl+C` 仍用于中断或退出。按 `Esc` 先取消文字选择。

按 `F8` 或输入 `/copy`（也可 `/copy all`）打开只读文本视图，支持鼠标选择、`Ctrl+A` 全选、`Ctrl+C` 复制及“复制全文”按钮，`Esc` 关闭；`/copy last` 只查看最近一条消息。复制长命令、代码及令牌时推荐使用文本视图，其保留原始消息换行，软换行不会进入剪贴板。视图是打开时的快照，后台任务继续运行。Windows 优先写入系统剪贴板，其他平台通过终端的 OSC 52 机制发送复制请求（取决于终端支持）。

## Workspace 注册与线程隔离

首次使用项目目录时，`--workspace PATH` 会在当前用户的本地注册表中创建稳定的
`workspace_id`。注册表位于 `.my_assistant/users/<user-id>/workspaces/registry.sqlite3`，保存显示名、规范化根路径、根目录指纹以及读写范围、忽略规则、Shell 和网络开关。

启动时也可以直接选择已注册的工作区：

```powershell
uv run my_assistant --workspace-id ws-<32位十六进制ID>
uv run my_assistant --workspace-id "项目名称"
```

TUI 中使用 `/workspace` 打开选择器，或使用下面的命令管理工作区：

| 命令 | 作用 |
|---|---|
| `/workspace list` | 列出当前用户的注册表、可用状态、根路径和能力策略 |
| `/workspace show` | 显示当前选中工作区的完整策略 |
| `/workspace refresh [--dry-run]` | 预览或将本地变化同步到当前线程的 Docker 工作副本；冲突时整次拒绝刷新 |
| `/workspace conflicts [path]` | 打开冲突审阅弹窗，直接选择保留本地或采用沙箱写回；多个冲突先选择文件 |
| `/workspace resolve <path> local\|sandbox <token>` | 备份双方版本后选择保留本地或沙箱；令牌绑定文件版本及当前用户、线程和策略 |
| `/workspace add <path> [name]` | 注册本地目录并切换到它；同一路径会复用稳定 ID |
| `/workspace use <id\|name\|local>` | 切换工作区，`local` 表示不绑定项目工作区 |
| `/workspace return` | 切回当前线程原来绑定的工作区并继续查看该线程 |
| `/workspace policy <id\|name> {JSON}` | 更新读写范围、忽略规则、Shell 或网络开关 |
| `/workspace remove <id\|name> confirm` | 删除注册项（tombstone），保留会话和任务元数据 |
| `/workspace migrate <id\|name> confirm` | 将未完成任务迁移到新线程，只复制目标、约束和待办步骤 |
| `/workspace export <relative-path>` | 在当前源工作区创建不超过 1 MiB 的 UTF-8 文本快照 |
| `/workspace exports` | 列出当前用户可引用的快照清单和完整 ID |
| `/workspace reference <export-id>` | 将快照复制到当前目标工作区线程，作为不可信数据引用 |
| `/workspace forget-export <export-id> confirm` | 撤销未来引用；已附加到线程的快照不会被删除 |

切换工作区会清空当前显示并进入新会话，不会把旧线程改绑。使用 `/resume` 恢复旧线程后，可以用 `/workspace return` 切回其原工作区。任务迁移创建一个暂停的新线程，先用 `/task show` 核对，再用 `/task continue` 继续；历史消息、文件事务、搜索索引和已修改文件不会复制。

策略采用相对工作区根目录的路径或模式；空的读写范围表示拒绝该类文件访问。例如，将某个工作区设为只读并关闭 Shell、网络：

```text
/workspace policy "项目名称" {"allowed_read_paths": ["."], "allowed_write_paths": [], "shell_enabled": false, "network_enabled": false}
```

默认忽略规则始终保护 `.env`、私钥、SSH 配置等敏感路径，自定义 `ignored_patterns` 只增加保护。Shell 和网络开关默认关闭，`--workspace-shell`、`--workspace-network` 会显式开启并保存对应策略。

每个线程和结构化任务都绑定工作区 ID、根路径及根目录指纹。恢复时如果当前工作区不匹配、注册被删除、根目录被替换或策略已变化，历史记录仍可查看，但模型调用和文件动作会被拒绝；系统不会按同名目录自动改绑。跨工作区复制只能通过 `/workspace export <relative-path>` 生成文本快照，再在目标工作区执行 `/workspace reference <export-id>`，快照会以不可信数据引用写入目标线程，不开放源工作区的实时路径、索引或任务状态。

当前配置使用 Docker AIO 沙箱。启动 Docker Desktop 的 Linux 容器引擎或本机 Docker Engine 28+ 后，可以运行 `uv run my_assistant --workspace <目录> --workspace-shell`。首次需要准备配置中指定的 AIO 镜像和网络代理镜像，Provider 负责创建和回收容器。模型调用和运行记录由宿主机上的 Harness 管理。

| 容器目录 | 宿主机内容 | 容器权限 |
| --- | --- | --- |
| `/mnt/user-data/workspace` | 当前线程的 Workspace 副本，位于 `<运行数据目录>/users/<用户ID>/threads/<线程ID>/docker-workspace/<策略指纹>/files` | 读写副本 |
| `/mnt/user-data/uploads` | Workspace 的 `.my-assistant/threads/<线程ID>/uploads` | 只读 |
| `/mnt/user-data/outputs` | Workspace 的 `.my-assistant/threads/<线程ID>/outputs` | 读写 |
| `/mnt/skills/...` | 已启用技能的投影目录 | 只读 |

省略 `--workspace` 时，前三项映射到线程自己的 `user-data/workspace`、`user-data/uploads` 和 `user-data/outputs`。绑定 Workspace 后，首次创建副本会筛掉受保护文件及符号链接，副本随同一线程保留。

本地新增、修改或删除文件后，可在 Agent 空闲时执行 `/workspace refresh --dry-run` 预览，再用 `/workspace refresh` 同步到已有副本，无需重建线程或容器。刷新比较同步基线、本地文件和副本：仅本地变化时同步，仅副本变化时保留，两边变成相同内容时对齐基线；同一路径两边改成不同内容，或出现不安全路径、文件与目录类型冲突时，整次预检查拒绝刷新。返回 `changes`、`preserved` 和 `conflicts`，其中 `aligned` 表示两边已相同、只需更新基线。受保护文件、符号链接和读范围外文件仍不导入。该命令只修改副本及其基线，不写回本地源文件，也不清空沙箱独有文件。刷新与写回共用锁，每个文件原子替换并复核版本；磁盘错误等导致批次中断后可重试，但不提供跨文件崩溃原子性。TUI 在执行任务时拒绝刷新；Python 调用方也应确保该线程及其他直接写入副本的进程空闲。

Python 嵌入式调用可使用 `client.refresh_workspace_copy(thread_id, workspace_context, dry_run=True)` 预览，以及 `client.refresh_workspace_copy(thread_id, workspace_context)` 执行。

用 `/workspace diff` 查看变更列表，用 `/workspace diff <相对路径>` 查看文本差异和 SHA-256，然后用 `/workspace apply <相对路径> <SHA-256>` 将核对过的文件写回。写回检查权限、宿主文件原始版本和副本内容，事务状态可由现有文件恢复流程处理。删除会列在差异中，写回入口支持新增和修改。

文件工具回执会包含 `storage_scope`、`sync_status`、虚拟路径及对应本地路径：写入 Workspace 副本标为 `sandbox_workspace_copy / pending_apply`，本地直接文件操作标为 `host_workspace / host_updated`，共享输出目录单独标为 `host_outputs`。Shell 成功只表示命令执行成功，不证明本地原文件已更新。这些位置事实同时进入结构化工具回执。

回答前，运行时会读取实际文件状态；结束后还会追加一条“文件同步状态（程序核验）”，明确显示沙箱待写回、双方冲突、本地已写回等结果。判定同时检查当前字节及写回记录，不把“内容碰巧相同”或模型自行声称完成当作写回证明。流式界面保留模型正文，并在其后显示独立核验结果；Python `chat()` 返回正文和核验说明。目录扫描暂不可用时会明确标注无法核验。

遇到冲突时，输入 `/workspace conflicts test.txt` 打开弹窗，查看双方差异及版本哈希，点击“保留本地”将本地版本同步到沙箱，或点击“采用沙箱写回”将沙箱版本写回本地。也可按 `L` / `S` 选择，按 `Esc` 取消；默认焦点为“取消”，直接回车不覆盖文件。程序自动携带本次审核的版本令牌，无需复制或手填。`/workspace conflicts` 在只有一个冲突时直接打开弹窗，多个冲突先选择文件；详情支持全选复制。

选择某侧不存在的文件表示接受该侧的删除。冲突处理只允许在空闲时由用户执行，模型不能自行批准；审阅后任一文件、基线或策略变化都会使旧令牌失效，切换线程、工作区或开始任务后也拒绝执行旧选择，需重新审阅。覆盖或删除前，双方原内容及操作日志保存到该线程工作副本目录旁的 `conflict-backups/<操作ID>/`（不挂载给容器），处理结果返回具体备份路径。采用沙箱文件写回使用现有文件事务；删除保留独立备份和准备/完成日志。手工合并时，可先在编辑器中保存合并结果，再重新审核并选择该侧版本。原手动命令 `/workspace resolve <path> local|sandbox <token>` 仍可使用。

对应嵌入式接口为 `list_workspace_conflicts`、`review_workspace_conflict` 和 `resolve_workspace_conflict`，同样校验线程归属、工作区策略和版本令牌。`resolve` 属于调用方明确授权的管理操作，不注册为模型工具。

本机 Docker 沙箱启用内置 seccomp、CPU、内存和进程数限制。默认配置只开放内部控制面，容器应用的对外访问受到隔离网络限制。命令超时会终止容器，回合结束时释放容器并保留文件副本。

## 个人知识库问答

先选择工作区内的资料目录，再建立索引。索引按中文 BM25 返回带文件版本和位置的段落。支持 UTF-8 Markdown、TXT、文本型 PDF、DOCX。扫描件会报告没有可提取文字；更新或删除资料后重新运行 `index` 即可刷新，省略 `--source` 会继续使用上次选定的目录。

```powershell
uv run my_assistant kb --workspace D:\projects\my-project index --source docs
uv run my_assistant kb --workspace D:\projects\my-project list
uv run my_assistant kb --workspace D:\projects\my-project search "差旅报销期限" --variant "出差 提交"
uv run my_assistant kb --workspace D:\projects\my-project ask "报销期限是多少？" --allow-model-data
uv run my_assistant kb --workspace D:\projects\my-project audit
```

使用真实模型时，问题和选中的资料段落会进入模型请求。CLI 可通过 `--allow-model-data` 为本次调用开启，或者在 `config.yaml` 配置 `rag.allow_model_data: true` 后在终端使用 `/askkb 报销期限是多少？`。索引和审计存放在当前用户的本地运行目录。审计记录包含 `trace_id`、索引版本、候选 ID、核验状态和耗时，省略问题原文、文件路径、摘录和模型响应。

```yaml
rag:
  allow_model_data: false
  output_mode: strict
  # model: deepseek-flash
  # verifier_model: deepseek-flash
  audit_max_runs: 500
```

`strict` 只发布通过核验的结论，并标明证据不足或来源冲突。`annotate` 额外显示明确标记的候选结论。程序在发布前重新检查文件权限、文件版本、引用位置与原文摘录；语义核验由模型完成。使用 `ask --json` 可获得逐条状态和结构化引用。

离线测试可用 `ask "问题" --mock fixture.json --json`。从 `search` 输出获取当前 `chunk_id`，将文件原文连续摘录填入 `quote`。示例：

```json
{
  "extract": {
    "topics": [{
      "question": "报销期限是多少？",
      "claims": [{
        "text": "出差结束后30天内提交报销。",
        "quotes": [{"chunk_id": "<search 返回的 chunk_id>", "quote": "出差结束后30天内提交报销。"}]
      }]
    }]
  },
  "verify": {
    "judgments": [{"claim_id": "1", "verdict": "supported", "evidence_ids": ["1.1"]}]
  }
}
```

模拟模式替换事实提取和核验的模型响应，仍会执行真实索引查询、引用校验、输出策略和本地审计。测试文件为 `packages/harness/tests/test_rag_*.py`。

## Python 调用

```python
from my_assistant.client import MyAssistantClient

client = MyAssistantClient()
print(client.chat("你好", thread_id="t1"))
```

## 配置

`config.yaml`、`.env` 和 `skills/` 会从项目根目录自动发现。`.env` 用于提供模型等服务所需的环境变量，不要提交或放进 Agent 可访问的工作区。

常用环境变量：

| 变量 | 作用 |
|---|---|
| `MY_ASSISTANT_CONFIG_PATH` | 指定配置文件 |
| `MY_ASSISTANT_HOME` | 运行数据目录，默认位于项目根目录的 `.my_assistant` |
| `MY_ASSISTANT_SKILLS_PATH` | 指定技能目录 |
| `MY_ASSISTANT_EXTENSIONS_CONFIG_PATH` | 指定扩展配置文件 |
| `LANGFUSE_TRACING` | 设为 `true` 启用 Langfuse 对话与工具调用追踪 |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | Langfuse 项目的 API 凭据；启用追踪时必填 |
| `LANGFUSE_BASE_URL` | Langfuse 地址，默认 `https://cloud.langfuse.com`；自托管或其他区域可覆盖 |
| `LANGFUSE_TRACING_ENVIRONMENT` | 可选的 Langfuse 环境标签，如 `development`、`production` |

Langfuse 追踪按对话轮次创建 trace，并按 `thread_id` 归入 session。发送到 Langfuse 的观测数据会遮蔽常见凭据、邮箱和中国手机号；根 trace 只保留本轮用户输入和最终回复，LLM 与工具的子观测仍保留实际上下文和入参/结果，便于排查执行过程。遮蔽规则无法识别所有业务隐私，启用前应确认数据可以发送到配置的 Langfuse 服务，或使用可信的自托管实例。

## 扩展方式

| 目标 | 位置 |
|---|---|
| 添加技能 | `skills/custom/<name>/SKILL.md` |
| 添加内置工具 | `packages/harness/my_assistant/tools/builtins/` |
| 添加中间件 | `packages/harness/my_assistant/agents/middlewares/` |
| 修改系统提示词 | `packages/harness/my_assistant/agents/lead_agent/prompt.py` |
| 添加模型 Provider | `packages/harness/my_assistant/models/`，并在配置中指定 `use` |
| 添加记忆后端 | `packages/harness/my_assistant/agents/memory/backends/` |
| 添加 Sandbox Provider | 实现 `SandboxProvider` 并配置 `sandbox.use` |
