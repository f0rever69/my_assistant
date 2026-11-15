# MyAssistant 一次对话的完整执行链路分析

下面按当前项目的真实执行方式，分析一次普通对话：

```text
CLI / Python 调用
      │
      ▼
MyAssistantClient.stream() / chat()
      │
      ├─ 读取配置、解析 thread_id、建立运行上下文
      ├─ 构建或复用 Lead Agent
      ├─ 加载模型、工具、技能、中间件、记忆和沙箱
      ├─ 调用 LangGraph Agent
      │      ┌─────────────────────────────┐
      │      │ before_agent                 │
      │      │ before_model                 │
      │      │ LLM                         │
      │      │ after_model                 │
      │      │ tools / ToolMessage          │
      │      │ 循环回到 before_model        │
      │      │ after_agent                  │
      │      └─────────────────────────────┘
      ├─ 持久化 checkpoint / 事件 / 记忆
      └─ 转换成 StreamEvent 或最终字符串
```

分析范围是：

```python
client = MyAssistantClient()
client.chat("用户问题", thread_id="t1")
```

CLI 最终也是调用这套客户端：

- `my_assistant --print` -> `client.chat()`
- `my_assistant --json` -> `client.stream()`
- TUI -> `client.stream()`

代码入口主要在 `packages/harness/my_assistant/client.py` 和 `packages/harness/my_assistant/tui/cli.py`。

---

## 一、LLM 和 Harness 的能力边界

### 1. LLM 本身拥有的能力

LLM 是一次“概率模型调用”，主要负责：

| LLM 能力 | 说明 |
|---|---|
| 理解用户输入 | 从消息中识别意图、约束和上下文 |
| 语言生成 | 生成自然语言回答、代码、结构化文本 |
| 推理和规划 | 根据上下文决定下一步做什么 |
| 工具选择 | 从 Harness 提供的工具 schema 中选择工具 |
| 工具参数生成 | 生成工具调用所需要的 JSON 参数 |
| 多轮决策 | 看到工具结果后继续推理 |
| 思考模式 | 如果模型支持，生成内部 reasoning 或 extended thinking |
| 视觉理解 | 只有模型支持 vision 且 Harness 注入图片时才具备 |
| Token 流式输出 | Provider 支持时，逐 token 返回结果 |

LLM 输出的核心形式大致是：

```text
AIMessage(
    content="最终回答"
)
```

或者：

```text
AIMessage(
    content="",
    tool_calls=[
        {
            "name": "web_search",
            "args": {"query": "..."},
            "id": "call_123"
        }
    ]
)
```

LLM 可以“提出”调用 `web_search`，但不能自己执行搜索。

### 2. LLM 不具备的能力

以下能力不是 LLM 本身拥有的，而是 Harness 或外部系统提供的：

- 真实访问互联网
- 读取和写入本地文件
- 执行 Bash 或其他命令
- 访问 MCP 服务
- 调用数据库
- 启动子 Agent
- 保存多轮上下文
- 保存和检索长期记忆
- 维护任务状态
- 限制 token、递归次数和并发数
- 工具权限控制
- 沙箱隔离
- 路径穿越防护
- 失败重试和错误转化
- 运行审计
- 流式事件分发
- 用户澄清暂停和恢复
- 对话标题生成
- 上下文压缩
- 运行中止和资源清理

因此可以把边界概括为：

```text
LLM：决定“应该说什么、下一步做什么、调用哪个工具”
Harness：决定“工具是否存在、是否允许调用、如何执行、结果如何返回、状态如何保存”
```

### 3. 工具调用不是 LLM 的执行能力

LLM 只生成工具调用意图：

```text
LLM -> tool_call(name="read_file", args={...})
```

真正的执行流程是：

```text
Harness 收到 tool_call
    ↓
根据工具名查找 BaseTool
    ↓
执行工具中间件
    ↓
检查权限、沙箱、参数和预算
    ↓
调用真实 Python 函数 / 外部服务
    ↓
生成 ToolMessage
    ↓
把 ToolMessage 放回对话状态
    ↓
再次调用 LLM
```

这是整个系统最关键的能力边界。

---

## 二、一次对话的完整执行链路

### 阶段 0：入口选择

#### CLI 入口

定义在：

```text
packages/harness/my_assistant/tui/cli.py
```

调用示例：

```powershell
uv run my_assistant --print "介绍一下这个项目"
uv run my_assistant --json "搜索最新信息"
uv run my_assistant
```

CLI 的主要工作是：

1. 解析参数。
2. 判断使用 TUI、普通输出还是 JSON 流。
3. 创建会话。
4. 创建 `MyAssistantClient`。
5. 调用 `chat()` 或 `stream()`。

#### Python 入口

```python
from my_assistant.client import MyAssistantClient

client = MyAssistantClient()
answer = client.chat("你好", thread_id="t1")
```

此入口在当前进程中直接调用 Agent 图，属于嵌入式本地链路。

### 阶段 1：初始化 MyAssistantClient

构造函数在：

```text
packages/harness/my_assistant/client.py
```

`MyAssistantClient.__init__()` 主要做这些事情：

1. 加载 `config.yaml` 和 `.env`。
2. 得到 `AppConfig`。
3. 读取模型、沙箱、记忆、技能、数据库等配置。
4. 固定 checkpoint 模式和 snapshot frequency。
5. 保存调用者传入的选项：

```text
model_name
thinking_enabled
subagent_enabled
plan_mode
agent_name
available_skills
middlewares
```

6. 不立即构建 Agent，而是延迟到第一次调用。

因此：

```python
client = MyAssistantClient()
```

通常只完成配置加载，真正的模型和工具装配发生在第一次 `stream()` 或 `chat()`。

### 阶段 2：进入 stream()

`chat()` 实际上只是对 `stream()` 的封装：

```text
chat()
  └─ 调用 stream()
       └─ 累积最后一个 AI 消息的文本
```

`stream()` 的第一步是：

#### 2.1 解析线程 ID

```python
thread_id = resolve_thread_id(thread_id)
```

如果调用者没有传入，就生成一个新的线程 ID。

线程 ID 的作用是：

- 区分不同对话
- 关联 checkpoint
- 隔离上传文件
- 隔离沙箱路径
- 关联记忆和运行事件
- 恢复历史会话

#### 2.2 创建 RunnableConfig

配置中会放入：

```text
thread_id
model_name
thinking_enabled
is_plan_mode
subagent_enabled
recursion_limit
```

并通过 `configurable` 传给 LangGraph。

#### 2.3 注入 checkpoint 模式

当前配置是：

```yaml
database:
  backend: sqlite
  checkpoint_channel_mode: full
```

因此会使用 SQLite checkpoint。

如果配置为 delta 模式，Harness 会增加 checkpoint 历史缓存和增量状态处理。

#### 2.4 创建 trace 上下文

Harness 为每个请求生成或继承一个 trace ID，用于：

- 日志关联
- Langfuse
- Monocle
- 子 Agent 追踪
- 工具调用追踪

#### 2.5 构造运行上下文

上下文中包含：

```text
thread_id
run_id
user_id
权限信息
agent_name
trace_id
环境信息
```

这些信息不会全部直接交给 LLM，而是被中间件、工具和审计逻辑使用。

### 阶段 3：首次调用时构建 Agent

`stream()` 调用：

```python
self._ensure_agent(config, context=context)
```

这是 Agent 装配的核心，位置是 `packages/harness/my_assistant/client.py`。

#### 3.1 确定当前用户和权限身份

Harness 会解析：

- `user_id`
- `user_role`
- OAuth 身份
- Channel 身份
- 内部调用标记
- RBAC 属性

然后用授权系统检查模型和工具。

相关模块：

```text
my_assistant.authz.principal
my_assistant.authz.tool_filter
my_assistant.authz.runtime
```

#### 3.2 确定模型

模型选择顺序大致是：

```text
本次请求 model_name
    ↓
自定义 Agent 的模型配置
    ↓
config.yaml 中第一个模型
```

本项目当前配置使用：

```yaml
name: deepseek-flash
use: my_assistant.models.patched_deepseek:PatchedChatDeepSeek
```

模型由 `my_assistant.models.factory.create_chat_model()` 创建。

这个工厂负责：

- 加载 Provider 类
- 注入 API Key
- 注入 endpoint
- 处理 thinking 参数
- 处理 reasoning effort
- 处理 max_tokens、temperature 等覆盖值
- 设置流式 chunk timeout
- 设置重试参数
- 绑定 tracing
- 兼容 OpenAI 风格 Provider

此时 Harness 只是创建模型客户端，不会主动产生回答。

#### 3.3 收集工具

工具由 `my_assistant.tools.get_available_tools()` 收集。

工具来源包括：

```text
config.yaml 中声明的工具
内置工具
MCP 工具
ACP Agent 工具
子 Agent 工具
视觉工具
记忆工具
技能工具
```

当前配置包括：

```text
web_search
web_fetch
image_search
ls
read_file
glob
grep
write_file
str_replace
bash
```

内置工具包括：

```text
present_file
ask_clarification
review_skill_package
task
batch_task
batch_status
cancel_batch
list_uploaded_files
list_background_tasks
cancel_background_task
view_image
```

工具可能被配置、权限、模型能力和当前 Agent 再次过滤。

#### 3.4 加载技能

技能目录默认是 `skills/public/`。每个技能通常包含：

```text
SKILL.md
scripts/
references/
templates/
```

技能系统主要做四件事：

1. 扫描技能。
2. 读取 frontmatter。
3. 根据启用状态和 Agent 权限筛选技能。
4. 把技能描述注入 system prompt，或延迟到需要时加载完整内容。

例如 `deep-research` 技能不会自动给 LLM 一个搜索结果，它只是提供研究方法和触发规则。真正的搜索仍由 `web_search` 工具完成。

#### 3.5 组装 MCP 工具

```text
extensions_config.json
    ↓
读取启用的 MCP Server
    ↓
创建 MCP Client
    ↓
发现远程工具
    ↓
转成 LangChain BaseTool
    ↓
加入 Agent 工具列表
```

MCP 服务可以通过 stdio、HTTP 或 SSE 连接。MCP 工具对 LLM 看起来和普通工具类似，但执行时会转发到远端 MCP Server。

#### 3.6 工具授权和延迟发现

工具进入模型前还会经过：

```text
apply_tool_authorization()
assemble_deferred_tools()
build_mcp_routing_middleware()
```

这里可能发生：

- 禁用某个工具
- 按角色过滤工具
- 把 MCP 工具从默认 schema 中隐藏
- 只在模型搜索到工具后动态加入
- 限制技能可使用的工具集合
- 检查工具是否属于当前 Agent 的允许范围

所以：

```text
系统中存在的工具 != LLM 当前能看到的工具
```

#### 3.7 生成系统提示词

系统提示词由 `my_assistant.agents.lead_agent.prompt.apply_prompt_template()` 生成，内容通常包括：

- Agent 身份
- 当前日期
- 工作原则
- 工具使用规则
- 子 Agent 使用规则
- 技能目录摘要
- 记忆上下文
- 文件操作约束
- 安全规则
- MCP 工具提示
- 计划模式提示
- 任务连续性提示

这一步是 Harness 把“平台规则”转成 LLM 可理解指令的地方。

#### 3.8 构建中间件链

中间件由 `my_assistant.agents.lead_agent.agent.build_middlewares()` 生成。

输入和模型调用层：

```text
InputSanitizationMiddleware
ToolOutputBudgetMiddleware
ToolResultSanitizationMiddleware
```

负责清洗不可信输入、清洗远程工具返回内容、限制工具输出大小，防止外部文本伪装成系统指令。

Agent 初始化层：

```text
ThreadDataMiddleware
UploadsMiddleware
SandboxMiddleware
```

负责注入线程数据、上传文件，创建线程沙箱，建立沙箱执行租约，准备用户数据和输出目录。

工具执行保护层：

```text
ToolReceiptMiddleware
GuardrailMiddleware
SandboxAuditMiddleware
ReadBeforeWriteMiddleware
ToolProgressMiddleware
ToolErrorHandlingMiddleware
```

负责工具调用审计、权限检查、沙箱审计、写文件前必须读取、工具进度上报以及工具异常到 ToolMessage 的转换。

Agent 行为层：

```text
SkillActivationMiddleware
SkillToolPolicyMiddleware
DurableContextMiddleware
SummarizationMiddleware
TodoMiddleware
TitleMiddleware
MemoryMiddleware
ViewImageMiddleware
```

负责技能激活、技能工具权限、持久化上下文、长上下文压缩、Todo 计划、自动标题、记忆写入和注入、图片理解。

循环和预算控制层：

```text
SubagentLimitMiddleware
LoopDetectionMiddleware
TokenBudgetMiddleware
TerminalResponseMiddleware
ModelLengthFinishReasonMiddleware
SafetyFinishReasonMiddleware
ClarificationMiddleware
```

负责限制子 Agent 总数、检测重复工具调用、限制 token、防止空响应、处理模型长度截断、处理 Provider 安全终止，以及将澄清请求转换成人机交互状态。

这些中间件的顺序很重要，它们共同包裹着模型节点和工具节点。

#### 3.9 创建 LangGraph Agent

最终由 `langchain.agents.create_agent(...)` 生成图，传入：

```text
model
tools
middleware
system_prompt
state_schema
checkpointer
```

图中有两个核心节点：

```text
model
tools
```

以及大量由中间件生成的节点：

```text
xxx.before_agent
xxx.before_model
xxx.after_model
xxx.after_agent
```

### 阶段 4：创建初始状态

用户输入被包装为：

```python
state = {
    "messages": [
        HumanMessage(
            content=message,
            additional_kwargs={"run_id": run_id}
        )
    ]
}
```

此时状态中可能还会从 checkpoint 恢复：

```text
历史消息
summary_text
title
todos
artifacts
sandbox 状态
目标状态
任务连续性信息
```

如果是同一个 `thread_id` 且存在 checkpointer，LangGraph 会恢复同一个线程的历史状态。

### 阶段 5：LangGraph 开始执行

```text
START
  ↓
before_agent 中间件
  ↓
before_model 中间件
  ↓
model 节点
  ↓
after_model 中间件
  ↓
有 tool_calls？
  ├─ 是 -> tools 节点 -> ToolMessage -> 回到 before_model
  └─ 否 -> after_agent -> END
```

LangChain 的核心模型节点大致相当于：

```python
messages = [system_message, *state["messages"]]
output = model.invoke(messages)
```

异步路径对应：

```python
output = await model.ainvoke(messages)
```

---

## 三、一次模型调用内部发生什么

### 1. Harness 准备发送给 LLM 的内容

```text
SystemMessage:
  Agent 规则
  工具规则
  技能描述
  记忆内容
  当前日期
  安全约束

HumanMessage:
  用户问题

历史消息:
  之前的 HumanMessage / AIMessage / ToolMessage

工具 schema:
  当前允许的工具名称、参数和描述
```

重要边界：

```text
system prompt、历史消息、工具 schema 都由 Harness 组织
LLM 只负责基于这些输入生成下一步输出
```

### 2. Provider 调用外部模型服务

当前模型是 `PatchedChatDeepSeek`，通过 API Key 和网络请求访问 DeepSeek 服务。此层可能发生：

- 网络错误
- API Key 错误
- 超时
- Provider 限流
- 模型上下文超限
- safety finish reason
- 输出长度截断

Harness 的 `LLMErrorHandlingMiddleware`、重试逻辑和 Provider 工厂会处理这些情况。

### 3. LLM 返回两类结果

最终回答：

```python
AIMessage(content="这是最终答案")
```

工具调用：

```python
AIMessage(
    content="",
    tool_calls=[
        {
            "name": "web_search",
            "args": {"query": "..."}
        }
    ]
)
```

工具调用表示 LLM 请求 Harness 执行工具，LLM 不决定工具的真实执行结果。

---

## 四、工具调用链路

假设模型决定调用 `read_file(path="...")`。

### 1. LangGraph 路由到 tools 节点

LangGraph 检查最后一个 AIMessage 是否包含 `tool_calls`。如果存在，就把执行转给 ToolNode。

### 2. 工具中间件包裹调用

```text
输入参数
  ↓
工具授权检查
  ↓
Guardrail 检查
  ↓
SandboxAudit
  ↓
ReadBeforeWrite / 读写策略
  ↓
ToolProgress
  ↓
具体工具执行
  ↓
ToolErrorHandling
  ↓
ToolReceipt
  ↓
ToolMessage
```

### 3. 工具函数执行真实动作

文件工具：

```text
my_assistant.sandbox.tools.read_file_tool()
    ↓
获取当前线程 Sandbox
    ↓
校验虚拟路径
    ↓
检查用户权限
    ↓
读取文件
    ↓
截断过大的输出
    ↓
返回字符串
```

Bash 工具：

```text
bash_tool()
    ↓
检查 allow_host_bash
    ↓
获取 SandboxProvider
    ↓
检查执行租约
    ↓
执行命令
    ↓
限制超时和输出长度
    ↓
返回结果
```

当前配置中：

```yaml
sandbox:
  use: my_assistant.sandbox.local:LocalSandboxProvider
  allow_host_bash: false
```

所以宿主机 Bash 默认不开放。

### 4. 生成 ToolMessage

```python
ToolMessage(
    content="工具执行结果",
    tool_call_id="call_123",
    name="read_file"
)
```

失败也尽量转换为模型可见的工具消息：

```python
ToolMessage(
    content="工具执行失败：...",
    status="error"
)
```

### 5. ToolMessage 回到模型

下一轮发送给 LLM 的历史变成：

```text
HumanMessage
AIMessage(tool_calls=[...])
ToolMessage(content="工具结果")
```

LLM 看到工具结果后可能继续调用工具、修正参数、调用另一个工具、直接回答或请求用户澄清。

---

## 五、典型工具循环

```text
第 1 次模型调用：
  LLM -> web_search("某主题")

Harness：
  执行 web_search
  -> 返回搜索结果

第 2 次模型调用：
  LLM -> web_fetch("某个搜索结果 URL")

Harness：
  执行 web_fetch
  -> 返回网页内容

第 3 次模型调用：
  LLM -> 最终总结

Harness：
  检测无 tool_calls
  -> 结束
```

实际状态变化：

```text
messages:
  HumanMessage
  AIMessage(tool_calls=web_search)
  ToolMessage(web_search result)
  AIMessage(tool_calls=web_fetch)
  ToolMessage(web_fetch result)
  AIMessage(final answer)
```

每个图 super-step 之后，LangGraph 可能把状态写入 checkpoint。

---

## 六、技能在链路中的位置

技能不是独立的 Agent，也不是传统意义上的工具。技能主要影响：

```text
技能加载
  ├─ system prompt 中的技能描述
  └─ 工具执行时的技能策略
```

以 `deep-research` 为例：

```text
用户问题
  ↓
模型判断触发 deep-research
  ↓
Harness 允许加载对应 SKILL.md
  ↓
技能内容进入上下文
  ↓
LLM 按技能规定组织研究流程
  ↓
LLM 调用 web_search / web_fetch
  ↓
Harness 执行真实网络访问
```

技能的作用是为 LLM 提供流程、规范、模板和工具使用策略；真正的动作仍由工具和 Harness 完成。

---

## 七、子 Agent 链路

如果主 Agent 调用：

```text
task(
    prompt="分析某个文件",
    subagent_type="general-purpose"
)
```

链路变为：

```text
Lead LLM
  ↓
生成 task 工具调用
  ↓
Harness 的 task_tool
  ↓
验证 subagent_type 和权限
  ↓
检查并发和总数量限制
  ↓
创建子 Agent 运行上下文
  ↓
构建子 Agent 图
  ↓
子 Agent 自己经历：
    配置
    模型调用
    工具循环
    checkpoint
    中间件
  ↓
子 Agent 返回结果
  ↓
Lead Harness 生成 ToolMessage
  ↓
Lead LLM 继续推理
```

子 Agent 受这些 Harness 能力限制：

```text
max_concurrent_subagents
max_total_subagents
allowed_subagents
queue_timeout
token budget
verification / receipt
```

项目还提供 delegation ledger、receipt verification、acceptance criteria、result status 和 acceptance verdict。这些都是 Harness 的结果治理能力。

---

## 八、记忆链路

当前配置：

```yaml
memory:
  enabled: true
  injection_enabled: true
  manager_class: mymem
  mode: middleware
```

### 1. 对话开始时读取记忆

```text
MemoryMiddleware
  ↓
MemoryManager
  ↓
mymem backend
  ↓
读取用户记忆
  ↓
截断到 token 预算
  ↓
注入 system context
  ↓
发送给 LLM
```

LLM 只是看到已经注入的记忆文本，它并不直接访问 `memory.json` 或 FTS5 数据库。

### 2. 对话结束后写入记忆

```text
本轮消息完成
  ↓
MemoryMiddleware 捕获对话
  ↓
MemoryManager 提取事实
  ↓
按置信度、类别和容量策略处理
  ↓
写入 memory.json / retrieval index
```

记忆后端可能使用：

```text
.my-assistant/users/default/memory.json
.my-assistant/.retrieval/memory-fts5.sqlite3
```

---

## 九、上下文压缩链路

当历史消息过长时：

```text
SummarizationMiddleware
  ↓
检测 token 或消息长度
  ↓
选择需要压缩的历史消息
  ↓
调用总结模型
  ↓
生成 summary_text
  ↓
移除旧消息
  ↓
保留 summary + 最近消息
  ↓
继续主 Agent 循环
```

这里可能发生一次额外的 LLM 调用：主模型调用不等于总结模型调用。总结模型也是 Harness 创建和管理的。

---

## 十、澄清和人工介入

如果 LLM 调用 `ask_clarification`：

```text
LLM 请求澄清
  ↓
ask_clarification 工具
  ↓
ClarificationMiddleware 拦截
  ↓
生成 Human Input 状态
  ↓
本轮暂停
  ↓
等待用户输入
  ↓
恢复同一个 thread_id
  ↓
继续 LangGraph
```

这不是模型自动等待，而是 Harness 保存中间状态并负责后续恢复。

---

## 十一、流式输出链路

`MyAssistantClient` 使用：

```python
agent.stream(
    state,
    config=config,
    context=context,
    stream_mode=["values", "messages", "custom"],
)
```

### 1. messages

用于 AI 文本增量、AI 工具调用和 ToolMessage，最终被转换为：

```python
StreamEvent(
    type="messages-tuple",
    data={...}
)
```

### 2. values

用于完整状态快照：

```python
StreamEvent(
    type="values",
    data={
        "title": ...,
        "summary_text": ...,
        "messages": ...,
        "artifacts": ...
    }
)
```

### 3. custom

用于中间件、工具或扩展上报的自定义事件。

### 4. end

Agent 完成后输出：

```python
StreamEvent(
    type="end",
    data={
        "usage": {
            "input_tokens": ...,
            "output_tokens": ...,
            "total_tokens": ...
        }
    }
)
```

`stream()` 还负责合并 token 增量、防止 AI 消息重复发送、合并 tool call fragments、累计 token 使用量、处理消息替换和注入 metadata。

---

## 十二、chat() 如何得到最终答案

`chat()` 不直接调用模型，只消费流：

```text
client.chat()
  ↓
client.stream()
  ↓
接收 messages-tuple
  ↓
按 message id 累积 AI 文本
  ↓
记录最后一个 AI 消息 id
  ↓
返回最后一个 AI 消息文本
```

因此 `chat()` 的返回值通常是最后一个 AIMessage 的最终文本。工具调用、工具结果、规划消息和中间 AI 消息不会直接出现在返回值中。

如需完整过程：

```python
for event in client.stream("问题"):
    print(event)
```

---

## 十三、Checkpoint 和持久化

当前项目配置使用 SQLite：

```yaml
database:
  backend: sqlite
  sqlite_dir: .my-assistant/data
```

checkpointer 由 `my_assistant.runtime.checkpointer.provider.get_checkpointer()` 创建，负责保存：

```text
消息状态
标题
摘要
Todo
Artifacts
线程上下文
任务状态
checkpoint metadata
```

持久化位置包括：

```text
.my-assistant/data/my_assistant.db
```

这使得：

```python
client.chat("第一轮", thread_id="t1")
client.chat("继续刚才的话题", thread_id="t1")
```

可以恢复同一个线程的历史。

需要区分：

```text
Checkpoint：保存 LangGraph 状态
Memory：保存跨对话长期事实
Run Event：保存运行过程事件
Artifact：保存文件产物
```

---

## 十四、一次对话的结束阶段

当模型返回没有 `tool_calls` 的 AIMessage 后：

```text
model
  ↓
after_model middlewares
  ↓
after_agent middlewares
  ↓
LangGraph END
```

结束时可能执行：

- 最终响应兜底
- safety finish reason 处理
- token 统计
- 标题写入
- memory flush
- checkpoint 写入
- sandbox lease 释放
- trace 完成
- stream 结束事件
- 运行资源清理

`client.py` 中的 `_stream_with_sandbox_lease_cleanup()` 会在流被提前关闭时尝试释放沙箱执行租约。

---

## 十五、完整时序图

```text
用户
 │
 │  输入问题
 ▼
CLI / Python Client
 │
 │  resolve_thread_id
 │  创建 RunnableConfig
 │  创建 run_id / trace_id
 ▼
MyAssistantClient.stream()
 │
 │  _ensure_agent()
 │
 ├── AppConfig
 ├── Model Factory
 ├── Tool Registry
 ├── Skill Storage
 ├── MCP Cache
 ├── Authorization
 ├── Memory Config
 ├── Sandbox Config
 ├── Middleware Builder
 ├── Prompt Builder
 └── Checkpointer
 │
 ▼
LangGraph Agent
 │
 ├── before_agent
 │     ├── ThreadData
 │     ├── Uploads
 │     └── Sandbox
 │
 ├── before_model
 │     ├── 输入清洗
 │     ├── 记忆注入
 │     ├── 技能激活
 │     ├── 摘要压缩
 │     ├── 工具筛选
 │     └── SystemMessage 合并
 │
 ├── Model Node
 │     └── 调用外部 LLM API
 │
 ├── after_model
 │     ├── token 统计
 │     ├── 安全终止检查
 │     ├── 循环检测
 │     ├── 空响应兜底
 │     └── 是否继续工具循环
 │
 ├── Tools Node
 │     ├── 工具授权
 │     ├── 沙箱审计
 │     ├── 文件/网络/MCP/子 Agent 执行
 │     ├── 错误转换
 │     └── 生成 ToolMessage
 │
 ├── 回到 before_model
 │
 └── after_agent
       ├── checkpoint
       ├── memory flush
       ├── title
       ├── trace
       └── cleanup
 │
 ▼
StreamEvent
 │
 ├── messages-tuple
 ├── values
 ├── custom
 └── end
 │
 ▼
chat()
 │
 └── 返回最后一个 AI 消息文本
```

---

## 十六、后续逐环节分析建议

建议按以下顺序深入：

1. `client.py`：请求入口、状态和流式输出。
2. `lead_agent/agent.py`：模型、工具、技能、中间件的装配。
3. `agents/factory.py`：Agent 图结构和 middleware feature assembly。
4. LangChain 的 `create_agent()`：`model -> tools -> model` 循环。
5. `models/factory.py`：LLM Provider 创建和参数转换。
6. `tools/tools.py`：工具发现、过滤、去重和注册。
7. `agents/middlewares/`：每一个运行时约束。
8. `sandbox/tools.py`：文件和命令实际执行。
9. `runtime/checkpointer/`：线程状态持久化。
10. `agents/memory/`：记忆读写和长期状态。
11. `subagents/`：子 Agent 和并发执行。
12. `client.py` 的 `StreamEvent` 转换：内部 LangGraph 事件如何变成外部输出。

最核心的一句话是：

```text
LLM 负责生成下一步决策；
LangGraph 负责推进状态机；
Harness 负责提供工具、执行副作用、施加策略、保存状态并把结果重新喂给 LLM。
```
