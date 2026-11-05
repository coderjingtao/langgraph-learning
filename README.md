# LangGraph 官方文档体系化自学教程 (Python 3.13 + uv)

本项目根据 [LangGraph 官方文档](https://langchain-ai.github.io/langgraph/) 的核心架构与章节脉络，设计了 **12 个循序渐进、可独立运行的 Python 学习文件**。
每个 `.py` 文件对应一个核心知识模块，内置详尽的**中文概念解析、代码注释与断言验证**，且**无需配置付费 API Key** 即可在本地直接运行验证结果。

---

## 🛠️ 环境信息与使用方法

本项目使用 `uv` 管理虚拟环境与依赖，固定使用 **Python 3.13 稳定版**。

### 1. 查看虚拟环境与 Python 版本
```bash
uv run python --version
# 输出: Python 3.13.x
```

### 2. 单独运行任意章节脚本
```bash
uv run python 01_hello_langgraph.py
uv run python 02_state_and_reducers.py
# ... 以此类推
```

### 3. 一键运行全部章节验证
```bash
uv run python run_all.py
```

---

## 📚 章节目录与知识点导航

| 章节文件 | 核心主题 | 官方文档对应核心概念 |
| :--- | :--- | :--- |
| [`01_hello_langgraph.py`](./01_hello_langgraph.py) | **基础入门：第一个状态图** | `StateGraph`、`State`、`Nodes`、`Edges`、`START` / `END`、`compile()`、`invoke()` |
| [`02_state_and_reducers.py`](./02_state_and_reducers.py) | **状态管理与归约器** | 默认覆盖行为、`Annotated`、`operator.add`、自定义 Reducer、`input_schema` / `output_schema` |
| [`03_conditional_edges.py`](./03_conditional_edges.py) | **条件边与循环控制** | `add_conditional_edges`、动态路由函数（Routing Function）、Loop 循环反思、`recursion_limit` |
| [`04_messages_and_chat.py`](./04_messages_and_chat.py) | **消息状态与对话机器人** | `MessagesState`、`add_messages` 归约器、按消息 ID 原地更新、`RemoveMessage` 上下文裁剪 |
| [`05_tools_and_react_agent.py`](./05_tools_and_react_agent.py) | **工具调用与 ReAct 架构** | `@tool` 定义、`AIMessage.tool_calls`、预构建 `ToolNode`、`tools_condition` 循环 |
| [`06_memory_and_persistence.py`](./06_memory_and_persistence.py) | **短期记忆与时间旅行** | `InMemorySaver` (Checkpointer)、`thread_id` 多会话隔离、`get_state`、`get_state_history` 分叉回溯 |
| [`07_human_in_the_loop.py`](./07_human_in_the_loop.py) | **人机协同与动态中断** | `interrupt()` 动态暂停、`__interrupt__` 载荷检查、`Command(resume=...)` 审批恢复 |
| [`08_streaming.py`](./08_streaming.py) | **流式输出全解析** | `stream_mode` (`"values"`, `"updates"`, `"custom"`)、`get_stream_writer()` 自定义流事件 |
| [`09_parallel_and_map_reduce.py`](./09_parallel_and_map_reduce.py) | **并行执行与 Map-Reduce** | 节点并发执行（Fan-out / Fan-in）、`Send` API 动态数量并行 Worker 分发与汇聚 |
| [`10_subgraphs.py`](./10_subgraphs.py) | **子图与多智能体编排** | 子图（Subgraphs）嵌套、状态 Schema 共享与隔离、使用 `Command(update=..., goto=...)` 路由 |
| [`11_long_term_memory_store.py`](./11_long_term_memory_store.py) | **跨会话长期记忆存储** | 短期记忆 vs 长期记忆、`InMemoryStore` (`BaseStore`)、`namespace`、`put` / `search` 依赖注入 |
| [`12_functional_api.py`](./12_functional_api.py) | **函数式 API 编排** | `@entrypoint`、`@task` 异步并发与 Checkpoint 幂等缓存、原生 Python 控制流结合 `interrupt` |
