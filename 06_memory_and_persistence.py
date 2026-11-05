"""
第 06 章：短期记忆、检查点与时间旅行 (Checkpointer, Thread & Time Travel)
=========================================================================

【官方文档核心知识点】
默认情况下，编译后的 StateGraph 是“无状态”的——每次 `invoke` 结束后状态就会丢弃。
为了支持多轮对话上下文（短期记忆）、故障恢复以及时间旅行（Time Travel），LangGraph 引入了 **Checkpointer（检查点保存器）**：

1. `InMemorySaver` (或 `MemorySaver`)：
   - 在图的每个超步（Super-step）边界自动保存当前的状态快照（Checkpoint）。
   - 编译时传入：`builder.compile(checkpointer=InMemorySaver())`。
   - 生产环境中可替换为 `SqliteSaver` 或 `PostgresSaver` 将状态持久化到数据库。
2. `thread_id`（会话线程 ID）：
   - 启用 Checkpointer 后，调用 `invoke` 时**必须**通过 `config={"configurable": {"thread_id": "..."}}` 指定线程 ID。
   - 不同的 `thread_id` 之间状态完全隔离（轻松支持多用户并发聊天）。
3. 状态查询与时间旅行（Time Travel）：
   - `app.get_state(config)`：获取指定线程的最新状态快照（`StateSnapshot`）。
   - `app.get_state_history(config)`：获取该线程从头到尾的所有历史检查点列表。
   - `app.update_state(config, values)`：手动修改某个线程的状态，或者从某个历史 `checkpoint_id` 分叉（Fork）重新执行！

运行方式：
    uv run python 06_memory_and_persistence.py
"""

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph


# ============================================================================
# 1. 定义带计数器的对话状态与节点
# ============================================================================
class MemoryDemoState(MessagesState):
    turn_count: int


def assistant_node(state: MemoryDemoState) -> dict:
    turn = state.get("turn_count", 0) + 1
    user_messages = [m.content for m in state["messages"] if isinstance(m, HumanMessage)]
    reply = f"[第 {turn} 轮回复] 我记得你一共说过 {len(user_messages)} 句话：{user_messages}"
    print(f"  [Node: assistant] -> {reply}")
    return {
        "messages": [AIMessage(content=reply)],
        "turn_count": turn,
    }


# ============================================================================
# 2. 编译带 Checkpointer 的图
# ============================================================================
def build_persistent_graph():
    builder = StateGraph(MemoryDemoState)
    builder.add_node("assistant", assistant_node)
    builder.add_edge(START, "assistant")
    builder.add_edge("assistant", END)

    # 关键步骤：实例化 InMemorySaver 并传入 compile(checkpointer=...)
    memory = InMemorySaver()
    return builder.compile(checkpointer=memory)


# ============================================================================
# 3. 独立执行与验证（多轮记忆、多会话隔离、时间旅行回溯）
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("第 06 章：短期记忆、检查点与时间旅行 (Checkpointer & Time Travel)")
    print("=" * 60)

    app = build_persistent_graph()

    # 配置两个不同用户的会话线程
    config_alice = {"configurable": {"thread_id": "user-alice-001"}}
    config_bob = {"configurable": {"thread_id": "user-bob-002"}}

    print("\n【1. 同一线程 (Alice) 的多轮连续对话（自动记住上下文）】")
    app.invoke({"messages": [HumanMessage(content="你好，我是 Alice")]}, config=config_alice)
    app.invoke({"messages": [HumanMessage(content="我最喜欢的编程语言是 Python")]}, config=config_alice)

    print("\n【2. 另一线程 (Bob) 的隔离验证（不会串台看到 Alice 的记录）】")
    bob_state = app.invoke({"messages": [HumanMessage(content="你好，我是 Bob")]}, config=config_bob)
    assert bob_state["turn_count"] == 1
    assert len(bob_state["messages"]) == 2

    print("\n【3. 检查 Alice 当前的状态快照 (app.get_state)】")
    current_snapshot = app.get_state(config_alice)
    print(f"  当前轮次 (turn_count): {current_snapshot.values['turn_count']}")
    print(f"  当前消息总数: {len(current_snapshot.values['messages'])}")
    print(f"  当前 Checkpoint ID: {current_snapshot.config['configurable']['checkpoint_id']}")

    print("\n【4. 浏览历史检查点并执行“时间旅行 (Time Travel)”分叉重跑】")
    history = list(app.get_state_history(config_alice))
    print(f"  Alice 线程共保存了 {len(history)} 个历史 Checkpoint 快照。")

    # 找到 Alice 第一轮对话刚结束时的那个 Checkpoint（当时 turn_count == 1）
    first_turn_snapshot = next(
        snap for snap in history if snap.values.get("turn_count") == 1 and snap.next == ()
    )
    fork_config = first_turn_snapshot.config
    print(f"  回溯到第 1 轮结束时的 Checkpoint ID: {fork_config['configurable']['checkpoint_id']}")

    # 从第 1 轮结束的时间点分叉，输入一句全新的第 2 轮话语！
    forked_result = app.invoke(
        {"messages": [HumanMessage(content="【平行宇宙】我改主意了，我现在喜欢 Rust！")]},
        config=fork_config,
    )
    print(f"  分叉执行后的最新回复: {forked_result['messages'][-1].content}")

    assert forked_result["turn_count"] == 2
    assert "Rust" in forked_result["messages"][-1].content
    print("\n✅ 第 06 章执行与验证成功！")
