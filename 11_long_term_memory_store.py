"""
第 11 章：跨会话长期记忆存储 (Long-Term Memory & InMemoryStore)
===============================================================

【官方文档核心知识点】
在 LangGraph 中，“记忆（Memory）”分为两个维度：
1. **短期记忆（Short-term Memory）**：由 `Checkpointer`（如 `InMemorySaver`）管理，绑定在单个 `thread_id` 上，用于保存当前对话线程的上下文。
2. **长期记忆（Long-term Memory）**：由 `BaseStore`（如 `InMemoryStore` 或 `PostgresStore`）管理，用于**跨会话（Cross-thread）**存储用户画像、偏好设置、历史经验知识库。

【Store 的核心用法】
- 命名空间（Namespace）：采用元组形式组织层级目录，例如 `("users", user_id, "preferences")`。
- `store.put(namespace, key, value_dict)`：写入或更新一条记忆。
- `store.get(namespace, key)`：精确读取某条记忆。
- `store.search(namespace)`：检索某个命名空间下的所有记忆条目。
- **在图节点中使用 Store**：
  编译图时传入 `builder.compile(checkpointer=..., store=in_memory_store)`，
  在节点函数签名中添加 `*, store: BaseStore` 参数，LangGraph 会通过依赖注入自动把 `store` 实例传给节点！

运行方式：
    uv run python 11_long_term_memory_store.py
"""

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore


# ============================================================================
# 1. 定义注入了 BaseStore 和 RunnableConfig 的节点函数
# ============================================================================
def personalized_agent_node(
    state: MessagesState,
    config: RunnableConfig,
    *,
    store: BaseStore,  # LangGraph 自动注入编译时绑定的 Store 对象
) -> dict:
    """
    该节点演示如何：
    1. 从 config["configurable"]["user_id"] 获取当前用户标识
    2. 从长期记忆 Store 中读取该用户的偏好设置
    3. 如果用户在对话中表达了新偏好，将其写入 Store（跨 thread_id 永久生效）
    """
    user_id = config["configurable"].get("user_id", "anonymous")
    namespace = ("user_profiles", user_id)

    last_user_msg = str(state["messages"][-1].content)

    # 检测用户是否在告知新的个人偏好（例如：“记住：我喜欢简洁的代码风格”）
    if "记住：" in last_user_msg:
        fact = last_user_msg.split("记住：", 1)[1].strip()
        key = f"pref_{len(store.search(namespace)) + 1}"
        store.put(namespace, key, {"fact": fact})
        print(f"  [Node: agent] 💾 已将新偏好写入长期记忆 Store -> namespace={namespace}, key={key}, fact='{fact}'")

    # 从长期记忆 Store 中检索该用户的所有历史偏好
    memories = store.search(namespace)
    known_facts = [item.value["fact"] for item in memories]

    reply = (
        f"你好 (user_id={user_id})！"
        f"我从跨会话长期记忆库中查到关于你的 {len(known_facts)} 条偏好：{known_facts}。"
    )
    print(f"  [Node: agent] 🤖 回复 -> {reply}")
    return {"messages": [AIMessage(content=reply)]}


# ============================================================================
# 2. 构建同时具备短期记忆 (Checkpointer) 与长期记忆 (Store) 的图
# ============================================================================
def build_memory_agent():
    # 短期记忆：按 thread_id 隔离会话消息
    checkpointer = InMemorySaver()
    # 长期记忆：按 namespace 跨会话共享用户知识
    long_term_store = InMemoryStore()

    builder = StateGraph(MessagesState)
    builder.add_node("agent", personalized_agent_node)
    builder.add_edge(START, "agent")
    builder.add_edge("agent", END)

    # 同时挂载 checkpointer 和 store
    app = builder.compile(checkpointer=checkpointer, store=long_term_store)
    return app, long_term_store


# ============================================================================
# 3. 独立执行与验证（测试跨会话 thread_id 共享长期记忆）
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("第 11 章：跨会话长期记忆存储 (BaseStore & InMemoryStore) 演示")
    print("=" * 60)

    app, store = build_memory_agent()

    # ------------------------------------------------------------------------
    # 会话 1 (thread-morning)：用户 u-888 告诉 Agent 自己的两个偏好
    # ------------------------------------------------------------------------
    print("\n【1. 周一上午的会话 (thread_id='thread-morning', user_id='u-888')】")
    config_session_1 = {"configurable": {"thread_id": "thread-morning", "user_id": "u-888"}}

    app.invoke(
        {"messages": [HumanMessage(content="记住：我使用 Python 3.13 和 uv 管理环境")]},
        config=config_session_1,
    )
    app.invoke(
        {"messages": [HumanMessage(content="记住：我喜欢带有中文注释的示例代码")]},
        config=config_session_1,
    )

    # ------------------------------------------------------------------------
    # 会话 2 (thread-evening)：开启全新的 thread_id，验证能否读取到长期记忆！
    # ------------------------------------------------------------------------
    print("\n【2. 周五晚上的全新会话 (thread_id='thread-evening', user_id='u-888')】")
    config_session_2 = {"configurable": {"thread_id": "thread-evening", "user_id": "u-888"}}

    res_session_2 = app.invoke(
        {"messages": [HumanMessage(content="嗨，你还记得我的开发环境偏好吗？")]},
        config=config_session_2,
    )

    # 验证：虽然在全新会话 thread-evening 中只有 2 条消息（1 问 1 答），但长期记忆成功保留了 2 条偏好
    assert len(res_session_2["messages"]) == 2
    assert "Python 3.13" in res_session_2["messages"][-1].content
    assert "中文注释" in res_session_2["messages"][-1].content
    print("\n✅ 第 11 章执行与验证成功！")
