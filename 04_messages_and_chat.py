"""
第 04 章：消息状态与对话机器人 (MessagesState, add_messages & Chatbots)
=======================================================================

【官方文档核心知识点】
由于绝大多数大语言模型（LLM）应用都以“对话消息列表（Messages）”作为核心数据结构，
LangGraph 专门提供了针对消息列表的内置工具：

1. `add_messages` 归约器（Reducer）：
   比普通的 `operator.add` 更智能！
   - 自动将新消息追加到历史消息列表末尾。
   - **支持按 ID 更新/覆盖消息**：如果新消息的 `id` 与历史中某条消息的 `id` 相同，它会原地替换旧消息，而不是重复追加。
   - **支持删除消息**：配合 `RemoveMessage(id=...)` 可以从状态中精确删除指定消息（常用于上下文窗口裁剪）。
   - 自动将字典格式 `{"role": "user", "content": "..."}` 反序列化为 LangChain 标准的 `HumanMessage` / `AIMessage`。

2. `MessagesState` 预定义状态：
   因为定义 `messages: Annotated[list[AnyMessage], add_messages]` 太常用了，
   LangGraph 直接提供了 `MessagesState` 基类，你可以直接使用或继承它添加额外字段。

运行方式：
    uv run python 04_messages_and_chat.py
"""

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.message import add_messages


# ============================================================================
# 1. 演示 add_messages 的智能合并、ID 替换与删除能力
# ============================================================================
def demo_add_messages_reducer():
    print("\n【1. 测试 add_messages 归约器的核心特性】")

    # (a) 正常追加消息
    msg1 = [HumanMessage(content="你好，我想学习 LangGraph", id="msg-1")]
    msg2 = [AIMessage(content="太棒了！正在为你生成大纲...", id="msg-2")]
    history = add_messages(msg1, msg2)
    print(f"  a) 追加后消息数量: {len(history)} -> {[m.content for m in history]}")

    # (b) 相同 ID 的消息会被原地更新（而非重复追加）
    msg2_updated = [AIMessage(content="太棒了！这是为你生成的完整 LangGraph 学习路线。", id="msg-2")]
    history = add_messages(history, msg2_updated)
    print(f"  b) 同 ID 更新后数量: {len(history)} -> {[m.content for m in history]}")

    # (c) 使用 RemoveMessage 删除指定 ID 的消息
    delete_instruction = [RemoveMessage(id="msg-1")]
    history = add_messages(history, delete_instruction)
    print(f"  c) 删除 msg-1 后数量: {len(history)} -> {[m.content for m in history]}")
    assert len(history) == 1 and history[0].id == "msg-2"


# ============================================================================
# 2. 继承 MessagesState 构建对话机器人状态
# ============================================================================
# MessagesState 已经内置了 `messages: Annotated[list[AnyMessage], add_messages]`
class ChatbotState(MessagesState):
    user_tier: str  # 我们可以扩展自定义字段，例如用户等级


# ============================================================================
# 3. 定义对话机器人节点（无需 API Key 的本地模拟 LLM + 真实 LLM 替换说明）
# ============================================================================
def mock_llm_respond(messages: list[AnyMessage], user_tier: str) -> AIMessage:
    """
    本地确定性模拟 LLM，确保本脚本无需配置 OPENAI_API_KEY 即可独立运行。
    【实战提示】在真实项目中，只需将此处替换为：
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model="gpt-4o-mini")
        return llm.invoke(messages)
    """
    last_user_msg = messages[-1].content
    return AIMessage(
        content=f"[{user_tier}专属助手] 我收到了你的问题：“{last_user_msg}”。当前上下文共有 {len(messages)} 条消息。"
    )


def trim_old_messages_node(state: ChatbotState) -> dict:
    """
    演示：如果历史消息超过 4 条，自动使用 RemoveMessage 清理最早的非 System 消息，防止 Token 超限
    """
    messages = state["messages"]
    if len(messages) <= 4:
        return {}

    # 保留 SystemMessage 和最近的 3 条消息，删除其余旧消息
    to_remove = []
    for msg in messages[:-3]:
        if not isinstance(msg, SystemMessage) and msg.id:
            to_remove.append(RemoveMessage(id=msg.id))

    if to_remove:
        print(f"  [Node: trim_old_messages] 触发上下文裁剪，清理了 {len(to_remove)} 条早期消息")
    return {"messages": to_remove}


def chatbot_node(state: ChatbotState) -> dict:
    """调用模型生成回复"""
    messages = state["messages"]
    tier = state.get("user_tier", "普通用户")
    ai_response = mock_llm_respond(messages, tier)
    print(f"  [Node: chatbot] AI 回复 -> {ai_response.content}")
    # 返回列表，add_messages 会自动将其追加到 state["messages"] 中
    return {"messages": [ai_response]}


# ============================================================================
# 4. 构建并运行图
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("第 04 章：消息状态与对话机器人 (MessagesState & add_messages)")
    print("=" * 60)

    # 先验证 add_messages 的底层行为
    demo_add_messages_reducer()

    # 构建对话图：START -> trim_old_messages -> chatbot -> END
    builder = StateGraph(ChatbotState)
    builder.add_node("trim_old_messages", trim_old_messages_node)
    builder.add_node("chatbot", chatbot_node)
    builder.add_edge(START, "trim_old_messages")
    builder.add_edge("trim_old_messages", "chatbot")
    builder.add_edge("chatbot", END)
    app = builder.compile()

    print("\n【2. 执行多轮消息追加与自动裁剪演示】")
    # 注意：由于 add_messages 支持字典格式，我们也可以直接传入 {"role": "user", "content": "..."}
    initial_messages = [
        SystemMessage(content="你是一位资深 Python 架构师。", id="sys-1"),
        HumanMessage(content="第一轮问题：什么是 StateGraph？", id="u-1"),
        AIMessage(content="第一轮回答：StateGraph 是状态机图。", id="a-1"),
        HumanMessage(content="第二轮问题：什么是 Reducer？", id="u-2"),
        AIMessage(content="第二轮回答：Reducer 决定状态如何合并。", id="a-2"),
        HumanMessage(content="第三轮问题：请总结一下 MessagesState 的作用。", id="u-3"),
    ]

    final_state = app.invoke({
        "messages": initial_messages,
        "user_tier": "VIP会员",
    })

    print("\n【3. 最终保留在 State 中的消息列表】")
    for idx, msg in enumerate(final_state["messages"], 1):
        print(f"  {idx}. [{msg.__class__.__name__}] (id={msg.id}): {msg.content}")

    # 原本有 6 条消息，裁剪掉了 u-1 和 a-1（保留 sys-1 及最后 3 条），再加上新生成的 AIMessage，共 5 条
    assert len(final_state["messages"]) == 5
    print("\n✅ 第 04 章执行与验证成功！")
