"""
第 05 章：工具调用与 ReAct 智能体架构 (Tools, ToolNode & tools_condition)
=========================================================================

【官方文档核心知识点】
ReAct（Reasoning + Acting，推理与行动）是构建智能体（Agent）最经典的架构模式：
1. LLM 接收用户输入，决定是否需要调用外部工具（Tools）。
   - 如果需要调用工具，LLM 返回带有 `tool_calls` 属性的 `AIMessage`。
   - 如果不需要调用工具，LLM 直接返回最终文本回答。
2. `ToolNode`（预构建工具执行节点）：
   LangGraph 在 `langgraph.prebuilt` 中提供了 `ToolNode(tools)`。
   它会自动读取最后一条 `AIMessage.tool_calls`，并发执行对应的 `@tool` 函数，
   并将结果封装为 `ToolMessage` 追加回 `messages` 状态中。
3. `tools_condition`（预构建条件路由函数）：
   自动检查最后一条消息是否包含 `tool_calls`：
   - 若包含 -> 路由到 `"tools"` 节点执行工具
   - 若不包含 -> 路由到 `END` 结束循环
4. Tool 执行完毕后，通过一条普通边 `add_edge("tools", "agent")` 回到 LLM 节点，让模型根据工具结果生成最终回答。

运行方式：
    uv run python 05_tools_and_react_agent.py
"""

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition


# ============================================================================
# 1. 使用 @tool 装饰器定义工具函数
# ============================================================================
# 注意：函数的 docstring（文档字符串）和类型注解非常重要，真实 LLM 会依靠它们理解如何传参！
@tool
def get_weather(city: str) -> str:
    """查询指定城市的实时天气情况。"""
    weather_db = {
        "北京": "晴朗，气温 22°C，微风",
        "上海": "多云转小雨，气温 19°C",
        "深圳": "雷阵雨，气温 28°C",
    }
    return weather_db.get(city, f"{city}：晴间多云，气温 25°C")


@tool
def calculate_multiply(a: int, b: int) -> int:
    """计算两个整数的乘积。"""
    return a * b


TOOLS = [get_weather, calculate_multiply]


# ============================================================================
# 2. 模拟支持 Tool Calling 的 LLM 节点
# ============================================================================
def agent_llm_node(state: MessagesState) -> dict:
    """
    模拟绑定了工具的 ChatModel (`llm.bind_tools(TOOLS)`) 的推理行为：
    - 当最后一条消息是 HumanMessage 时，解析意图并发起 tool_calls。
    - 当最后一条消息是 ToolMessage 时，汇总工具返回的结果生成最终自然语言回复。
    """
    messages: list[AnyMessage] = state["messages"]
    last_msg = messages[-1]

    # 情况 A：刚刚执行完工具，收到 ToolMessage，进行最终回答汇总
    if isinstance(last_msg, ToolMessage):
        # 收集最近一轮的所有 ToolMessage 结果
        tool_Outputs = [m.content for m in messages if isinstance(m, ToolMessage)]
        final_reply = f"根据工具查询结果，为您汇总如下：{'；'.join(tool_Outputs)}。"
        print(f"  [Node: agent] 收到工具结果，生成最终回答 -> {final_reply}")
        return {"messages": [AIMessage(content=final_reply)]}

    # 情况 B：收到用户提问 HumanMessage，决定调用工具
    user_text = str(last_msg.content)
    tool_calls = []

    if "北京" in user_text or "天气" in user_text:
        tool_calls.append({
            "name": "get_weather",
            "args": {"city": "北京"},
            "id": "call_weather_001",
            "type": "tool_call",
        })
    if "乘" in user_text or "计算" in user_text:
        tool_calls.append({
            "name": "calculate_multiply",
            "args": {"a": 12, "b": 15},
            "id": "call_math_002",
            "type": "tool_call",
        })

    if tool_calls:
        print(f"  [Node: agent] 决定调用 {len(tool_calls)} 个工具: {[tc['name'] for tc in tool_calls]}")
        return {"messages": [AIMessage(content="", tool_calls=tool_calls)]}

    # 情况 C：无需调用工具的普通闲聊
    print("  [Node: agent] 无需调用工具，直接回复")
    return {"messages": [AIMessage(content="你好！我可以帮你查天气或做乘法计算。")]}


# ============================================================================
# 3. 构建经典的 ReAct Agent 状态图
# ============================================================================
def build_react_agent_graph():
    builder = StateGraph(MessagesState)

    # (1) 添加大模型推理节点
    builder.add_node("agent", agent_llm_node)

    # (2) 添加 LangGraph 内置的 ToolNode（自动并行执行 tool_calls 并返回 ToolMessage）
    # 注意：tools_condition 默认跳转的目标节点名称为 "tools"
    builder.add_node("tools", ToolNode(TOOLS))

    # (3) 连接边：
    # START -> agent
    builder.add_edge(START, "agent")

    # agent -> (tools_condition 判断是否有 tool_calls) -> "tools" 或 END
    builder.add_conditional_edges("agent", tools_condition)

    # "tools" 执行完成后，必须回到 "agent" 让模型观察结果并决定下一步
    builder.add_edge("tools", "agent")

    return builder.compile()


# ============================================================================
# 4. 独立执行与验证
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("第 05 章：工具调用与 ReAct 智能体架构 (ToolNode & tools_condition)")
    print("=" * 60)

    app = build_react_agent_graph()

    print("\n【1. ReAct Agent 图的 Mermaid 结构】")
    print(app.get_graph().draw_mermaid())

    print("【2. 执行多工具调用查询】")
    query = "请帮我查一下北京的天气，顺便计算 12 乘以 15 等于多少？"
    print(f"  用户提问: {query}\n")

    final_state = app.invoke({"messages": [HumanMessage(content=query)]})

    print("\n【3. 消息流转全过程 (Message Trace)】")
    for i, msg in enumerate(final_state["messages"], 1):
        msg_type = msg.__class__.__name__
        if isinstance(msg, AIMessage) and msg.tool_calls:
            print(f"  {i}. [{msg_type}] 发起工具调用: {msg.tool_calls}")
        elif isinstance(msg, ToolMessage):
            print(f"  {i}. [{msg_type}] (tool_call_id={msg.tool_call_id}): {msg.content}")
        else:
            print(f"  {i}. [{msg_type}]: {msg.content}")

    assert len(final_state["messages"]) == 5  # Human -> AI(tool_calls) -> Tool1 -> Tool2 -> AI(final)
    assert "180" in final_state["messages"][-1].content
    assert "22°C" in final_state["messages"][-1].content
    print("\n✅ 第 05 章执行与验证成功！")
