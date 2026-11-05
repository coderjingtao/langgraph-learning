"""
第 01 章：Hello LangGraph —— 核心概念与第一个状态图 (StateGraph)
====================================================================

【官方文档核心知识点】
LangGraph 是用于构建“有状态、多步骤、支持循环（Cycles）与多参与者”应用的底层编排框架。
它的三大核心基石是：
1. State（状态）：一个共享的数据结构，代表应用程序当前的快照（通常使用 TypedDict、dataclass 或 Pydantic BaseModel 定义）。
2. Nodes（节点）：Python 函数（同步或异步），接收当前 State 作为输入，执行计算逻辑，并返回“需要更新的状态字段字典”。
3. Edges（边）：定义节点之间的执行流向，决定下一个要执行的节点是谁。
   - 普通边 (Normal Edge)：固定从节点 A 走向节点 B。
   - 特殊节点 START 与 END：分别代表图的“入口点”和“终点”。

【本章学习目标】
- 理解如何定义 State
- 理解 Node 函数的签名与返回值规范（返回增量更新字典，而非完整 State）
- 掌握 StateGraph 的定义、添加节点、连接边、编译（compile）与执行（invoke）

运行方式：
    uv run python 01_hello_langgraph.py
"""

from typing import TypedDict
from langgraph.graph import StateGraph, START, END


# ============================================================================
# 1. 定义状态 (State)
# ============================================================================
# State 是整个图中所有节点共享的数据结构。
# 默认情况下，节点返回的同名字段会“直接覆盖 (Override)”State 中的旧值。
class GreetingState(TypedDict):
    user_name: str      # 输入的用户名
    greeting_msg: str   # 处理后的问候语
    step_count: int     # 记录经过了几个处理步骤


# ============================================================================
# 2. 定义节点 (Nodes)
# ============================================================================
# 每个节点都是一个普通的 Python 函数，接收当前 `state: GreetingState` 作为第一个参数。
# 返回值是一个字典（dict），只需包含“想要修改的字段”，不需要返回整个 state。
def normalize_name_node(state: GreetingState) -> dict:
    """节点 1：规范化用户名（去除空白并转为首字母大写）"""
    raw_name = state.get("user_name", "访客")
    clean_name = raw_name.strip().title()
    current_steps = state.get("step_count", 0)
    print(f"  [Node 1: normalize_name_node] 处理用户名: '{raw_name}' -> '{clean_name}'")

    # 返回要更新的状态字段
    return {
        "user_name": clean_name,
        "step_count": current_steps + 1,
    }


def generate_greeting_node(state: GreetingState) -> dict:
    """节点 2：根据规范化后的用户名生成问候语"""
    name = state["user_name"]
    current_steps = state["step_count"]
    msg = f"你好，{name}！欢迎来到 LangGraph 的状态图世界！"
    print(f"  [Node 2: generate_greeting_node] 生成问候语: '{msg}'")

    return {
        "greeting_msg": msg,
        "step_count": current_steps + 1,
    }


# ============================================================================
# 3. 构建并编译状态图 (Build & Compile StateGraph)
# ============================================================================
def build_graph():
    # (1) 初始化 StateGraph，传入我们定义的状态类型 GreetingState
    builder = StateGraph(GreetingState)

    # (2) 注册节点：add_node(节点名称, 节点函数)
    builder.add_node("normalize_name", normalize_name_node)
    builder.add_node("generate_greeting", generate_greeting_node)

    # (3) 添加边（定义执行顺序）：
    # START -> normalize_name -> generate_greeting -> END
    builder.add_edge(START, "normalize_name")
    builder.add_edge("normalize_name", "generate_greeting")
    builder.add_edge("generate_greeting", END)

    # (4) 编译图：compile() 会检查图结构（例如是否有孤立节点），并返回可执行的 CompiledStateGraph
    graph = builder.compile()
    return graph


# ============================================================================
# 4. 独立执行与验证
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("第 01 章：Hello LangGraph 基础状态图演示")
    print("=" * 60)

    app = build_graph()

    # 打印图的 Mermaid 结构，帮助直观理解拓扑结构
    print("\n【1. 图的 Mermaid 结构定义】")
    print(app.get_graph().draw_mermaid())

    # 准备初始状态并调用 invoke()
    initial_state: GreetingState = {
        "user_name": "  alice smith  ",
        "greeting_msg": "",
        "step_count": 0,
    }

    print("【2. 开始执行图 (app.invoke)】")
    final_state = app.invoke(initial_state)

    print("\n【3. 最终状态结果 (Final State)】")
    print(f"  规范化用户名 (user_name)   : {final_state['user_name']}")
    print(f"  最终问候语   (greeting_msg): {final_state['greeting_msg']}")
    print(f"  总执行步骤数 (step_count)  : {final_state['step_count']}")

    # 简单断言验证
    assert final_state["user_name"] == "Alice Smith"
    assert final_state["step_count"] == 2
    assert "Alice Smith" in final_state["greeting_msg"]
    print("\n✅ 第 01 章执行与验证成功！")
