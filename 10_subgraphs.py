"""
第 10 章：子图与多智能体编排 (Subgraphs & Command Routing)
==========================================================

【官方文档核心知识点】
在复杂系统或多智能体（Multi-Agent）架构中，将庞大的单图拆分为模块化的**子图（Subgraphs）**是最佳实践：

1. 子图嵌套的两种方式：
   - **方式 A（共享状态 Schema）**：父图和编译后的子图拥有相同的状态键（或交集键），可直接将编译后的子图作为父图的一个节点：`parent_builder.add_node("sub_agent", compiled_subgraph)`。
   - **方式 B（不同状态 Schema）**：子图拥有完全独立的私有 State，在父图的一个普通节点函数内部手动调用 `subgraph.invoke(transformed_input)`，再将输出转换回父图 State。

2. 使用 `Command` 同时完成“更新状态 + 决定路由”（无需单独写 `add_conditional_edges`）：
   - 节点函数可以直接返回 `Command(update={...}, goto="next_node_name")`！
   - 更强大的是，在子图内部可以返回 `Command(update={...}, goto="parent_node", graph=Command.PARENT)`，直接从子图跨层级跳转回父图的指定节点（非常适合多智能体 Handoff 交接）！

运行方式：
    uv run python 10_subgraphs.py
"""

from typing import Literal, TypedDict
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command


# ============================================================================
# 1. 构建一个拥有独立私有状态的【数据清洗子图 (Data Cleaning Subgraph)】
# ============================================================================
class SubgraphState(TypedDict):
    raw_text: str          # 与父图共享的输入键
    internal_tokens: list  # 子图内部的私有中间键（父图无需关心）
    cleaned_text: str      # 与父图共享的输出键


def tokenize_step(state: SubgraphState) -> dict:
    tokens = [w.strip().lower() for w in state["raw_text"].split() if w.strip()]
    print(f"    [Subgraph Node: tokenize] 拆分词元 -> {tokens}")
    return {"internal_tokens": tokens}


def join_step(state: SubgraphState) -> dict:
    cleaned = " ".join(state["internal_tokens"])
    print(f"    [Subgraph Node: join] 合并清洗结果 -> '{cleaned}'")
    return {"cleaned_text": cleaned}


def build_cleaning_subgraph():
    sub_builder = StateGraph(SubgraphState)
    sub_builder.add_node("tokenize", tokenize_step)
    sub_builder.add_node("join", join_step)
    sub_builder.add_edge(START, "tokenize")
    sub_builder.add_edge("tokenize", "join")
    sub_builder.add_edge("join", END)
    return sub_builder.compile()


# ============================================================================
# 2. 构建父图（使用 Command 实现多智能体分发与子图调用）
# ============================================================================
class ParentState(TypedDict):
    raw_text: str
    cleaned_text: str
    category: str
    final_output: str


def supervisor_triage_node(
    state: ParentState,
) -> Command[Literal["billing_specialist", "tech_specialist"]]:
    """
    使用 `Command` 对象同时完成：
    1. 更新状态 (`update={"category": ...}`)
    2. 动态指定下一个要跳转的节点 (`goto=...`)
    这样就不需要额外调用 `builder.add_conditional_edges` 了！
    """
    text = state["cleaned_text"]
    if "invoice" in text or "refund" in text:
        print("  [Parent Node: supervisor] 识别为财务账单问题 -> 使用 Command 跳转到 billing_specialist")
        return Command(update={"category": "billing"}, goto="billing_specialist")
    else:
        print("  [Parent Node: supervisor] 识别为技术支持问题 -> 使用 Command 跳转到 tech_specialist")
        return Command(update={"category": "tech"}, goto="tech_specialist")


def billing_specialist_node(state: ParentState) -> Command[Literal["__end__"]]:
    msg = f"[财务专员] 已处理您的账单请求：'{state['cleaned_text']}'"
    print(f"  [Parent Node: billing_specialist] {msg}")
    return Command(update={"final_output": msg}, goto=END)


def tech_specialist_node(state: ParentState) -> Command[Literal["__end__"]]:
    msg = f"[技术专家] 已为您排查技术故障：'{state['cleaned_text']}'"
    print(f"  [Parent Node: tech_specialist] {msg}")
    return Command(update={"final_output": msg}, goto=END)


def build_parent_graph():
    cleaning_subgraph = build_cleaning_subgraph()

    parent_builder = StateGraph(ParentState)

    # 直接将编译好的子图注册为父图的一个节点！
    # 父图与子图通过共享的键（raw_text 和 cleaned_text）自动传递数据
    parent_builder.add_node("cleaning_subgraph", cleaning_subgraph)
    parent_builder.add_node("supervisor", supervisor_triage_node)
    parent_builder.add_node("billing_specialist", billing_specialist_node)
    parent_builder.add_node("tech_specialist", tech_specialist_node)

    parent_builder.add_edge(START, "cleaning_subgraph")
    parent_builder.add_edge("cleaning_subgraph", "supervisor")
    # 注意：supervisor、billing_specialist、tech_specialist 都通过返回 Command(goto=...) 自动声明了路由，无需再写 add_edge！

    return parent_builder.compile()


# ============================================================================
# 3. 独立执行与验证
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("第 10 章：子图嵌套与 Command 动态路由 (Subgraphs & Command)")
    print("=" * 60)

    app = build_parent_graph()

    print("\n【1. 包含展开子图 (xray=True) 的 Mermaid 结构】")
    print(app.get_graph(xray=True).draw_mermaid())

    print("【2. 执行父图 -> 自动进入数据清洗子图 -> Supervisor 使用 Command 分发】")
    result = app.invoke({
        "raw_text": "    NEED   REFUND   FOR   INVOICE #1024   ",
        "cleaned_text": "",
        "category": "",
        "final_output": "",
    })

    print("\n【3. 最终父图状态】")
    print(f"  清洗后文本 (cleaned_text): '{result['cleaned_text']}'")
    print(f"  识别分类   (category)    : '{result['category']}'")
    print(f"  最终输出   (final_output): '{result['final_output']}'")

    assert result["cleaned_text"] == "need refund for invoice #1024"
    assert result["category"] == "billing"
    assert "财务专员" in result["final_output"]
    print("\n✅ 第 10 章执行与验证成功！")
