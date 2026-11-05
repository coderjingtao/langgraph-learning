"""
第 02 章：状态管理与归约器 (State Schema, Reducers & Input/Output Schema)
=========================================================================

【官方文档核心知识点】
1. 默认更新行为（Override）：
   在 State 中，如果不做特殊声明，节点返回的新值会“直接覆盖”旧值。
2. 归约器（Reducers）：
   当多个节点需要“追加（Append）”或“累加”同一个字段（例如日志列表、消息历史、计数器）时，
   我们使用 `typing.Annotated[类型, reducer函数]` 来指定该字段如何合并旧值与新值。
   - 常见内置 Reducer：`operator.add`（对列表进行拼接 `old + new`，对数字进行相加）。
   - 自定义 Reducer：编写 `def my_reducer(left, right) -> result` 函数，处理去重、重置或安全合并。
3. 输入与输出 Schema 分离（Input / Output Schema）：
   有时图内部需要很多中间状态字段，但对外暴露的输入（Input）和输出（Output）只需要精简字段。
   `StateGraph(OverallState, input_schema=InputState, output_schema=OutputState)` 可以实现接口过滤。

运行方式：
    uv run python 02_state_and_reducers.py
"""

import operator
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END


# ============================================================================
# 1. 自定义 Reducer 函数
# ============================================================================
def unique_tags_reducer(existing_tags: list[str] | None, new_tags: list[str] | None) -> list[str]:
    """
    自定义归约器：将新标签合并到现有标签列表中，并保持顺序去重。
    如果传入特殊指令 ["__RESET__"]，则清空列表。
    """
    if existing_tags is None:
        existing_tags = []
    if not new_tags:
        return existing_tags
    if new_tags == ["__RESET__"]:
        return []

    combined = list(existing_tags)
    for tag in new_tags:
        if tag not in combined:
            combined.append(tag)
    return combined


# ============================================================================
# 2. 定义带有 Reducers 的状态 (State)
# ============================================================================
class DocumentState(TypedDict):
    # 无 Reducer：默认后写覆盖前写 (Override)
    title: str

    # 使用 operator.add 作为列表 Reducer：每次返回 list 时自动执行 existing_logs + new_logs
    logs: Annotated[list[str], operator.add]

    # 使用 operator.add 作为数值 Reducer：每次返回 int 时自动执行 existing_score + delta_score
    total_score: Annotated[int, operator.add]

    # 使用自定义 Reducer：自动去重合并标签
    tags: Annotated[list[str], unique_tags_reducer]


# ============================================================================
# 3. 定义节点
# ============================================================================
def analyze_content_node(state: DocumentState) -> dict:
    print("  [Node 1: analyze_content_node] 分析文档内容...")
    return {
        # title 会覆盖旧值
        "title": state["title"].strip(),
        # logs 会通过 operator.add 追加到现有列表中
        "logs": ["完成内容基础分析"],
        # total_score 会通过 operator.add 与旧值相加 (+10)
        "total_score": 10,
        # tags 会通过 unique_tags_reducer 去重合并
        "tags": ["AI", "LangGraph", "Python"],
    }


def quality_check_node(state: DocumentState) -> dict:
    print("  [Node 2: quality_check_node] 执行质量评分与标签补充...")
    return {
        # 不需要返回 title，它会保留上一步的值
        "logs": ["完成质量检查", "评分已累加"],
        "total_score": 25,  # 自动与之前的 10 相加 -> 35
        "tags": ["Python", "Tutorial", "Agent"],  # "Python" 重复，会被自定义 Reducer 自动去重
    }


# ============================================================================
# 4. 进阶：Input / Output Schema 过滤演示
# ============================================================================
class InputSchema(TypedDict):
    """调用者只需传入 question"""
    question: str


class OutputSchema(TypedDict):
    """调用者最终只收到 answer，隐藏中间的 internal_draft"""
    answer: str


class OverallInternalState(TypedDict):
    """图内部完整状态（包含输入、中间变量、输出）"""
    question: str
    internal_draft: str
    answer: str


def draft_node(state: OverallInternalState) -> dict:
    return {"internal_draft": f"【内部草稿】针对“{state['question']}”的推理过程..."}


def finalize_node(state: OverallInternalState) -> dict:
    return {"answer": f"基于{state['internal_draft']}，最终答案是：使用 Annotated 定义 Reducer。"}


# ============================================================================
# 5. 独立执行与验证
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("第 02 章：状态管理与归约器 (Reducers) 演示")
    print("=" * 60)

    # --- 演示 A：Reducers (operator.add 与自定义 Reducer) ---
    builder = StateGraph(DocumentState)
    builder.add_node("analyze", analyze_content_node)
    builder.add_node("quality_check", quality_check_node)
    builder.add_edge(START, "analyze")
    builder.add_edge("analyze", "quality_check")
    builder.add_edge("quality_check", END)
    graph = builder.compile()

    print("\n【演示 A：执行带有 Reducers 的状态图】")
    result = graph.invoke({
        "title": "  LangGraph 核心指南  ",
        "logs": ["初始化文档状态"],
        "total_score": 5,
        "tags": ["AI"],
    })

    print(f"  文档标题 (Override)           : {result['title']}")
    print(f"  日志列表 (operator.add list)  : {result['logs']}")
    print(f"  累计得分 (operator.add int)   : {result['total_score']} (初始5 + 节点1的10 + 节点2的25)")
    print(f"  去重标签 (自定义 Reducer)      : {result['tags']}")

    assert result["total_score"] == 40
    assert len(result["logs"]) == 4
    assert result["tags"] == ["AI", "LangGraph", "Python", "Tutorial", "Agent"]

    # --- 演示 B：Input / Output Schema 过滤 ---
    print("\n【演示 B：Input / Output Schema 接口过滤】")
    io_builder = StateGraph(
        OverallInternalState,
        input_schema=InputSchema,
        output_schema=OutputSchema,
    )
    io_builder.add_node("draft", draft_node)
    io_builder.add_node("finalize", finalize_node)
    io_builder.add_edge(START, "draft")
    io_builder.add_edge("draft", "finalize")
    io_builder.add_edge("finalize", END)
    io_graph = io_builder.compile()

    io_result = io_graph.invoke({"question": "如何累加列表状态？"})
    print(f"  图返回结果字典的 Keys: {list(io_result.keys())} (注意 internal_draft 和 question 已被过滤)")
    print(f"  最终输出 answer: {io_result['answer']}")

    assert list(io_result.keys()) == ["answer"]
    print("\n✅ 第 02 章执行与验证成功！")
