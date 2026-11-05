"""
第 09 章：并行执行与 Map-Reduce 模式 (Fan-out / Fan-in & Send API)
====================================================================

【官方文档核心知识点】
LangGraph 支持两种形式的并行执行（同一个 Super-step 内的节点并发运行）：

1. 静态扇出与扇入（Static Fan-out & Fan-in）：
   - 当一个节点（如 `START` 或某前置节点）通过 `add_edge` 同时连接到多个下游节点 B 和 C 时，B 和 C 会在同一个超步中并行执行。
   - **重要注意事项**：并行节点如果同时写入同一个 State 字段，该字段**必须**配置 Reducer（如 `Annotated[list, operator.add]`），否则会报错 `InvalidUpdateError`（因为无法确定谁覆盖谁）。

2. 动态 Map-Reduce 模式（`Send` API）：
   - 很多时候，我们在运行前并不知道需要并行启动多少个分支（例如：上游生成了 N 个子主题，需要对每个子主题并发调用分析节点，最后汇总结果）。
   - `from langgraph.types import Send`：在条件路由函数中返回 `[Send("worker_node", {"item": item}) for item in items]`。
   - LangGraph 会动态克隆并并发执行多个 `worker_node` 实例，每个实例接收自己独立的输入字典，而它们的输出则通过主图 State 的 Reducer 自动汇聚（Reduce）！

运行方式：
    uv run python 09_parallel_and_map_reduce.py
"""

import operator
from typing import Annotated, TypedDict
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send


# ============================================================================
# 1. 定义主图状态与单个 Worker 的独立输入状态
# ============================================================================
class OverallMapReduceState(TypedDict):
    main_topic: str                                      # 主课题
    subtopics: list[str]                                 # 拆解出的子主题列表（数量动态）
    analyses: Annotated[list[str], operator.add]         # 关键：使用 operator.add 汇聚所有并行 Worker 的输出
    final_summary: str                                   # 最终汇总结论


class WorkerInputState(TypedDict):
    """单个并行 Worker 节点接收的局部状态（可以与主图 State 不同）"""
    subtopic: str


# ============================================================================
# 2. 定义 Map-Reduce 各个阶段的节点
# ============================================================================
def generate_subtopics_node(state: OverallMapReduceState) -> dict:
    """【阶段 1：拆分任务】根据主课题动态拆解出多个子主题"""
    topic = state["main_topic"]
    subtopics = [
        f"{topic} - 架构设计",
        f"{topic} - 性能优化",
        f"{topic} - 生产部署",
    ]
    print(f"  [Node: generate_subtopics] 已将《{topic}》拆解为 {len(subtopics)} 个并行子任务: {subtopics}")
    return {"subtopics": subtopics}


def route_to_workers(state: OverallMapReduceState) -> list[Send]:
    """
    【动态路由函数：Map 阶段】
    遍历 state["subtopics"]，为每个子主题创建一个 `Send("analyze_subtopic", {"subtopic": s})` 对象。
    LangGraph 会在同一个超步内并发调用多个 analyze_subtopic 节点！
    """
    return [
        Send("analyze_subtopic", {"subtopic": item})
        for item in state["subtopics"]
    ]


def analyze_subtopic_node(state: WorkerInputState) -> dict:
    """
    【阶段 2：并行 Worker 节点】
    注意：它的输入类型是 `WorkerInputState`（由 Send 传入），
    但它的返回值字段 `"analyses"` 会通过主图的 `operator.add` Reducer 自动合并到 `OverallMapReduceState` 中！
    """
    sub = state["subtopic"]
    result = f"【{sub}】分析结论：核心指标符合预期。"
    print(f"    -> [Parallel Worker: analyze_subtopic] 完成子任务: {sub}")
    return {"analyses": [result]}


def reduce_summary_node(state: OverallMapReduceState) -> dict:
    """【阶段 3：Reduce 汇总节点】等所有并行 Worker 完成后，汇总全部结果"""
    all_analyses = state["analyses"]
    summary = f"已完成 {len(all_analyses)} 个维度的并行分析：\n" + "\n".join(f"  * {a}" for a in all_analyses)
    print("  [Node: reduce_summary] 所有并行分支已汇聚完成！")
    return {"final_summary": summary}


# ============================================================================
# 3. 构建 Map-Reduce 状态图
# ============================================================================
def build_map_reduce_graph():
    builder = StateGraph(OverallMapReduceState)

    builder.add_node("generate_subtopics", generate_subtopics_node)
    builder.add_node("analyze_subtopic", analyze_subtopic_node)
    builder.add_node("reduce_summary", reduce_summary_node)

    builder.add_edge(START, "generate_subtopics")

    # 从 generate_subtopics 通过条件边返回 Send 列表，动态分发到 analyze_subtopic
    builder.add_conditional_edges(
        "generate_subtopics",
        route_to_workers,
        ["analyze_subtopic"],
    )

    # 所有并行 analyze_subtopic 执行完毕后，自动汇聚进入 reduce_summary
    builder.add_edge("analyze_subtopic", "reduce_summary")
    builder.add_edge("reduce_summary", END)

    return builder.compile()


# ============================================================================
# 4. 独立执行与验证
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("第 09 章：并行执行与 Map-Reduce 模式 (Send API) 演示")
    print("=" * 60)

    app = build_map_reduce_graph()

    print("\n【1. 图的 Mermaid 结构】")
    print(app.get_graph().draw_mermaid())

    print("【2. 执行动态 Map-Reduce 并行图】")
    final_state = app.invoke({
        "main_topic": "企业级多智能体系统",
        "subtopics": [],
        "analyses": [],
        "final_summary": "",
    })

    print("\n【3. Reduce 汇总结果】")
    print(final_state["final_summary"])

    assert len(final_state["analyses"]) == 3
    assert "已完成 3 个维度的并行分析" in final_state["final_summary"]
    print("\n✅ 第 09 章执行与验证成功！")
