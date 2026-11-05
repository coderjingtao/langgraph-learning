"""
第 08 章：流式输出全解析 (Streaming Modes & Custom Stream Writer)
==================================================================

【官方文档核心知识点】
在构建大模型应用时，用户体验高度依赖于实时流式反馈（如展示当前走到了哪个节点、逐字输出 Token、推送进度条）。
LangGraph 的 `app.stream(input, stream_mode=...)`（以及异步 `app.astream`）支持多种内置流模式：

1. `stream_mode="values"`：
   在每个超步（Super-step）结束后，输出**完整的当前 State 字典**。
2. `stream_mode="updates"`：
   在每个节点执行完成后，仅输出**该节点刚刚返回的增量更新字典** `{"node_name": {updated_fields}}`。
3. `stream_mode="custom"`（配合 `get_stream_writer()`）：
   允许在节点函数内部通过 `writer = get_stream_writer()` 随时向客户端推送任意自定义数据（如“正在检索第 2/5 个文档...”）。
4. `stream_mode="messages"`：
   当节点内部调用了支持流式的 LangChain ChatModel 时，逐个输出 `(AIMessageChunk, metadata)` 元组。
5. 多模式组合：
   可以传入列表 `stream_mode=["updates", "custom"]`，同时接收多种流事件！

运行方式：
    uv run python 08_streaming.py
"""

import operator
from typing import Annotated, TypedDict
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph


# ============================================================================
# 1. 定义状态与带自定义流输出的节点
# ============================================================================
class ResearchState(TypedDict):
    topic: str
    steps_done: Annotated[list[str], operator.add]
    report: str


def search_sources_node(state: ResearchState) -> dict:
    """节点 1：模拟检索资料，并使用 get_stream_writer() 发送实时进度"""
    # 获取自定义流写入器（在 stream_mode 包含 "custom" 时生效）
    writer = get_stream_writer()

    for i in range(1, 4):
        writer({"event": "progress", "node": "search_sources", "detail": f"正在检索第 {i}/3 个学术数据库..."})

    return {"steps_done": ["完成数据库检索"]}


def synthesize_report_node(state: ResearchState) -> dict:
    """节点 2：生成最终研究报告"""
    writer = get_stream_writer()
    writer({"event": "progress", "node": "synthesize_report", "detail": "正在汇总提纲并生成报告..."})

    return {
        "steps_done": ["完成报告撰写"],
        "report": f"关于《{state['topic']}》的深度研究报告已生成完毕。",
    }


# ============================================================================
# 2. 构建图
# ============================================================================
def build_streaming_graph():
    builder = StateGraph(ResearchState)
    builder.add_node("search_sources", search_sources_node)
    builder.add_node("synthesize_report", synthesize_report_node)

    builder.add_edge(START, "search_sources")
    builder.add_edge("search_sources", "synthesize_report")
    builder.add_edge("synthesize_report", END)

    return builder.compile()


# ============================================================================
# 3. 独立执行与验证不同 stream_mode 的输出差异
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("第 08 章：流式输出全解析 (values / updates / custom)")
    print("=" * 60)

    app = build_streaming_graph()
    initial_input = {"topic": "LangGraph 流式架构", "steps_done": [], "report": ""}

    # ------------------------------------------------------------------------
    # 模式 1：stream_mode="values"（每步输出完整状态快照）
    # ------------------------------------------------------------------------
    print("\n【1. stream_mode='values'（每步输出完整 State）】")
    values_chunks = []
    for chunk in app.stream(initial_input, stream_mode="values"):
        values_chunks.append(chunk)
        print(f"  [values chunk] steps_done={chunk['steps_done']}, report='{chunk['report']}'")
    assert len(values_chunks) == 3  # 初始状态 + 节点1后状态 + 节点2后状态

    # ------------------------------------------------------------------------
    # 模式 2：stream_mode="updates"（每步仅输出节点增量返回值）
    # ------------------------------------------------------------------------
    print("\n【2. stream_mode='updates'（每步输出 {节点名: 节点增量更新}）】")
    updates_chunks = []
    for chunk in app.stream(initial_input, stream_mode="updates"):
        updates_chunks.append(chunk)
        print(f"  [updates chunk] {chunk}")
    assert len(updates_chunks) == 2  # 2 个节点的增量更新

    # ------------------------------------------------------------------------
    # 模式 3：多模式组合 stream_mode=["updates", "custom"]
    # ------------------------------------------------------------------------
    print("\n【3. 组合流模式 stream_mode=['updates', 'custom']（同时接收自定义进度与节点更新）】")
    custom_events_count = 0
    for mode, chunk in app.stream(initial_input, stream_mode=["updates", "custom"]):
        if mode == "custom":
            custom_events_count += 1
        print(f"  [{mode:7s}] -> {chunk}")

    assert custom_events_count == 4  # 3 次检索进度 + 1 次汇总进度
    print("\n✅ 第 08 章执行与验证成功！")
