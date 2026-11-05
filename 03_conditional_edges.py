"""
第 03 章：条件边与循环控制 (Conditional Edges, Routing & Loops)
================================================================

【官方文档核心知识点】
传统 DAG（有向无环图）只能单向执行，而 Agent 的核心特征是可以“自我反思、重试、循环执行直到满足条件”。
LangGraph 通过 `add_conditional_edges` 支持动态分支与循环（Cycles）：

1. 路由函数（Routing Function）：
   - 接收当前 `state` 作为输入。
   - 返回一个字符串（或字符串列表），代表“下一个要前往的节点名称”或路由键。
   - 注意：路由函数只负责决定“去哪里”，**不会修改 State**。
2. `builder.add_conditional_edges(source_node, routing_function, path_map)`：
   - `source_node`：从哪个节点出来后进行条件判断。
   - `routing_function`：判断逻辑函数。
   - `path_map`（可选但推荐）：将路由函数的返回值映射为目标节点名（有助于正确绘制 Mermaid 图）。
3. 递归限制（`recursion_limit`）：
   - 当图中包含循环时，为防止死循环，LangGraph 默认限制最多执行 25 个超步（Super-steps）。
   - 可通过 `graph.invoke(input, config={"recursion_limit": 50})` 调整。

运行方式：
    uv run python 03_conditional_edges.py
"""

import operator
from typing import Annotated, Literal, TypedDict
from langgraph.graph import StateGraph, START, END


# ============================================================================
# 1. 定义状态
# ============================================================================
class CodeReviewState(TypedDict):
    code: str                                       # 当前代码草稿
    iteration: int                                  # 当前重试/修改次数
    max_iterations: int                             # 最大允许修改次数
    quality_passed: bool                            # 是否通过质量检查
    feedback_history: Annotated[list[str], operator.add]  # 历次审查意见


# ============================================================================
# 2. 定义节点
# ============================================================================
def write_code_node(state: CodeReviewState) -> dict:
    """生成或改进代码的节点"""
    iteration = state.get("iteration", 0) + 1
    print(f"\n  [Node: write_code] 第 {iteration} 次编写代码...")

    # 模拟：第 1 次写的代码缺少类型注解；第 2 次加上了类型注解但缺少文档字符串；第 3 次完美通过
    if iteration == 1:
        draft = "def add(a, b): return a + b"
    elif iteration == 2:
        draft = "def add(a: int, b: int) -> int: return a + b"
    else:
        draft = 'def add(a: int, b: int) -> int:\n    """计算两数之和"""\n    return a + b'

    print(f"  当前代码:\n    {draft.replace(chr(10), chr(10) + '    ')}")
    return {
        "code": draft,
        "iteration": iteration,
    }


def review_code_node(state: CodeReviewState) -> dict:
    """审查代码质量的节点"""
    code = state["code"]
    issues = []

    if ": int" not in code:
        issues.append("缺少类型注解 (Type Hints)")
    if '"""' not in code:
        issues.append("缺少文档字符串 (Docstring)")

    if not issues:
        print("  [Node: review_code] 审查结果：✅ 代码质量达标！")
        return {
            "quality_passed": True,
            "feedback_history": [f"第 {state['iteration']} 轮：审查通过"],
        }
    else:
        reason = "、".join(issues)
        print(f"  [Node: review_code] 审查结果：❌ 未通过 ({reason})")
        return {
            "quality_passed": False,
            "feedback_history": [f"第 {state['iteration']} 轮失败原因：{reason}"],
        }


def publish_code_node(state: CodeReviewState) -> dict:
    """最终发布节点"""
    print("  [Node: publish_code] 🚀 代码已正式合入主分支！")
    return {"feedback_history": ["已完成发布流程"]}


# ============================================================================
# 3. 定义条件路由函数 (Routing Function)
# ============================================================================
# 使用 Literal 类型注解返回值，LangGraph 可以自动推导出有哪些目标分支！
def route_after_review(state: CodeReviewState) -> Literal["publish", "rewrite", "abort"]:
    """
    根据审查结果和重试次数决定下一步走向：
    - 如果通过审查 -> 前往 publish 节点
    - 如果未通过但未超过最大次数 -> 回到 rewrite (write_code) 节点形成循环
    - 如果超过最大次数 -> 直接 abort (结束)
    """
    if state["quality_passed"]:
        return "publish"
    if state["iteration"] >= state["max_iterations"]:
        print("  [Router] ⚠️ 已达到最大重试次数，终止循环！")
        return "abort"
    print("  [Router] 🔄 打回重写 (进入下一轮循环)...")
    return "rewrite"


# ============================================================================
# 4. 构建包含条件分支与循环的状态图
# ============================================================================
def build_review_graph():
    builder = StateGraph(CodeReviewState)

    builder.add_node("write_code", write_code_node)
    builder.add_node("review_code", review_code_node)
    builder.add_node("publish_code", publish_code_node)

    builder.add_edge(START, "write_code")
    builder.add_edge("write_code", "review_code")

    # 添加条件边：从 review_code 出发，根据 route_after_review 的返回值跳转
    builder.add_conditional_edges(
        "review_code",
        route_after_review,
        {
            "publish": "publish_code",  # 审查通过 -> 发布
            "rewrite": "write_code",    # 审查不通过 -> 回到写代码节点（形成 Loop）
            "abort": END,               # 超过最大次数 -> 直接结束
        },
    )

    builder.add_edge("publish_code", END)
    return builder.compile()


# ============================================================================
# 5. 独立执行与验证
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("第 03 章：条件边与循环控制 (Conditional Edges & Loops) 演示")
    print("=" * 60)

    app = build_review_graph()

    print("\n【1. 图的 Mermaid 结构（注意 review_code 回到 write_code 的循环边）】")
    print(app.get_graph().draw_mermaid())

    print("【2. 执行代码自动改进循环 (max_iterations=5)】")
    final_state = app.invoke(
        {
            "code": "",
            "iteration": 0,
            "max_iterations": 5,
            "quality_passed": False,
            "feedback_history": [],
        },
        config={"recursion_limit": 20},
    )

    print("\n【3. 执行完成后的状态汇总】")
    print(f"  总共迭代轮数: {final_state['iteration']}")
    print(f"  是否通过审查: {final_state['quality_passed']}")
    print("  审查历史记录:")
    for item in final_state["feedback_history"]:
        print(f"    - {item}")

    assert final_state["iteration"] == 3
    assert final_state["quality_passed"] is True
    print("\n✅ 第 03 章执行与验证成功！")
