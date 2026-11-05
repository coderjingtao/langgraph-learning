"""
第 07 章：人机协同与动态中断控制 (Human-in-the-Loop, interrupt & Command)
==========================================================================

【官方文档核心知识点】
在执行敏感操作（如转账、删除数据库、发送邮件）前，Agent 往往需要暂停并等待人类审批或补充信息。
LangGraph 基于 Checkpointer 提供了强大的 `interrupt()` 原语与 `Command` 恢复机制：

1. `interrupt(value)` 动态中断：
   - 在任意节点函数内部调用 `from langgraph.types import interrupt`。
   - 当执行到 `answer = interrupt(payload)` 时，图会**立即暂停**，并将 `payload` 暴露给调用者（在返回结果的 `__interrupt__` 字段或 `get_state().tasks` 中）。
   - 注意：**必须配置 Checkpointer**，因为暂停时需要保存当前状态现场。
2. `Command(resume=value)` 恢复执行：
   - 当人类审批完成或输入数据后，调用 `app.invoke(Command(resume=人类输入的值), config=thread_config)`。
   - 节点会从头重新执行到 `interrupt()` 处，此时 `interrupt()` 不再暂停，而是**直接返回 `resume` 传入的值**，继续往下执行！
3. 注意事项：
   - 因为恢复时包含 `interrupt` 的节点会从该节点开头重新运行，不要在 `interrupt()` 之前放置不可重入的副作用操作（或者将其拆分为独立的前置节点）。

运行方式：
    uv run python 07_human_in_the_loop.py
"""

from typing import TypedDict
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


# ============================================================================
# 1. 定义转账审批流程状态
# ============================================================================
class TransferState(TypedDict):
    recipient: str          # 收款人
    amount: float           # 转账金额
    status: str             # 状态：pending / approved / rejected / completed
    reviewer_note: str      # 审批人备注


# ============================================================================
# 2. 定义节点（包含 interrupt 动态人工审批）
# ============================================================================
def prepare_transfer_node(state: TransferState) -> dict:
    print(f"  [Node: prepare_transfer] 准备向 '{state['recipient']}' 转账 ¥{state['amount']:.2f}")
    return {"status": "pending"}


def human_approval_node(state: TransferState) -> dict:
    """
    如果金额超过 1000 元，触发 interrupt() 暂停图执行，等待人工审批结果。
    """
    if state["amount"] <= 1000:
        print("  [Node: human_approval] 金额 <= 1000，系统自动审批通过！")
        return {"status": "approved", "reviewer_note": "小额自动放行"}

    print("  [Node: human_approval] ⚠️ 检测到高额转账，触发 interrupt() 等待人工审批...")

    # 调用 interrupt() 会暂停图，并将字典信息传递给外部调用方。
    # 当外部使用 Command(resume=...) 恢复时，human_decision 将收到 resume 的值！
    human_decision: dict = interrupt({
        "question": "是否批准该笔高额转账？",
        "recipient": state["recipient"],
        "amount": state["amount"],
    })

    approved = human_decision.get("approved", False)
    note = human_decision.get("note", "")
    new_amount = human_decision.get("modified_amount", state["amount"])

    if approved:
        print(f"  [Node: human_approval] ✅ 人工审批通过！(备注: {note}, 确认金额: ¥{new_amount:.2f})")
        return {"status": "approved", "reviewer_note": note, "amount": new_amount}
    else:
        print(f"  [Node: human_approval] ❌ 人工审批拒绝！(原因: {note})")
        return {"status": "rejected", "reviewer_note": note}


def execute_transfer_node(state: TransferState) -> dict:
    """根据审批结果决定是否真正执行银行接口"""
    if state["status"] == "approved":
        print(f"  [Node: execute_transfer] 💰 已成功向 '{state['recipient']}' 汇款 ¥{state['amount']:.2f}！")
        return {"status": "completed"}
    else:
        print(f"  [Node: execute_transfer] 🛑 转账已取消，未发生资金变动。")
        return {}


# ============================================================================
# 3. 构建并编译图（必须传入 checkpointer）
# ============================================================================
def build_hitl_graph():
    builder = StateGraph(TransferState)
    builder.add_node("prepare", prepare_transfer_node)
    builder.add_node("human_approval", human_approval_node)
    builder.add_node("execute", execute_transfer_node)

    builder.add_edge(START, "prepare")
    builder.add_edge("prepare", "human_approval")
    builder.add_edge("human_approval", "execute")
    builder.add_edge("execute", END)

    return builder.compile(checkpointer=InMemorySaver())


# ============================================================================
# 4. 独立执行与验证
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("第 07 章：人机协同与动态中断控制 (interrupt & Command)")
    print("=" * 60)

    app = build_hitl_graph()
    thread_config = {"configurable": {"thread_id": "transfer-tx-999"}}

    print("\n【步骤 1：发起一笔 50000 元的高额转账请求】")
    first_run_result = app.invoke(
        {
            "recipient": "供应商A",
            "amount": 50000.0,
            "status": "init",
            "reviewer_note": "",
        },
        config=thread_config,
    )

    # 检查图是否处于中断等待状态
    snapshot = app.get_state(thread_config)
    print(f"\n【步骤 2：查看暂停时的状态快照】")
    print(f"  下一个待执行节点 (snapshot.next): {snapshot.next}")
    print(f"  捕获到的中断请求 (__interrupt__): {first_run_result.get('__interrupt__')}")

    assert snapshot.next == ("human_approval",)
    assert "__interrupt__" in first_run_result

    print("\n【步骤 3：模拟财务主管审批通过，并将金额微调为 48000 元后恢复执行 (Command(resume=...))】")
    final_result = app.invoke(
        Command(
            resume={
                "approved": True,
                "note": "合同核对无误，扣除2000元质保金后放款",
                "modified_amount": 48000.0,
            }
        ),
        config=thread_config,
    )

    print("\n【步骤 4：恢复执行后的最终结果】")
    print(f"  最终状态 (status)       : {final_result['status']}")
    print(f"  实际执行金额 (amount)   : ¥{final_result['amount']:.2f}")
    print(f"  审批备注 (reviewer_note): {final_result['reviewer_note']}")

    assert final_result["status"] == "completed"
    assert final_result["amount"] == 48000.0
    print("\n✅ 第 07 章执行与验证成功！")
