"""
第 12 章：函数式 API (Functional API: @entrypoint 与 @task)
===========================================================

【官方文档核心知识点】
除了显式声明节点和边的 `StateGraph`（图 API）外，LangGraph 官方文档还提供了 **Functional API（函数式 API）**。
它允许你直接使用标准的 Python 控制流（`if/else`、`for` 循环、函数调用），无需显式定义 `State` 和 `add_edge`，
就能享受到 LangGraph 的持久化检查点（Checkpointer）、人机中断（`interrupt`）、流式输出与并发任务特性！

1. `@entrypoint(checkpointer=...)`：
   - 将一个主函数标记为工作流的入口点。
   - 编译后同样支持 `.invoke()`、`.stream()`、`.get_state()` 等与 `StateGraph` 完全一致的接口。
   - 支持可选的 `previous` 参数，用于访问同一个 `thread_id` 上一次执行返回的持久化结果。
2. `@task`：
   - 将子函数标记为可持久化的异步任务单元（只能在 `@entrypoint` 内部调用）。
   - 调用 `@task` 函数会立即返回一个 `Future` 对象：
     * 调用 `future.result()` 会等待该任务完成并获取结果。
     * 如果同时触发多个 `@task` 再分别调用 `.result()`，它们会自动并发执行！
   - 当工作流从 `interrupt()` 恢复时，已经完成过的 `@task` **不会重复执行**，而是直接从 Checkpointer 缓存中读取结果（保证幂等性）！

运行方式：
    uv run python 12_functional_api.py
"""

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.func import entrypoint, task
from langgraph.types import Command, interrupt


# ============================================================================
# 1. 使用 @task 定义可缓存、可并发的子任务
# ============================================================================
@task
def generate_outline(topic: str) -> list[str]:
    """任务 1：生成文章大纲（当从后续 interrupt 恢复时，已完成的 @task 不会重复运行）"""
    print(f"  [@task: generate_outline] 正在为《{topic}》生成章节大纲...")
    return [
        f"第一章：{topic} 的核心原理",
        f"第二章：{topic} 的工程实践",
        f"第三章：{topic} 的未来展望",
    ]


@task
def draft_section(section_title: str) -> str:
    """任务 2：撰写单个章节内容（支持并发调用）"""
    print(f"    [@task: draft_section] 并发撰写章节 -> {section_title}")
    return f"【{section_title}】这里是详实的正文内容..."


# ============================================================================
# 2. 使用 @entrypoint 定义主工作流函数
# ============================================================================
memory_saver = InMemorySaver()


@entrypoint(checkpointer=memory_saver)
def article_workflow(topic: str) -> dict:
    """
    使用原生 Python 语法编写包含“大纲生成 -> 并发写稿 -> 人工审阅中断 -> 最终发布”的工作流。
    """
    print(f"\n[Entrypoint] 启动文章生成工作流，主题: {topic}")

    # 步骤 1：调用 @task 获取大纲（通过 .result() 同步等待结果）
    outline = generate_outline(topic).result()

    # 步骤 2：并发启动多个 @task 撰写各个章节，然后收集结果
    section_futures = [draft_section(sec) for sec in outline]
    drafted_sections = [fut.result() for fut in section_futures]
    full_draft = "\n".join(drafted_sections)

    # 步骤 3：触发 interrupt() 等待人类编辑审阅
    print("  [Entrypoint] ⏸️ 初稿已完成，触发 interrupt() 等待人类编辑确认...")
    editor_feedback = interrupt({
        "message": "请审阅文章初稿",
        "outline": outline,
        "draft_preview": full_draft,
    })

    # 步骤 4：收到 Command(resume=...) 后继续执行（注意：前面的 @task 不会重跑，直接读缓存！）
    print(f"  [Entrypoint] ▶️ 收到编辑反馈: {editor_feedback}")
    return {
        "topic": topic,
        "sections_count": len(drafted_sections),
        "editor_comment": editor_feedback,
        "status": "published",
    }


# ============================================================================
# 3. 独立执行与验证
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("第 12 章：函数式 API (@entrypoint 与 @task) 演示")
    print("=" * 60)

    config = {"configurable": {"thread_id": "func-workflow-001"}}

    print("\n【阶段 1：首次调用 article_workflow.invoke()，执行到 interrupt 处暂停】")
    step1_out = article_workflow.invoke("LangGraph 函数式 API", config=config)
    print(f"  捕获到的中断信息: {step1_out.get('__interrupt__')}")

    print("\n【阶段 2：使用 Command(resume=...) 恢复执行】")
    print("  （注意观察控制台输出：@task 修饰的 generate_outline 和 draft_section 不会重复打印，因为直接复用了 Checkpoint 缓存！）")
    final_output = article_workflow.invoke(
        Command(resume="结构清晰，同意发布！"),
        config=config,
    )

    print("\n【阶段 3：最终工作流返回结果】")
    print(f"  文章主题 (topic)         : {final_output['topic']}")
    print(f"  章节数量 (sections_count): {final_output['sections_count']}")
    print(f"  编辑评语 (editor_comment): {final_output['editor_comment']}")
    print(f"  发布状态 (status)        : {final_output['status']}")

    assert final_output["status"] == "published"
    assert final_output["sections_count"] == 3
    print("\n✅ 第 12 章执行与验证成功！")
