"""
第 13 章：错误处理机制与重试策略 (Error Handling, RetryPolicy & Fault Tolerance)
================================================================================

【官方文档核心知识点】
在生产环境中，Agent 工作流会频繁调用外部 LLM、数据库和第三方 API，随时可能遇到网络抖动、
限流（HTTP 429/5xx）、工具参数错误或节点异常。LangGraph 官方文档提供了多层次的容错与错误处理机制：

1. 节点级自动重试（`RetryPolicy`）：
   - 通过 `builder.add_node(..., retry_policy=RetryPolicy(...))` 或 `@task(retry_policy=...)` 配置。
   - 核心参数：
     * `max_attempts`（默认 3）：最大尝试总次数（**包含首次执行**）。
     * `initial_interval`（默认 0.5 秒）：首次重试前的等待时间。
     * `backoff_factor`（默认 2.0）：指数退避倍率（每次等待时间按倍率增长）。
     * `max_interval`（默认 128.0 秒）：单次重试等待时间上限。
     * `jitter`（默认 True）：是否加入随机抖动，避免并发重试造成下游“惊群效应”。
     * `retry_on`：决定哪些异常触发重试。可传异常类、异常类元组/列表，或自定义判断函数 `Callable[[Exception], bool]`。
       - 默认策略 `default_retry_on` 会重试 `ConnectionError` 及 HTTP 5xx 错误；
       - 不会重试 `ValueError`、`TypeError`、`KeyError`、`RuntimeError`、`OSError` 等程序逻辑错误。
   - 也支持传入 `Sequence[RetryPolicy]`，按顺序匹配第一个命中的策略。

2. 重试耗尽后的节点级兜底（`error_handler` + `NodeError`）：
   - LangGraph 节点容错遵循严格的执行顺序：
     **节点执行 -> 抛出异常 -> `RetryPolicy` 判断并退避重试 -> 重试全部耗尽 -> 触发 `error_handler`**。
   - 在 `add_node(..., error_handler=handler_fn)` 中注册兜底函数，并通过参数类型注解 `error: NodeError`
     自动注入错误上下文（`error.node` 为失败节点名，`error.error` 为原始异常对象）。
   - `error_handler` 可返回降级后的 State 更新字典，或返回 `Command(update=..., goto=...)` 跳转到备用节点。
   - 还可通过 `builder.set_node_defaults(retry_policy=..., error_handler=...)` 为整张图设置全局默认策略：
     * 默认策略在 `compile()` 时生效，**不会**自动被子图（Subgraphs）继承；
     * 单个节点通过 `add_node(..., retry_policy=...)` 显式传入的策略永远优先覆盖全图默认；
     * 若某个非幂等节点需要“关闭”全图默认重试，应显式传入 `retry_policy=RetryPolicy(max_attempts=1)`（传 `None` 会继续继承默认）。

3. 工具调用错误处理（`ToolNode` 的 `handle_tool_errors`）：
   - 预构建的 `ToolNode` 默认开启 `handle_tool_errors=True`（内置默认处理函数）：
     当工具抛出异常时，不会直接让整张图崩溃，而是将错误信息封装成 `ToolMessage(status="error")`
     返回给 LLM，使模型能够“看到报错并自我纠正参数”。
   - 也可自定义 `handle_tool_errors` 为字符串提示、指定异常元组，或自定义格式化函数。

4. 持久化检查点与故障断点续跑（Checkpointer Fault Tolerance）：
   - 配合 Checkpointer（如 `InMemorySaver`），LangGraph 会在每个超步（Super-step）成功后自动保存状态快照。
   - 若后续节点发生未捕获的严重异常导致图中断，修复问题后只需用同一个 `thread_id` 调用
     `graph.invoke(None, config=config)`，图会自动从**最后一个成功的检查点**恢复执行，
     已经成功完成的前置节点**不会重复执行**！

运行方式：
    uv run python 13_error_handling_and_retry.py
"""

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import NodeError
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, RetryPolicy


# ============================================================================
# 场景 1：RetryPolicy 自动重试（瞬态故障自愈 vs 不可重试异常快速失败）
# ============================================================================
class FetchState(TypedDict):
    url: str
    data: str
    logs: Annotated[list[str], operator.add]


def demo_retry_policy():
    print("\n" + "-" * 60)
    print("【场景 1：RetryPolicy 节点自动重试（瞬态网络故障自愈）】")
    print("-" * 60)

    # 模拟一个不稳定的外部 API：前 2 次抛出 ConnectionError，第 3 次成功
    attempt_counter = {"count": 0}

    def fetch_external_api(state: FetchState) -> dict:
        attempt_counter["count"] += 1
        current = attempt_counter["count"]
        print(f"  [Node: fetch_api] 第 {current} 次尝试请求 {state['url']} ...")
        if current < 3:
            raise ConnectionError(f"模拟瞬态网络超时 (第 {current} 次失败)")
        print(f"  [Node: fetch_api] ✅ 第 {current} 次请求成功！")
        return {
            "data": f"来自 {state['url']} 的实时数据",
            "logs": [f"第 {current} 次尝试成功获取数据"],
        }

    # 为该节点配置 RetryPolicy（演示用极短退避时间，实际生产通常 0.5s~30s）
    builder = StateGraph(FetchState)
    builder.add_node(
        "fetch_api",
        fetch_external_api,
        retry_policy=RetryPolicy(
            initial_interval=0.02,
            backoff_factor=2.0,
            max_interval=0.2,
            max_attempts=3,
            jitter=False,
            retry_on=(ConnectionError, TimeoutError),
        ),
    )
    builder.add_edge(START, "fetch_api")
    builder.add_edge("fetch_api", END)
    graph = builder.compile()

    result = graph.invoke({"url": "https://api.example.com/v1/price", "data": "", "logs": []})
    print(f"  最终结果: {result['data']}")
    assert attempt_counter["count"] == 3
    assert "实时数据" in result["data"]

    # 额外验证：非 retry_on 列表中的异常（如 ValueError）会立即失败，不会浪费重试次数
    bad_input_attempts = {"count": 0}

    def strict_parser(state: FetchState) -> dict:
        bad_input_attempts["count"] += 1
        raise ValueError("参数格式非法：无法通过重试修复")

    fail_fast_builder = StateGraph(FetchState)
    fail_fast_builder.add_node(
        "strict_parser",
        strict_parser,
        retry_policy=RetryPolicy(max_attempts=3, initial_interval=0.01),  # 使用 default_retry_on
    )
    fail_fast_builder.add_edge(START, "strict_parser")
    fail_fast_builder.add_edge("strict_parser", END)
    fail_fast_graph = fail_fast_builder.compile()

    try:
        fail_fast_graph.invoke({"url": "bad://url", "data": "", "logs": []})
        raise AssertionError("预期应抛出 ValueError")
    except ValueError as exc:
        print(f"  [快速失败验证] 捕获到不可重试异常: {exc}（实际执行次数: {bad_input_attempts['count']}）")
        assert bad_input_attempts["count"] == 1  # default_retry_on 对 ValueError 不重试


# ============================================================================
# 场景 2：RetryPolicy + error_handler（重试耗尽后的节点级优雅降级）
# ============================================================================
class ServiceState(TypedDict):
    query: str
    answer: str
    source: str
    error_info: str


def demo_error_handler_and_defaults():
    print("\n" + "-" * 60)
    print("【场景 2：RetryPolicy + error_handler（重试耗尽后自动降级）】")
    print("-" * 60)

    primary_attempts = {"count": 0}

    def call_primary_llm(state: ServiceState) -> dict:
        primary_attempts["count"] += 1
        print(f"  [Node: primary_llm] 第 {primary_attempts['count']} 次调用主服务...")
        raise ConnectionError("主服务 503 Service Unavailable")

    def primary_error_handler(state: ServiceState, error: NodeError) -> Command:
        """
        只有当 primary_llm 的 RetryPolicy 重试全部耗尽后，才会执行此 error_handler。
        通过类型注解 `error: NodeError` 自动注入失败节点名与异常实例，
        并用 Command 将流程平滑路由到备用降级节点。
        """
        print(
            f"  [ErrorHandler] 捕获节点 '{error.node}' 最终失败: {error.error} -> 路由至备用节点"
        )
        return Command(
            update={"error_info": f"{error.node} failed: {error.error}"},
            goto="fallback_cache",
        )

    def fallback_cache(state: ServiceState) -> dict:
        print("  [Node: fallback_cache] 启用本地缓存降级响应...")
        return {
            "answer": f"【降级缓存回复】关于“{state['query']}”的基础摘要",
            "source": "local_cache",
        }

    builder = StateGraph(ServiceState)
    # 也可通过 set_node_defaults 为整张图设置统一的默认重试策略（单个节点可覆盖）
    builder.set_node_defaults(
        retry_policy=RetryPolicy(
            max_attempts=2,
            initial_interval=0.02,
            jitter=False,
            retry_on=ConnectionError,
        )
    )
    builder.add_node(
        "primary_llm",
        call_primary_llm,
        error_handler=primary_error_handler,
        destinations=("fallback_cache", END),
    )
    builder.add_node("fallback_cache", fallback_cache)
    builder.add_edge(START, "primary_llm")
    builder.add_edge("primary_llm", END)
    builder.add_edge("fallback_cache", END)

    graph = builder.compile()
    result = graph.invoke({"query": "LangGraph 容错机制", "answer": "", "source": "", "error_info": ""})

    print(f"  最终来源: {result['source']}")
    print(f"  最终回复: {result['answer']}")
    print(f"  记录错误: {result['error_info']}")
    assert primary_attempts["count"] == 2  # 先按全图默认 RetryPolicy 重试满 2 次
    assert result["source"] == "local_cache"
    assert "primary_llm failed" in result["error_info"]

    # 补充验证：单节点显式传入 retry_policy=RetryPolicy(max_attempts=1) 覆盖全图默认策略（关闭重试）
    non_idempotent_attempts = {"count": 0}

    def charge_payment(state: ServiceState) -> dict:
        non_idempotent_attempts["count"] += 1
        print(f"  [Node: charge_payment] 第 {non_idempotent_attempts['count']} 次扣款尝试（已覆盖全图默认，禁止重试）...")
        raise ConnectionError("支付网关超时（非幂等操作不应盲目重试）")

    override_builder = StateGraph(ServiceState)
    override_builder.set_node_defaults(
        retry_policy=RetryPolicy(max_attempts=3, initial_interval=0.01, retry_on=ConnectionError)
    )
    override_builder.add_node(
        "charge_payment",
        charge_payment,
        retry_policy=RetryPolicy(max_attempts=1),  # 覆盖全图默认 max_attempts=3 -> 仅执行 1 次
    )
    override_builder.add_edge(START, "charge_payment")
    override_builder.add_edge("charge_payment", END)
    override_graph = override_builder.compile()

    try:
        override_graph.invoke({"query": "扣款", "answer": "", "source": "", "error_info": ""})
        raise AssertionError("预期应立即抛出 ConnectionError")
    except ConnectionError as exc:
        print(f"  [单节点覆盖默认策略验证] 捕获异常: {exc}（实际执行次数: {non_idempotent_attempts['count']}）")
        assert non_idempotent_attempts["count"] == 1  # 未受全图默认 max_attempts=3 影响


# ============================================================================
# 场景 3：ToolNode 的工具错误处理（handle_tool_errors）
# ============================================================================
@tool
def divide_numbers(a: int, b: int) -> float:
    """计算 a 除以 b 的商。"""
    if b == 0:
        raise ZeroDivisionError("除数 b 不能为 0，请修正参数后重试")
    return a / b


def demo_tool_node_error_handling():
    print("\n" + "-" * 60)
    print("【场景 3：ToolNode 的 handle_tool_errors 工具异常捕获】")
    print("-" * 60)

    def format_tool_error(exc: Exception) -> str:
        return f"工具调用失败（已拦截，未中断图执行）：[{type(exc).__name__}] {exc}"

    tool_node = ToolNode(
        [divide_numbers],
        handle_tool_errors=format_tool_error,
    )

    builder = StateGraph(MessagesState)
    builder.add_node("tools", tool_node)
    builder.add_edge(START, "tools")
    builder.add_edge("tools", END)
    graph = builder.compile()

    # 模拟 LLM 发出一个会触发除零异常的 tool_call
    bad_tool_call_msg = AIMessage(
        content="",
        tool_calls=[
            {"name": "divide_numbers", "args": {"a": 10, "b": 0}, "id": "call_div_001"}
        ],
    )
    out = graph.invoke({"messages": [bad_tool_call_msg]})
    tool_msg = out["messages"][-1]

    print(f"  返回消息类型 : {type(tool_msg).__name__}")
    print(f"  工具消息状态 : {tool_msg.status}")
    print(f"  工具返回内容 : {tool_msg.content}")
    assert isinstance(tool_msg, ToolMessage)
    assert tool_msg.status == "error"
    assert "ZeroDivisionError" in tool_msg.content


# ============================================================================
# 场景 4：Checkpointer 故障恢复与断点续跑（Fault Tolerance）
# ============================================================================
class PipelineState(TypedDict):
    order_id: str
    step1_done: bool
    step2_done: bool
    summary: str


def demo_checkpoint_recovery():
    print("\n" + "-" * 60)
    print("【场景 4：Checkpointer 持久化检查点与故障断点续跑】")
    print("-" * 60)

    execution_counts = {"step1": 0, "step2": 0}
    db_online = {"ready": False}

    def step1_validate_order(state: PipelineState) -> dict:
        execution_counts["step1"] += 1
        print(f"  [Node: step1] 校验订单 {state['order_id']}（耗时操作，累计执行 {execution_counts['step1']} 次）")
        return {"step1_done": True}

    def step2_write_database(state: PipelineState) -> dict:
        execution_counts["step2"] += 1
        if not db_online["ready"]:
            print("  [Node: step2] ❌ 数据库维护中，抛出 RuntimeError 中断执行！")
            raise RuntimeError("数据库暂时不可用")
        print("  [Node: step2] ✅ 数据库已恢复，订单写入成功！")
        return {
            "step2_done": True,
            "summary": f"订单 {state['order_id']} 处理完成",
        }

    builder = StateGraph(PipelineState)
    builder.add_node("step1", step1_validate_order)
    builder.add_node("step2", step2_write_database)
    builder.add_edge(START, "step1")
    builder.add_edge("step1", "step2")
    builder.add_edge("step2", END)

    checkpointer = InMemorySaver()
    graph = builder.compile(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "order-recovery-001"}}

    # 第一次运行：step1 成功并落盘到 Checkpointer，step2 报错中断
    print("  >>> 首次运行（模拟 step2 故障）:")
    try:
        graph.invoke(
            {"order_id": "ORD-2026-001", "step1_done": False, "step2_done": False, "summary": ""},
            config=config,
        )
        raise AssertionError("预期 step2 应抛出 RuntimeError")
    except RuntimeError as exc:
        print(f"  捕获故障异常: {exc}")

    # 查看故障后的持久化状态快照：step1 已完成，下一个待执行节点是 step2
    snapshot = graph.get_state(config)
    print(f"  故障后快照状态: step1_done={snapshot.values['step1_done']}, next={snapshot.next}")
    assert snapshot.values["step1_done"] is True
    assert snapshot.next == ("step2",)

    # 修复外部依赖后，传入 None 从最近检查点直接续跑
    print("  >>> 故障修复后，传入 None 断点续跑:")
    db_online["ready"] = True
    recovered = graph.invoke(None, config=config)

    print(f"  续跑完成结果: {recovered['summary']}")
    print(f"  节点执行统计: step1={execution_counts['step1']} 次（未重复执行）, step2={execution_counts['step2']} 次")
    assert recovered["step2_done"] is True
    assert execution_counts["step1"] == 1  # step1 直接复用检查点，没有重复运行！
    assert execution_counts["step2"] == 2


# ============================================================================
# 主入口：独立执行与验证
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("第 13 章：错误处理机制与重试策略 (RetryPolicy & Error Handling)")
    print("=" * 60)

    demo_retry_policy()
    demo_error_handler_and_defaults()
    demo_tool_node_error_handling()
    demo_checkpoint_recovery()

    print("\n✅ 第 13 章执行与验证成功！")
