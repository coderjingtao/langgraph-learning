"""
第 14 章：节点缓存机制 (Node Caching: CachePolicy & InMemoryCache)
==================================================================

【官方文档核心知识点】
在 LangGraph 中，许多节点执行成本很高（如调用 LLM、向量检索、复杂数据库查询或外部 API）。
官方文档提供了原生的 **节点级缓存（Node Caching）** 机制，允许根据节点输入缓存输出结果，
在后续收到相同输入时直接复用缓存，大幅降低延迟与成本：

1. 启用节点缓存的两个必要条件：
   - **图编译时提供缓存后端**：`builder.compile(cache=InMemoryCache())`
     （本地开发/测试常用 `InMemoryCache`，生产持久化可使用 `SqliteCache` 或自定义 `BaseCache`）。
   - **节点上挂载 `CachePolicy`**：`builder.add_node("node_name", fn, cache_policy=CachePolicy(...))`
     （也支持通过 `builder.set_node_defaults(cache_policy=...)` 为整张图设置默认策略，
     或在 Functional API 中使用 `@entrypoint(cache=...)` + `@task(cache_policy=...)`）。

2. `CachePolicy` 核心参数：
   - `ttl: int | None = None`：缓存存活时间（单位：秒）。超过 `ttl` 后缓存自动过期失效；为 `None` 表示永不过期。
   - `key_func: Callable[..., str | bytes] = default_cache_key`：缓存键生成函数。
     * 默认使用 `pickle` 对整个节点输入做序列化哈希。
     * 实际工程中，State 常包含 `request_id`、`timestamp` 等每次都变化的字段，此时必须自定义 `key_func`，
       只提取真正影响结果的业务字段作为缓存键，才能有效命中缓存！

3. 识别缓存命中（`__metadata__.cached`）：
   - 使用 `graph.stream(..., stream_mode="updates")` 时，若某节点命中了缓存，
     返回的事件字典中会带有 `'__metadata__': {'cached': True}` 标记。

4. 与 Checkpointer 的区别：
   - **Checkpointer（第 06 / 12 章）**：按 `thread_id` 保存会话执行进度，用于多轮对话记忆、人工中断恢复和故障断点续跑。
   - **Node Cache（本章）**：按节点输入（`key_func`）跨会话/跨调用共享节点计算结果，用于性能优化与节省 Token/API 开销。

运行方式：
    uv run python 14_node_caching.py
"""

import time
from typing import TypedDict

from langgraph.cache.memory import InMemoryCache
from langgraph.func import entrypoint, task
from langgraph.graph import END, START, StateGraph
from langgraph.types import CachePolicy


# ============================================================================
# 场景 1：基础节点缓存 + 流式输出中识别缓存命中 (__metadata__.cached)
# ============================================================================
class ComputeState(TypedDict):
    x: int
    result: int


def demo_basic_node_caching():
    print("\n" + "-" * 60)
    print("【场景 1：基础节点缓存与 __metadata__.cached 命中识别】")
    print("-" * 60)

    call_count = {"expensive_node": 0}

    def expensive_compute(state: ComputeState) -> dict:
        call_count["expensive_node"] += 1
        print(f"  [Node: expensive_compute] 正在执行耗时计算 (x={state['x']})，累计真实执行: {call_count['expensive_node']} 次")
        return {"result": state["x"] * state["x"]}

    builder = StateGraph(ComputeState)
    builder.add_node(
        "expensive_compute",
        expensive_compute,
        cache_policy=CachePolicy(ttl=60),  # 缓存 60 秒
    )
    builder.add_edge(START, "expensive_compute")
    builder.add_edge("expensive_compute", END)

    # 必须在 compile() 时传入 cache 后端，CachePolicy 才会生效
    graph = builder.compile(cache=InMemoryCache())

    # 第 1 次调用：缓存未命中，真实执行节点
    events_1 = list(graph.stream({"x": 7, "result": 0}, stream_mode="updates"))
    print(f"  第 1 次调用 (x=7) 事件: {events_1}")
    assert call_count["expensive_node"] == 1
    assert events_1[0]["expensive_compute"]["result"] == 49
    assert "__metadata__" not in events_1[0]

    # 第 2 次调用相同输入：直接命中缓存，节点函数不会执行！
    events_2 = list(graph.stream({"x": 7, "result": 0}, stream_mode="updates"))
    print(f"  第 2 次调用 (x=7) 事件: {events_2}")
    assert call_count["expensive_node"] == 1  # 真实执行次数仍为 1
    assert events_2[0]["expensive_compute"]["result"] == 49
    assert events_2[0].get("__metadata__", {}).get("cached") is True

    # 第 3 次调用不同输入：缓存未命中，重新执行
    events_3 = list(graph.stream({"x": 9, "result": 0}, stream_mode="updates"))
    print(f"  第 3 次调用 (x=9) 事件: {events_3}")
    assert call_count["expensive_node"] == 2
    assert events_3[0]["expensive_compute"]["result"] == 81


# ============================================================================
# 场景 2：自定义 key_func（忽略 request_id 等易变字段，提高缓存命中率）
# ============================================================================
class SearchState(TypedDict):
    request_id: str  # 每次请求都不同的追踪 ID（不应参与缓存键计算）
    query: str       # 真正决定查询结果的业务字段
    answer: str


def demo_custom_key_func():
    print("\n" + "-" * 60)
    print("【场景 2：自定义 key_func 忽略无关状态字段】")
    print("-" * 60)

    search_calls = {"count": 0}

    def semantic_search_node(state: SearchState) -> dict:
        search_calls["count"] += 1
        print(
            f"  [Node: semantic_search] 执行真实检索 (request_id={state['request_id']}, query='{state['query']}')"
        )
        return {"answer": f"关于「{state['query'].strip()}」的权威检索结果"}

    def normalize_query_key(state: SearchState) -> str:
        """只根据标准化后的 query 生成缓存键，忽略 request_id 与首尾空格/大小写差异。"""
        return state["query"].strip().lower()

    builder = StateGraph(SearchState)
    builder.add_node(
        "semantic_search",
        semantic_search_node,
        cache_policy=CachePolicy(key_func=normalize_query_key, ttl=300),
    )
    builder.add_edge(START, "semantic_search")
    builder.add_edge("semantic_search", END)

    graph = builder.compile(cache=InMemoryCache())

    # 请求 1：首次查询
    res1 = graph.invoke({"request_id": "req-001", "query": "  LangGraph Cache ", "answer": ""})
    # 请求 2：request_id 不同且大小写不同，但标准化后的 query 一致 -> 命中缓存！
    res2 = graph.invoke({"request_id": "req-002", "query": "langgraph cache", "answer": ""})

    print(f"  请求 1 结果: {res1['answer']}")
    print(f"  请求 2 结果: {res2['answer']}")
    print(f"  节点真实执行次数: {search_calls['count']}（2 次不同 request_id 请求仅执行 1 次！）")
    assert search_calls["count"] == 1
    assert res1["answer"] == res2["answer"]


# ============================================================================
# 场景 3：TTL 过期失效机制
# ============================================================================
class QuoteState(TypedDict):
    symbol: str
    price_tag: str


def demo_ttl_expiration():
    print("\n" + "-" * 60)
    print("【场景 3：TTL (Time-To-Live) 缓存过期自动刷新】")
    print("-" * 60)

    fetch_count = {"count": 0}

    def fetch_stock_quote(state: QuoteState) -> dict:
        fetch_count["count"] += 1
        print(f"  [Node: fetch_quote] 拉取 {state['symbol']} 最新报价（第 {fetch_count['count']} 次拉取）")
        return {"price_tag": f"{state['symbol']}@v{fetch_count['count']}"}

    builder = StateGraph(QuoteState)
    builder.add_node(
        "fetch_quote",
        fetch_stock_quote,
        cache_policy=CachePolicy(ttl=1),  # 1 秒后过期
    )
    builder.add_edge(START, "fetch_quote")
    builder.add_edge("fetch_quote", END)
    graph = builder.compile(cache=InMemoryCache())

    r1 = graph.invoke({"symbol": "GOOG", "price_tag": ""})
    r2 = graph.invoke({"symbol": "GOOG", "price_tag": ""})
    print(f"  TTL 过期前：第 1 次={r1['price_tag']}，第 2 次={r2['price_tag']}（命中缓存）")
    assert r1["price_tag"] == r2["price_tag"] == "GOOG@v1"
    assert fetch_count["count"] == 1

    print("  ⏳ 等待 1.1 秒让 1s TTL 过期...")
    time.sleep(1.1)

    r3 = graph.invoke({"symbol": "GOOG", "price_tag": ""})
    print(f"  TTL 过期后：第 3 次={r3['price_tag']}（缓存失效，重新拉取）")
    assert r3["price_tag"] == "GOOG@v2"
    assert fetch_count["count"] == 2


# ============================================================================
# 场景 4：Functional API 中的 @task(cache_policy=...)
# ============================================================================
def demo_functional_api_caching():
    print("\n" + "-" * 60)
    print("【场景 4：Functional API 中使用 @task(cache_policy=...)】")
    print("-" * 60)

    task_calls = {"count": 0}

    @task(cache_policy=CachePolicy(ttl=60))
    def embed_text(text: str) -> list[float]:
        task_calls["count"] += 1
        print(f"  [@task: embed_text] 计算文本向量 '{text}'（第 {task_calls['count']} 次）")
        return [float(len(text)), 0.42, 0.99]

    @entrypoint(cache=InMemoryCache())
    def embedding_workflow(text: str) -> dict:
        vec = embed_text(text).result()
        return {"text": text, "vector": vec}

    out1 = embedding_workflow.invoke("LangGraph 节点缓存")
    out2 = embedding_workflow.invoke("LangGraph 节点缓存")

    print(f"  第 1 次输出: {out1}")
    print(f"  第 2 次输出: {out2}")
    print(f"  @task 真实执行次数: {task_calls['count']}")
    assert out1 == out2
    assert task_calls["count"] == 1


# ============================================================================
# 场景 5：set_node_defaults(cache_policy=...) 全图默认缓存与单节点覆盖
# ============================================================================
class PipelineCacheState(TypedDict):
    topic: str
    outline: str
    live_metric: str


def demo_set_node_defaults_caching():
    print("\n" + "-" * 60)
    print("【场景 5：set_node_defaults(cache_policy=...) 全图默认缓存与单节点覆盖】")
    print("-" * 60)

    counts = {"step_default": 0, "step_override": 0}

    def generate_outline(state: PipelineCacheState) -> dict:
        counts["step_default"] += 1
        print(f"  [Node: step_default] 生成大纲（继承全图默认 ttl=60s，第 {counts['step_default']} 次执行）")
        return {"outline": f"{state['topic']}-大纲v{counts['step_default']}"}

    def fetch_live_metric(state: PipelineCacheState) -> dict:
        counts["step_override"] += 1
        print(f"  [Node: step_override] 获取实时指标（单节点覆盖 ttl=1s，第 {counts['step_override']} 次执行）")
        return {"live_metric": f"score=9{counts['step_override']}"}

    builder = StateGraph(PipelineCacheState)
    # 1. 为整张图的所有普通节点设置默认缓存策略（compile() 时生效；不会应用于 error_handler，也不会被子图继承）
    builder.set_node_defaults(cache_policy=CachePolicy(ttl=60))

    # 2. step_default 不传 cache_policy -> 自动继承全图默认的 CachePolicy(ttl=60)
    builder.add_node("step_default", generate_outline)
    # 3. step_override 显式传入自己的 CachePolicy(ttl=1) -> 优先覆盖全图默认策略
    builder.add_node(
        "step_override",
        fetch_live_metric,
        cache_policy=CachePolicy(ttl=1),
    )
    builder.add_edge(START, "step_default")
    builder.add_edge("step_default", "step_override")
    builder.add_edge("step_override", END)

    graph = builder.compile(cache=InMemoryCache())
    payload: PipelineCacheState = {"topic": "LangGraph", "outline": "", "live_metric": ""}

    # 第 1 次运行：两个节点均未命中缓存，各执行 1 次
    r1 = graph.invoke(payload)
    # 第 2 次立即运行：两个节点均命中缓存，真实执行次数都不变
    r2 = graph.invoke(payload)
    print(f"  立即重跑：r1={r1['outline']}/{r1['live_metric']}，r2={r2['outline']}/{r2['live_metric']}")
    assert counts == {"step_default": 1, "step_override": 1}

    # 等待 1.1 秒：step_override 的 1s 缓存过期重跑，而继承 60s 默认缓存的 step_default 依然命中缓存！
    print("  ⏳ 等待 1.1 秒（step_override 1s 缓存过期，step_default 60s 默认缓存仍有效）...")
    time.sleep(1.1)

    r3 = graph.invoke(payload)
    print(f"  1.1s 后重跑：r3={r3['outline']}/{r3['live_metric']}")
    print(f"  节点执行统计: step_default={counts['step_default']} 次（仍命中全图默认缓存）, step_override={counts['step_override']} 次（已按覆盖 TTL 刷新）")
    assert counts["step_default"] == 1
    assert counts["step_override"] == 2
    assert r3["outline"] == "LangGraph-大纲v1"
    assert r3["live_metric"] == "score=92"


# ============================================================================
# 主入口：独立执行与验证
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("第 14 章：节点缓存机制 (CachePolicy & InMemoryCache)")
    print("=" * 60)

    demo_basic_node_caching()
    demo_custom_key_func()
    demo_ttl_expiration()
    demo_functional_api_caching()
    demo_set_node_defaults_caching()

    print("\n✅ 第 14 章执行与验证成功！")
