# -*- coding: utf-8 -*-
"""
L1 补测: Tavily(真实 key) vs Bing, 4 条查询对比
- 3 条实时问题(技术更新类/近三天科技新闻/依赖库最新版本号)
- 1 条中文查询("2026冬奥会举办城市")
记录: 条数 / 相关性 / 延迟 / snippet长度 / content长度(content 不截断)
打印: Tavily 请求参数(key打码) + 结构化结果
"""
import sys
import time
import json

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from web_providers import (
    TavilyProvider, BingProvider, simplify_query,
)

REALTIME_QS = [
    "LangChain 最新版本更新了什么",        # 技术更新类
    "2026年10月AI大模型最新发布",          # 近三天科技新闻类
    "PyTorch 最新版本号是多少",            # 依赖库最新版本号
]
CHINESE_Q = "2026冬奥会举办城市"


def run(label, q, provider):
    kw = simplify_query(q)
    t0 = time.time()
    try:
        res = provider.search(kw, top_k=5)
        dt = time.time() - t0
    except Exception as e:
        dt = time.time() - t0
        print(f"\n### {label}\n问题: {q}\n搜索词: {kw}\n[异常] {e} (耗时 {dt:.1f}s)")
        return None
    print(f"\n### {label}\n问题: {q}\n搜索词: {kw}\n耗时: {dt:.1f}s | 条数: {len(res)}")
    for i, r in enumerate(res, 1):
        snip = r["snippet"]
        content = r["content"]
        pub = r["published_at"] or "-"
        print(f"  [{i}] {r['title']}")
        print(f"      url: {r['url']}")
        print(f"      published_at: {pub} | snippet_len={len(snip)} | content_len={len(content)}")
        if snip:
            print(f"      snippet: {snip[:200]}")
        if content:
            print(f"      content(完整): {content}")
    return res


def main():
    # 打码校验
    tp = TavilyProvider()
    print("Tavily key 打码:", tp._mask_key(), "| 实际长度:", len(tp.api_key))
    print("Tavily last_params:", tp.last_params)

    print("\n" + "=" * 70)
    print("TAVILY 实测 (真实 key)")
    for q in REALTIME_QS:
        run(f"Tavily: {q}", q, tp)

    print("\n" + "=" * 70)
    print("BING 对照")
    bp = BingProvider()
    for q in REALTIME_QS:
        run(f"Bing: {q}", q, bp)

    print("\n" + "=" * 70)
    print("中文查询对比")
    run("Tavily 中文: " + CHINESE_Q, CHINESE_Q, tp)
    run("Bing 中文: " + CHINESE_Q, CHINESE_Q, bp)

    print("\n" + "=" * 70)
    print("Tavily 实际请求参数(key 已打码)")
    # 重新触发一次记录
    tp.search(simplify_query(REALTIME_QS[0]), top_k=5)
    print(json.dumps(tp.last_params, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
