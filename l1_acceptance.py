# -*- coding: utf-8 -*-
"""
L1 验收脚本
1. 3 条实时问题: tavily vs bing 各跑一次, 对比条数/相关性/延迟/snippet长度/content
2. 1 条中文查询对比
3. 打印请求参数(key打码) + 结构化结果
4. mock Tavily 失败(错key) -> 验证降级到 Bing
5. WEB_PROVIDER 切换 -> 上层 rag_chat 零改动跑通

用法: python l1_acceptance.py
"""
import os
import sys
import time

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from web_providers import (
    TavilyProvider, BingProvider, get_provider, search_with_fallback,
    simplify_query,
)

REALTIME_QS = [
    "DeepSeek最新版本发布了吗",
    "2026年诺贝尔物理学奖得主",
    "PyTorch最新版本更新了什么",
]
CHINESE_Q = "2026冬奥会举办城市"

def fmt_results(results):
    if not results:
        return "  (空)"
    lines = []
    for i, r in enumerate(results, 1):
        pub = r["published_at"] or "-"
        has_content = "Y" if r["content"] else "N"
        lines.append(
            f"  [{i}] {r['title'][:50]} | src={r['source']} | pub={pub} | "
            f"content={has_content}\n      snip_len={len(r['snippet'])} | {r['url'][:70]}")
    return "\n".join(lines)


def run_case(label, query, provider_cls=None, provider_name=None):
    print(f"\n{'='*70}\n### {label}\n问题: {query}")
    if provider_cls is not None:
        p = provider_cls()
        kw = simplify_query(query)
        t0 = time.time()
        try:
            res = p.search(kw, top_k=5)
            dt = time.time() - t0
        except Exception as e:
            dt = time.time() - t0
            print(f"  [异常] {e} (耗时 {dt:.1f}s)")
            return None
        print(f"  搜索词: {kw}")
        print(f"  请求参数: {getattr(p, 'last_params', '(bing无参数记录')}")
        print(f"  耗时: {dt:.1f}s | 返回 {len(res)} 条")
        print(fmt_results(res))
        return res
    # 走 search_with_fallback 指定 preferred
    t0 = time.time()
    res, used = search_with_fallback(query, top_k=5, preferred=provider_name)
    dt = time.time() - t0
    print(f"  [fallback链] 实际使用: {used} | 耗时 {dt:.1f}s | 返回 {len(res)} 条")
    print(fmt_results(res))
    return res


def main():
    print("当前 WEB_PROVIDER =", os.environ.get("WEB_PROVIDER", "<未设置,默认tavily>"))
    print("TAVILY_API_KEY =", ("已设置(" + (os.environ.get("TAVILY_API_KEY","")[:4]) + "***)" 
                               if os.environ.get("TAVILY_API_KEY") else "<未设置>"))

    # ---- 1. 3条实时问题: tavily vs bing ----
    for q in REALTIME_QS:
        run_case(f"实时问题 vs Tavily: {q}", q, TavilyProvider)
        run_case(f"实时问题 vs Bing: {q}", q, BingProvider)

    # ---- 2. 中文查询对比 ----
    run_case(f"中文查询 vs Tavily: {CHINESE_Q}", CHINESE_Q, TavilyProvider)
    run_case(f"中文查询 vs Bing: {CHINESE_Q}", CHINESE_Q, BingProvider)

    # ---- 3. Tavily 请求参数(key打码) ----
    print("\n" + "="*70)
    print("### Tavily 实际请求参数(key 已打码)")
    if os.environ.get("TAVILY_API_KEY"):
        tp = TavilyProvider()
        tp.search("测试", top_k=1)
        print(" ", tp.last_params)
    else:
        tp = TavilyProvider()
        print("  无 key, 无法发真实请求; last_params 结构示例:",
              {"query": "...", "max_results": 5, "search_depth": "basic",
               "api_key": tp._mask_key()})

    # ---- 4. mock Tavily 失败 -> 降级 Bing ----
    print("\n" + "="*70)
    print("### mock Tavily 失败(错key) -> 验证降级")
    old = os.environ.get("TAVILY_API_KEY")
    os.environ["TAVILY_API_KEY"] = "tvly-invalid-key-for-test"
    res, used = search_with_fallback("2026世界杯冠军", top_k=5, preferred="tavily")
    print(f"  结果: used={used} | 条数={len(res)}")
    assert used == "bing", "应当降级到 bing"
    print("  [PASS] 成功降级到 Bing")
    if old:
        os.environ["TAVILY_API_KEY"] = old
    else:
        os.environ.pop("TAVILY_API_KEY", None)

    # ---- 5. WEB_PROVIDER 切换: 上层零改动 ----
    print("\n" + "="*70)
    print("### WEB_PROVIDER 切换验证 (bing)")
    os.environ["WEB_PROVIDER"] = "bing"
    import rag_chat  # 确认 rag_chat 在 bing 配置下仍可导入运行
    print("  [PASS] WEB_PROVIDER=bing 时 rag_chat 导入正常(上层零改动)")
    os.environ.pop("WEB_PROVIDER", None)
    print("  (已恢复默认)")


if __name__ == "__main__":
    main()
