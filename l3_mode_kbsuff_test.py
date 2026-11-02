# -*- coding: utf-8 -*-
"""收尾2: MIX_MODE=kb_sufficient 配置切换补测。

验证 1/2 都在默认 tool_calling 下跑过; 本脚本切 MIX_MODE=kb_sufficient,
对灰区 14 条(与 l3_mixed 相同)端到端跑 ask(), 确认路由结果与阶段一一致
(阶段一灰区 14 条实测 11/14)。不改代码, 只验证配置切换路径没坏。
"""
import io
import os
import sys
from contextlib import redirect_stdout

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

os.environ["MIX_MODE"] = "kb_sufficient"   # 切模式(必须在 import rag_chat 前)

from langchain_ollama import OllamaEmbeddings
import rag_chat as rc

# 灰区 14 条(与 l3_mixed / 替换前阶段一验收相同)
GREY_CASES = [
    ("什么是pv操作？",              "B", "KB"),
    ("什么是B树？",                "B", "KB"),
    ("什么是编译原理中的词法分析？",  "B", "KB"),
    ("什么是哈希表？",              "B", "KB"),
    ("什么是页表？",                "B", "KB"),
    ("什么是LRU缓存？",            "B", "KB"),
    ("什么是零拷贝？",              "B", "KB"),
    ("2026年最新发布的开源大模型是什么？", "C", "WEB"),
    ("2026年AI领域有哪些最新进展？",  "C", "WEB"),
    ("最新版PyTorch的版本号是多少？", "C", "WEB"),
    ("2026年10月AI大模型最新发布了哪些？", "C", "WEB"),
    ("2026年最新款的游戏显卡是什么？", "C", "WEB"),
    ("苹果最新发布的手机是什么型号？",  "C", "WEB"),
    ("最新发布的iPhone是什么型号？",  "C", "WEB"),
]


def classify_route(detail):
    if "纯阈值直连" in detail:
        return "非灰区(直连)"   # 不应出现: 14 条全是灰区
    if "灰区kb_sufficient" in detail:
        return "灰区kb_sufficient"
    if "灰区tool calling" in detail:
        return "灰区tool calling"  # 不应出现: 已切 kb_sufficient
    return "未知:" + detail[:30]


def main():
    print("===== 补测2: MIX_MODE=kb_sufficient 灰区 14 条 =====")
    print(f"MIX_MODE={rc.MIX_MODE} | 灰区 [{rc.MIX_LOW}, {rc.MIX_HIGH}]")
    assert rc.MIX_MODE == "kb_sufficient", "MIX_MODE 未生效"
    print("确认: 配置切换生效。\n")

    cache = (OllamaEmbeddings(model=rc.EMBED_MODEL, base_url=rc.OLLAMA_BASE),
             *rc.load_index())

    print(f"{'问题':<36}{'类':<3}{'预期':<5}{'实际':<6}{'路由路径':<22}{'正确':<4}{'best':<8}")
    print("-" * 100)

    rows = []
    for q, cat, expect in GREY_CASES:
        buf = io.StringIO()
        with redirect_stdout(buf):
            mode, route_detail, best = rc.ask(q, _cache=cache)
        path = classify_route(route_detail)
        ok = (mode == ("knowledge_base" if expect == "KB" else "web_search"))
        rows.append(dict(q=q, cat=cat, expect=expect, mode=mode,
                         path=path, best=best, ok=ok))
        print(f"{q:<36}{cat:<3}{expect:<5}{mode:<6}{path:<22}{'✓' if ok else '✗'}{best:<8.3f}")

    n_ok = sum(1 for r in rows if r["ok"])
    print(f"\n===== 结果 =====  灰区 14 条正确率 {n_ok}/14 = {n_ok/14*100:.0f}%")
    b_ok = sum(1 for r in rows if r["cat"] == "B" and r["ok"])
    c_ok = sum(1 for r in rows if r["cat"] == "C" and r["ok"])
    print(f"B 类(应KB): {b_ok}/7  |  C 类(应WEB): {c_ok}/7")
    print(f"阶段一灰区参考: 11/14 (B 4/7 + C 7/7); kb_sufficient 模式应与之相当。")

    # 误判明细
    bad = [r for r in rows if not r["ok"]]
    if bad:
        print(f"\n误判 {len(bad)} 条(与阶段一行为对比):")
        for r in bad:
            print(f"  [{r['cat']}] {r['q']} best={r['best']:.3f} "
                  f"(预期{r['expect']} 实际{r['mode']})")
    else:
        print("\n无误判。")


if __name__ == "__main__":
    main()
