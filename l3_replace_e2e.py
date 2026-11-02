# -*- coding: utf-8 -*-
"""验证1(替换后必做): 端到端 28 条完整走 ask(), 记录每条实际路由路径。

要求:
  - A 类 7 条全部走 KB 直连, 不经过灰区判定
  - D 类 7 条全部走 WEB 直连, 不经过灰区判定
  - 灰区 14 条走 tool calling, 结果与 l3_mixed 一致
  - 总正确率与拼合数据 28/28 一致; 若不一致说明拼合假设有漏洞, 如实报
  - 额外: 打印 A 类最低(数据库事务)实跑相似度, 供边界脆弱性确认
"""
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from langchain_ollama import OllamaEmbeddings
import rag_chat as rc

# 28 条四类问题 (与 l3_compare_v2 / l3_mixed 相同)
CASES = [
    # A 高置信 KB (>0.75, 预期 KB 直连)
    ("什么是进程和线程？",              "A", "KB", 0.787),
    ("TCP三次握手的过程是什么？",        "A", "KB", 0.791),
    ("什么是HTTP协议？",                "A", "KB", 0.780),
    ("什么是虚拟内存的分页机制？",       "A", "KB", 0.761),
    ("什么是二叉搜索树？",              "A", "KB", 0.756),
    ("二叉树的遍历方式有哪些？",         "A", "KB", 0.755),
    ("什么是数据库事务？",              "A", "KB", 0.750),
    # B 低置信 KB (灰区, 预期 KB)
    ("什么是pv操作？",                  "B", "KB", 0.675),
    ("什么是B树？",                    "B", "KB", 0.666),
    ("什么是编译原理中的词法分析？",     "B", "KB", 0.662),
    ("什么是哈希表？",                  "B", "KB", 0.653),
    ("什么是页表？",                    "B", "KB", 0.644),
    ("什么是LRU缓存？",                "B", "KB", 0.642),
    ("什么是零拷贝？",                  "B", "KB", 0.589),
    # C 灰区实时 (灰区, 预期 WEB)
    ("2026年最新发布的开源大模型是什么？", "C", "WEB", 0.620),
    ("2026年AI领域有哪些最新进展？",    "C", "WEB", 0.610),
    ("最新版PyTorch的版本号是多少？",   "C", "WEB", 0.607),
    ("2026年10月AI大模型最新发布了哪些？", "C", "WEB", 0.602),
    ("2026年最新款的游戏显卡是什么？",   "C", "WEB", 0.584),
    ("苹果最新发布的手机是什么型号？",   "C", "WEB", 0.564),
    ("最新发布的iPhone是什么型号？",    "C", "WEB", 0.560),
    # D 无答案 (<0.55, 预期 WEB 直连)
    ("2026年世界杯冠军是谁？",          "D", "WEB", 0.495),
    ("2026年NBA总冠军是哪支球队？",     "D", "WEB", 0.487),
    ("2026年苹果秋季发布会的日期？",     "D", "WEB", 0.486),
    ("2026年冬奥会的举办城市是哪里？",   "D", "WEB", 0.485),
    ("华为mate70的发布时间是什么时候？", "D", "WEB", 0.458),
    ("2026年诺贝尔物理学奖得主是谁？",   "D", "WEB", 0.457),
    ("2026年美国总统是谁？",            "D", "WEB", 0.447),
]


def classify_route(route_detail):
    """按 route_detail 分类实际路由路径"""
    if "纯阈值直连→KB" in route_detail:
        return "A段KB直连"
    if "纯阈值直连→WEB" in route_detail:
        return "D段WEB直连"
    if "灰区tool calling" in route_detail:
        return "灰区tool calling"
    if "灰区kb_sufficient" in route_detail:
        return "灰区fallback(kb_sufficient)"
    return "未知:" + route_detail[:30]


def main():
    print("===== 验证1: 端到端 28 条实跑 (完整走 ask()) =====")
    print("预加载索引/embedding (一次性, 避免 28 次加载 606MB)...")
    cache = (OllamaEmbeddings(model=rc.EMBED_MODEL, base_url=rc.OLLAMA_BASE),
             *rc.load_index())
    print(f"索引加载完成。灰区 [{rc.MIX_LOW}, {rc.MIX_HIGH}], 模式={rc.MIX_MODE}")

    print(f"\n{'问题':<34}{'类':<3}{'预期':<5}{'实际':<6}{'路由路径':<24}{'正确':<4}{'best':<8}")
    print("-" * 100)

    rows = []
    for q, cat, expect, _eb in CASES:
        buf = io.StringIO()
        with redirect_stdout(buf):  # 吞掉 ask() 内部打印, 只看路由汇总
            mode, route_detail, best = rc.ask(q, _cache=cache)
        path = classify_route(route_detail)
        # 判定: expect 用 "KB"/"WEB", mode 用 "knowledge_base"/"web_search", 需映射
        ok = (mode == ("knowledge_base" if expect == "KB" else "web_search"))
        rows.append(dict(q=q, cat=cat, expect=expect, mode=mode,
                         path=path, best=best, ok=ok))
        print(f"{q:<34}{cat:<3}{expect:<5}{mode:<6}{path:<24}{'✓' if ok else '✗'}{best:<8.3f}")

    # ---- 分项统计 ----
    n_ok = sum(1 for r in rows if r["ok"])
    print(f"\n===== 结果 =====  总正确率 {n_ok}/28 = {n_ok/28*100:.0f}%")

    def cat_ok(cat):
        rs = [r for r in rows if r["cat"] == cat]
        return f"{sum(1 for r in rs if r['ok'])}/{len(rs)}"

    print(f"A 类(>0.75, 应KB直连): 正确 {cat_ok('A')}")
    print(f"D 类(<0.55, 应WEB直连): 正确 {cat_ok('D')}")
    print(f"B 类(灰区, 应KB): 正确 {cat_ok('B')}")
    print(f"C 类(灰区, 应WEB): 正确 {cat_ok('C')}")

    # ---- A/D 段是否绕开灰区判定 ----
    a_paths = [r["path"] for r in rows if r["cat"] == "A"]
    d_paths = [r["path"] for r in rows if r["cat"] == "D"]
    a_all_direct = all(p == "A段KB直连" for p in a_paths)
    d_all_direct = all(p == "D段WEB直连" for p in d_paths)
    print(f"\nA 类 7 条全部走 KB 直连(不经过灰区判定): {'✓' if a_all_direct else '✗ ' + str(a_paths)}")
    print(f"D 类 7 条全部走 WEB 直连(不经过灰区判定): {'✓' if d_all_direct else '✗ ' + str(d_paths)}")

    # ---- 灰区 14 条路径 ----
    grey = [r for r in rows if r["cat"] in ("B", "C")]
    grey_tc = [r for r in grey if r["path"] == "灰区tool calling"]
    grey_fb = [r for r in grey if r["path"] != "灰区tool calling"]
    print(f"灰区 14 条: tool calling {len(grey_tc)} 条, fallback {len(grey_fb)} 条")
    if grey_fb:
        print("  [注意] 灰区出现 fallback 路径(应与 l3_mixed 的 14/14 全 tool calling 对比):")
        for r in grey_fb:
            print(f"    {r['q']} -> {r['path']}")
    grey_ok = sum(1 for r in grey if r["ok"])
    print(f"灰区正确率: {grey_ok}/14")

    # ---- 与拼合数据 28/28 对比 ----
    if n_ok == 28:
        print("总正确率 28/28 与拼合数据一致 —— 拼合假设成立, 无漏洞。")
    else:
        print(f"!! 总正确率 {n_ok}/28 与拼合数据 28/28 不一致 —— 拼合假设有漏洞!")
        for r in rows:
            if not r["ok"]:
                print(f"    误判: [{r['cat']}] {r['q']} (预期{r['expect']} 实际{r['mode']}, "
                      f"best={r['best']:.3f}, 路径={r['path']})")

    # ---- 边界脆弱性: A 类最低(数据库事务) ----
    dt = next(r for r in rows if r["q"] == "什么是数据库事务？")
    print(f"\n===== 边界脆弱性 =====")
    print(f"\"什么是数据库事务？\" 实跑 best = {dt['best']:.4f} "
          f"(上界 MIX_HIGH={rc.MIX_HIGH})")
    if dt["best"] >= rc.MIX_HIGH - 0.01:
        print("  → 贴边: best 距上界 ≤ 0.01, 向量库微小变化即可推入灰区。")
    else:
        print(f"  → best 距上界 {rc.MIX_HIGH - dt['best']:.4f}, 未贴边。")

    # 保存结果供验证3引用
    import json
    with open(Path(__file__).resolve().parent / "_e2e28_result.json", "w",
              encoding="utf-8") as f:
        json.dump([dict(r) for r in rows], f, ensure_ascii=False, indent=1)
    print("\n结果已存 _e2e28_result.json")


if __name__ == "__main__":
    main()
