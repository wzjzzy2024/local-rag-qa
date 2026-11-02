# -*- coding: utf-8 -*-
"""盲测(收尾1): 20 条全新问题(四类各5), 未参与 0.55/0.75 调参, 端到端跑混合路由。

28/28 是调参集自测(28条同时是调参集与验证集, 存在乐观偏差);
本盲测用全新问题检验真实水平。预期总正确率可能掉到 ~85%, 属正常, 如实记录。

记录:
  - 总正确率
  - B/C 灰区在新样本上的路由正确率
  - 新样本中贴边条目(距 0.55 或 0.75 < 0.005)
"""
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from langchain_ollama import OllamaEmbeddings
import rag_chat as rc

# 20 条全新问题(未参与任何调参)
CASES = [
    # A 高置信 KB (预期 KB 直连)
    ("什么是死锁？",              "A", "KB"),
    ("什么是快速排序算法？",        "A", "KB"),
    ("什么是二叉树的中序遍历？",     "A", "KB"),
    ("什么是关系数据库中的外键？",   "A", "KB"),
    ("什么是哈希冲突？",           "A", "KB"),
    # B 灰区 KB (预期 KB, tool calling 判定)
    ("什么是分页存储管理？",        "B", "KB"),
    ("什么是银行家算法？",          "B", "KB"),
    ("什么是临界区？",             "B", "KB"),
    ("什么是中断？",              "B", "KB"),
    ("什么是DMA传输？",           "B", "KB"),
    # C 灰区实时 (预期 WEB, tool calling 判定)
    ("2026年诺贝尔化学奖得主是谁？",  "C", "WEB"),
    ("2026年最新发布的旗舰手机有哪些？", "C", "WEB"),
    ("OpenAI最新的模型是什么？",    "C", "WEB"),
    ("2026年AI芯片领域有哪些最新进展？", "C", "WEB"),
    ("2026年双十一的成交额是多少？",  "C", "WEB"),
    # D 无答案 (预期 WEB 直连)
    ("2026年格莱美年度专辑是哪个？",  "D", "WEB"),
    ("2026年F1世界冠军是谁？",      "D", "WEB"),
    ("2026年奥斯卡最佳影片是哪部？",  "D", "WEB"),
    ("2026年NBA全明星赛MVP是谁？",  "D", "WEB"),
    ("2026年诺贝尔文学奖得主是谁？",  "D", "WEB"),
]


def classify_route(detail):
    if "纯阈值直连→KB" in detail:
        return "A段KB直连"
    if "纯阈值直连→WEB" in detail:
        return "D段WEB直连"
    if "灰区tool calling" in detail:
        return "灰区tool calling"
    if "灰区kb_sufficient" in detail:
        return "灰区fallback(kb)"
    return "未知:" + detail[:30]


def main():
    print("===== 盲测: 20 条全新问题端到端 (未参与调参) =====")
    print(f"灰区 [{rc.MIX_LOW}, {rc.MIX_HIGH}], 模式={rc.MIX_MODE}")
    cache = (OllamaEmbeddings(model=rc.EMBED_MODEL, base_url=rc.OLLAMA_BASE),
             *rc.load_index())
    print("索引加载完成。\n")

    print(f"{'问题':<38}{'类':<3}{'预期':<5}{'实际':<6}{'路由路径':<22}{'正确':<4}{'best':<8}")
    print("-" * 104)

    rows = []
    for q, cat, expect in CASES:
        buf = io.StringIO()
        with redirect_stdout(buf):
            mode, route_detail, best = rc.ask(q, _cache=cache)
        path = classify_route(route_detail)
        ok = (mode == ("knowledge_base" if expect == "KB" else "web_search"))
        rows.append(dict(q=q, cat=cat, expect=expect, mode=mode,
                         path=path, best=best, ok=ok))
        print(f"{q:<38}{cat:<3}{expect:<5}{mode:<6}{path:<22}{'✓' if ok else '✗'}{best:<8.3f}")

    n_ok = sum(1 for r in rows if r["ok"])
    print(f"\n===== 总正确率 =====  {n_ok}/20 = {n_ok/20*100:.0f}%")

    def cat_stat(cat):
        rs = [r for r in rows if r["cat"] == cat]
        return f"{sum(1 for r in rs if r['ok'])}/{len(rs)}"

    print(f"A 类(应KB): {cat_stat('A')}  |  B 类(灰区应KB): {cat_stat('B')}  "
          f"|  C 类(灰区应WEB): {cat_stat('C')}  |  D 类(应WEB): {cat_stat('D')}")

    # B/C 灰区单独正确率
    grey = [r for r in rows if r["cat"] in ("B", "C")]
    grey_ok = sum(1 for r in grey if r["ok"])
    print(f"B/C 灰区在新样本上的路由正确率: {grey_ok}/{len(grey)} = "
          f"{grey_ok/len(grey)*100:.0f}%")

    # 贴边条目
    print(f"\n===== 贴边检查 (距 0.55 或 0.75 < 0.005) =====")
    edge = [r for r in rows
            if abs(r["best"] - 0.55) < 0.005 or abs(r["best"] - 0.75) < 0.005]
    if edge:
        for r in edge:
            d_low = abs(r["best"] - 0.55)
            d_high = abs(r["best"] - 0.75)
            side = "下界0.55" if d_low < d_high else "上界0.75"
            print(f"  {r['q']} best={r['best']:.4f} 距{side} "
                  f"{min(d_low, d_high):.4f}")
    else:
        print("  无贴边条目")

    # 误判明细(如实)
    bad = [r for r in rows if not r["ok"]]
    if bad:
        print(f"\n===== 误判明细 (共 {len(bad)} 条) =====")
        for r in bad:
            print(f"  [{r['cat']}] {r['q']} (预期{r['expect']} 实际{r['mode']}, "
                  f"best={r['best']:.3f}, 路径={r['path']})")

    print(f"\n说明: 28/28 是调参集自测(乐观偏差未排除); 盲测 {n_ok}/20 为未参与调参的真实水平。")

    import json
    with open(Path(__file__).resolve().parent / "_blind20_result.json", "w",
              encoding="utf-8") as f:
        json.dump([dict(r) for r in rows], f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
