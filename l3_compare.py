# -*- coding: utf-8 -*-
"""L3 阶段二并存对比: 阶段一(kb_sufficient 二级判定) vs tool calling 判定

20 条问题, 四类各 5 条 (A 高置信KB / B 低置信KB / C 灰区实时 / D 无答案)。
对比四维度:
  1. 路由决策正确率: 分 A/B/C/D 四类分别报 (KB/WEB 二分类)
  2. 边界带 (B+C) 单独正确率
  3. 延迟: 该走KB / 该走WEB / 该判定为不足 三场景均值 + 延迟不对称影响说明
  4. 判定理由可信度: B/C 两类打印双方判定理由, 人工比对
停止条件(写死):
  - tool calling 边界带正确率 >= 阶段一 且 延迟<=2倍 且 理由更可信 -> 建议替换(停下等确认)
  - tool calling 边界带正确率 < 阶段一 或 延迟>2倍 或 理由明显更差 -> 保留阶段一
  - 打平 -> 保留阶段一, tool calling 列 L4
不改 kb_sufficient, 不替换任何东西, 只做对比。
"""
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import faiss
from langchain_ollama import OllamaEmbeddings, ChatOllama

import rag_chat as rc

TOOLS = [{
    "type": "function",
    "function": {
        "name": "search",
        "description": "联网搜索外部实时信息。当问题涉及新闻、最新版本号、实时事件、"
                       "或本地知识库之外的时效性信息时调用。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词，中文优先"},
                "top_k": {"type": "integer", "description": "返回结果条数，默认5"},
            },
            "required": ["query"],
        },
    },
}]

# (问题, 类别, 预期路由)
CASES = [
    # A 类: 知识库内高置信 (>0.75)
    ("什么是进程和线程？",            "A", "KB"),
    ("TCP三次握手的过程是什么？",      "A", "KB"),
    ("什么是HTTP协议？",            "A", "KB"),
    ("二叉树的遍历方式有哪些？",        "A", "KB"),
    ("什么是数据库事务？",           "A", "KB"),
    # B 类: 知识库内低置信边界带 (0.55-0.68)
    ("什么是pv操作？",              "B", "KB"),
    ("什么是编译原理中的词法分析？",    "B", "KB"),
    ("什么是哈希表？",              "B", "KB"),
    ("什么是页表？",                "B", "KB"),
    ("什么是零拷贝？",              "B", "KB"),
    # C 类: 灰区实时 (0.55-0.65, 内容答不了 -> 回退联网)
    ("2026年AI领域有哪些最新进展？",   "C", "WEB"),
    ("最新版PyTorch的版本号是多少？",  "C", "WEB"),
    ("2026年10月AI大模型最新发布了哪些？", "C", "WEB"),
    ("苹果最新发布的手机是什么型号？",   "C", "WEB"),
    ("最新发布的iPhone是什么型号？",   "C", "WEB"),
    # D 类: 知识库完全无答案 (<0.55)
    ("2026年世界杯冠军是谁？",        "D", "WEB"),
    ("2026年NBA总冠军是哪支球队？",   "D", "WEB"),
    ("2026年冬奥会的举办城市是哪里？", "D", "WEB"),
    ("华为mate70的发布时间是什么时候？", "D", "WEB"),
    ("2026年诺贝尔物理学奖得主是谁？", "D", "WEB"),
]

KB_SUFF_PROMPT = (
    "判断下面的知识库片段是否足以回答用户的问题。\n"
    "问题: {q}\n知识库片段:\n{snippets}\n"
    "如果片段包含答案（或能据此给出可靠答案），回复: 足以回答\n"
    "如果片段只是相关主题背景、缺少回答所需的关键信息"
    "（尤其是实时/最新信息），回复: 不足以回答\n"
    "只输出\"足以回答\"或\"不足以回答\"。"
)

# ---- 阶段一判定: 复制 kb_sufficient 逻辑但捕获原始回复文本（不改原函数） ----
def judge_phase1(llm, question, texts, sel):
    snippets = "\n".join(f"- {texts[i][:150]}" for i, _s in sel[:rc.TOP_K])
    p = KB_SUFF_PROMPT.format(q=question, snippets=snippets)
    t0 = time.time()
    resp = llm.invoke(p)
    dt = time.time() - t0
    ans = resp.content.strip()
    suff = "足以回答" in ans and "不足以回答" not in ans
    return suff, dt, ans


# ---- tool calling 判定: 让模型决定调不调 search ----
def judge_toolcall(llm_tools, question):
    t0 = time.time()
    resp = llm_tools.invoke(question)
    dt = time.time() - t0
    content = getattr(resp, "content", "") or ""
    tool_calls = getattr(resp, "tool_calls", None) or []
    called = bool(tool_calls)
    if called:
        names = [tc["name"] for tc in tool_calls]
        args_list = [tc.get("args", {}) for tc in tool_calls]
        reason = f"TOOL_CALL {names} args={args_list}"
    else:
        reason = f"NO_CALL content_len={len(content)} 首80字:{content[:80]}"
    return called, dt, reason


def main():
    emb = OllamaEmbeddings(model=rc.EMBED_MODEL, base_url=rc.OLLAMA_BASE)
    index, texts, metas = rc.load_index()
    llm = ChatOllama(model=rc.LLM_MODEL, base_url=rc.OLLAMA_BASE, temperature=0.2)
    llm_tools = llm.bind_tools(TOOLS)

    rows = []
    for q, cat, expect in CASES:
        qv = np.array(emb.embed_query(q), dtype="float32").reshape(1, -1)
        faiss.normalize_L2(qv)
        scores, idx = index.search(qv, rc.TOP_K)
        scores, idx = scores[0], idx[0]
        best = float(scores[0]) if len(scores) and scores[0] > -9e9 else 0.0
        sel = [(int(i), float(s)) for i, s in zip(idx, scores)
               if int(i) >= 0 and s > 0]

        # 阶段一
        if best < rc.SCORE_THRESHOLD:
            p1_route, p1_dt, p1_reason = "WEB", 0.0, "best<0.55 直接联网"
        elif best >= rc.SECONDARY_ROUTE_CHECK_MAX:
            p1_route, p1_dt, p1_reason = "KB", 0.0, f"best={best:.3f}>=0.68 高置信跳过"
        else:
            suff, dt, ans = judge_phase1(llm, q, texts, sel)
            p1_route = "KB" if suff else "WEB"
            p1_dt, p1_reason = dt, f"判定回复: {ans}"

        # tool calling
        tc_called, tc_dt, tc_reason = judge_toolcall(llm_tools, q)
        tc_route = "WEB" if tc_called else "KB"

        rows.append(dict(q=q, cat=cat, expect=expect, best=best,
                         p1=p1_route, p1_dt=p1_dt, p1_reason=p1_reason,
                         tc=tc_route, tc_dt=tc_dt, tc_reason=tc_reason))

    # ---- 打印明细 ----
    print(f"===== L3 阶段二并存对比 =====  (CHECK_MAX={rc.SECONDARY_ROUTE_CHECK_MAX})")
    print(f"{'问题':<32}{'类':<3}{'best':<7}{'预期':<5}{'阶段一':<6}{'耗时':<7}"
          f"{'toolcall':<9}{'耗时':<7}{'一致':<5}")
    print("-" * 100)
    for r in rows:
        same = "✓" if r["p1"] == r["tc"] else "✗"
        print(f"{r['q']:<32}{r['cat']:<3}{r['best']:<7.3f}{r['expect']:<5}"
              f"{r['p1']:<6}{r['p1_dt']:<7.2f}{r['tc']:<9}{r['tc_dt']:<7.2f}{same}")

    # ---- 正确率统计 ----
    print("\n===== 1. 四类分项路由正确率 =====")
    print(f"{'类别':<6}{'说明':<30}{'阶段一':<12}{'toolcall':<12}")
    totals = {"p1": 0, "tc": 0}
    cat_stats = {}
    for cat in ["A", "B", "C", "D"]:
        sub = [r for r in rows if r["cat"] == cat]
        p1_ok = sum(1 for r in sub if r["p1"] == r["expect"])
        tc_ok = sum(1 for r in sub if r["tc"] == r["expect"])
        totals["p1"] += p1_ok
        totals["tc"] += tc_ok
        cat_stats[cat] = (p1_ok, tc_ok)
        print(f"{cat:<6}{cat_desc(cat):<30}{p1_ok}/{len(sub):<12}{tc_ok}/{len(sub)}")
    print(f"{'总':<6}{'':<30}{totals['p1']}/{len(rows):<12}{totals['tc']}/{len(rows)}")

    # ---- 2. 边界带 (B+C) ----
    bc = [r for r in rows if r["cat"] in ("B", "C")]
    p1_bc = sum(1 for r in bc if r["p1"] == r["expect"])
    tc_bc = sum(1 for r in bc if r["tc"] == r["expect"])
    print(f"\n===== 2. 边界带(B+C)正确率 =====  (阶段一 {p1_bc}/{len(bc)} = "
          f"{p1_bc/len(bc)*100:.0f}% | toolcall {tc_bc}/{len(bc)} = "
          f"{tc_bc/len(bc)*100:.0f}%)")

    # ---- 3. 延迟三场景 ----
    print("\n===== 3. 延迟三场景均值 =====")
    scenes = [
        ("该走KB (A+B)", [r for r in rows if r["cat"] in ("A", "B")]),
        ("该走WEB (D)", [r for r in rows if r["cat"] == "D"]),
        ("该判定为不足 (C)", [r for r in rows if r["cat"] == "C"]),
    ]
    for name, sub in scenes:
        if not sub:
            continue
        p1_avg = sum(r["p1_dt"] for r in sub) / len(sub)
        tc_avg = sum(r["tc_dt"] for r in sub) / len(sub)
        ratio = tc_avg / p1_avg if p1_avg > 0 else float("inf")
        print(f"{name:<18} 阶段一 {p1_avg:.2f}s | toolcall {tc_avg:.2f}s "
              f"| 比值 {ratio:.1f}x")

    # ---- 4. 判定理由 (B/C 全部) ----
    print("\n===== 4. 判定理由人工比对 (B/C 类) =====")
    for r in rows:
        if r["cat"] not in ("B", "C"):
            continue
        print(f"\n[{r['cat']}] {r['q']} (best={r['best']:.3f}, 预期 {r['expect']})")
        print(f"  阶段一  : {r['p1_reason']}")
        print(f"  toolcall: {r['tc_reason']}")

    # ---- 5. 结论(停止条件) ----
    print("\n===== 5. 结论(按写死停止条件) =====")
    print(f"边界带: 阶段一 {p1_bc}/{len(bc)} ({p1_bc/len(bc)*100:.0f}%) | "
          f"toolcall {tc_bc}/{len(bc)} ({tc_bc/len(bc)*100:.0f}%)")
    # 延迟比值: 取"该判定为不足"与"该走KB"两场景综合看
    kb_scene = [r for r in rows if r["cat"] in ("A", "B")]
    c_scene = [r for r in rows if r["cat"] == "C"]
    p1_lat = (sum(r["p1_dt"] for r in kb_scene) + sum(r["p1_dt"] for r in c_scene)) / (len(kb_scene) + len(c_scene))
    tc_lat = (sum(r["tc_dt"] for r in kb_scene) + sum(r["tc_dt"] for r in c_scene)) / (len(kb_scene) + len(c_scene))
    ratio = tc_lat / p1_lat if p1_lat > 0 else float("inf")
    print(f"判定相关延迟(该走KB+C): 阶段一 {p1_lat:.2f}s | toolcall {tc_lat:.2f}s | {ratio:.1f}x")

    if tc_bc >= p1_bc and ratio <= 2.0:
        print("[结论] tool calling 边界带正确率>=阶段一 且 延迟<=2倍 -> "
              "建议替换，但替换前停下等用户确认")
    elif tc_bc < p1_bc or ratio > 2.0:
        print("[结论] tool calling 边界带正确率低于阶段一 或 延迟>2倍 -> "
              "保留阶段一，tool calling 记'能力可用但不划算'，L3 停阶段一")
    else:
        print("[结论] 打平 -> 保留阶段一，tool calling 列 L4 候选")


def cat_desc(cat):
    return {"A": "高置信KB", "B": "低置信KB", "C": "灰区实时", "D": "无答案"}[cat]


if __name__ == "__main__":
    main()
