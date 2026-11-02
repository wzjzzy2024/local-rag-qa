# -*- coding: utf-8 -*-
"""重估: 带约束完整并存对比 (28 条, 每类 7 条)

关键修正:
  1. tool calling 带约束 "只输出 tool call JSON 或 'NO_TOOL', 不输出任何解释性文本"
  2. 延迟同口径:
     - 该走 KB (A+B): 阶段一 = kb_sufficient判定 + 知识库生成 两次调用;
                       tool calling = NO_TOOL/JSON判定 + 知识库生成 两次调用
                       (生成基线对两边共用: 各问题测一次普通生成耗时)
     - 该走 WEB (D): 同口径, 都只算判定到发出搜索请求为止
                     (阶段一 = 0 纯阈值直连; tool calling = TOOL_CALL JSON 耗时)
     - 判定不足 (C): 同口径, 都只算判定 (阶段一 kb_sufficient; tool calling TOOL_CALL JSON)
  3. 判定理由: B/C 打印双方原始输出
停止条件(重新写死):
  - 边界带(B+C)正确率 tc >= p1, 且三场景延迟均值均 <= 阶段一同口径 2 倍,
    且 A 类误判未恶化 -> 建议替换(停下等确认)
  - 否则 -> 保留阶段一, 结论维持"停用"
  - 边界情况: 边界带正确率更高但延迟略超 2 倍 (2.1-3x) -> 列数据停下让用户定夺
不改 kb_sufficient, 不替换任何东西, 只做对比。
"""
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import faiss
from langchain_core.messages import SystemMessage, HumanMessage
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

TC_CONSTRAINT = (
    "你是路由决策助手。判断当前问题是否需要联网搜索。\n"
    "规则: 1) 需要搜索时, 只输出 tool call JSON, 不要输出任何解释性文本; "
    "2) 不需要搜索(能用自身知识直接回答)时, 只输出 NO_TOOL 四个字母, "
    "不要输出任何解释性文本、答案或理由。"
)

# (问题, 类别, 预期路由)  —— 28 条, 每类 7 条
CASES = [
    # A 类: 知识库内高置信 (>0.75)
    ("什么是进程和线程？",            "A", "KB"),
    ("TCP三次握手的过程是什么？",      "A", "KB"),
    ("什么是HTTP协议？",            "A", "KB"),
    ("什么是虚拟内存的分页机制？",     "A", "KB"),
    ("什么是二叉搜索树？",           "A", "KB"),
    ("二叉树的遍历方式有哪些？",        "A", "KB"),
    ("什么是数据库事务？",           "A", "KB"),
    # B 类: 知识库内低置信边界带 (0.55-0.68)
    ("什么是pv操作？",              "B", "KB"),
    ("什么是B树？",                "B", "KB"),
    ("什么是编译原理中的词法分析？",    "B", "KB"),
    ("什么是哈希表？",              "B", "KB"),
    ("什么是页表？",                "B", "KB"),
    ("什么是LRU缓存？",            "B", "KB"),
    ("什么是零拷贝？",              "B", "KB"),
    # C 类: 灰区实时 (0.55-0.65, 内容答不了 -> 回退联网)
    ("2026年最新发布的开源大模型是什么？", "C", "WEB"),
    ("2026年AI领域有哪些最新进展？",   "C", "WEB"),
    ("最新版PyTorch的版本号是多少？",  "C", "WEB"),
    ("2026年10月AI大模型最新发布了哪些？", "C", "WEB"),
    ("2026年最新款的游戏显卡是什么？",  "C", "WEB"),
    ("苹果最新发布的手机是什么型号？",   "C", "WEB"),
    ("最新发布的iPhone是什么型号？",   "C", "WEB"),
    # D 类: 知识库完全无答案 (<0.55)
    ("2026年世界杯冠军是谁？",        "D", "WEB"),
    ("2026年NBA总冠军是哪支球队？",   "D", "WEB"),
    ("2026年苹果秋季发布会的日期？",   "D", "WEB"),
    ("2026年冬奥会的举办城市是哪里？", "D", "WEB"),
    ("华为mate70的发布时间是什么时候？", "D", "WEB"),
    ("2026年诺贝尔物理学奖得主是谁？", "D", "WEB"),
    ("2026年美国总统是谁？",         "D", "WEB"),
]

KB_SUFF_PROMPT = (
    "判断下面的知识库片段是否足以回答用户的问题。\n"
    "问题: {q}\n知识库片段:\n{snippets}\n"
    "如果片段包含答案（或能据此给出可靠答案），回复: 足以回答\n"
    "如果片段只是相关主题背景、缺少回答所需的关键信息"
    "（尤其是实时/最新信息），回复: 不足以回答\n"
    "只输出\"足以回答\"或\"不足以回答\"。"
)

GEN_PROMPT = "请直接回答下面的问题，给出简洁、准确的答案。\n问题: {q}\n答案:"


def judge_phase1(llm, question, texts, sel):
    """阶段一判定(复制 kb_sufficient 逻辑, 捕获原始回复)"""
    snippets = "\n".join(f"- {texts[i][:150]}" for i, _s in sel[:rc.TOP_K])
    p = KB_SUFF_PROMPT.format(q=question, snippets=snippets)
    t0 = time.time()
    resp = llm.invoke(p)
    dt = time.time() - t0
    ans = resp.content.strip()
    suff = "足以回答" in ans and "不足以回答" not in ans
    return suff, dt, ans


def judge_toolcall(llm_tools, question):
    """带约束 tool calling 判定"""
    msgs = [SystemMessage(content=TC_CONSTRAINT), HumanMessage(content=question)]
    t0 = time.time()
    resp = llm_tools.invoke(msgs)
    dt = time.time() - t0
    content = getattr(resp, "content", "") or ""
    tool_calls = getattr(resp, "tool_calls", None) or []
    called = bool(tool_calls)
    if called:
        names = [tc["name"] for tc in tool_calls]
        args_list = [tc.get("args", {}) for tc in tool_calls]
        reason = f"TOOL_CALL {names} args={args_list}"
    else:
        reason = f"NO_TOOL content_len={len(content)} 首100字:{content[:100]!r}"
    return called, dt, reason


def measure_gen(llm, question):
    """知识库生成基线: 一次普通生成耗时 (两边共用)"""
    p = GEN_PROMPT.format(q=question)
    t0 = time.time()
    llm.invoke(p)
    return time.time() - t0


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

        # tool calling (带约束)
        tc_called, tc_dt, tc_reason = judge_toolcall(llm_tools, q)
        tc_route = "WEB" if tc_called else "KB"

        # 生成基线 (仅 A/B 类需要, 因为 KB 场景要算判定+生成)
        gen_dt = 0.0
        if cat in ("A", "B"):
            gen_dt = measure_gen(llm, q)

        rows.append(dict(q=q, cat=cat, expect=expect, best=best,
                         p1=p1_route, p1_dt=p1_dt, p1_reason=p1_reason,
                         tc=tc_route, tc_dt=tc_dt, tc_reason=tc_reason,
                         gen_dt=gen_dt))

    # ---- 打印明细 ----
    print(f"===== 重估: 带约束并存对比 =====  样本 {len(rows)} 条, "
          f"CHECK_MAX={rc.SECONDARY_ROUTE_CHECK_MAX}")
    print(f"{'问题':<34}{'类':<3}{'best':<7}{'预期':<5}{'阶段一':<6}{'判定耗时':<9}"
          f"{'toolcall':<9}{'TC耗时':<8}{'一致':<5}")
    print("-" * 104)
    for r in rows:
        same = "✓" if r["p1"] == r["tc"] else "✗"
        print(f"{r['q']:<34}{r['cat']:<3}{r['best']:<7.3f}{r['expect']:<5}"
              f"{r['p1']:<6}{r['p1_dt']:<9.2f}{r['tc']:<9}{r['tc_dt']:<8.2f}{same}")

    # ---- 1. 四类分项正确率 ----
    print("\n===== 1. 四类分项路由正确率 =====")
    print(f"{'类别':<6}{'说明':<26}{'阶段一':<10}{'toolcall':<12}{'A类误判恶化':<12}")
    cat_ok = {}
    for cat in ["A", "B", "C", "D"]:
        sub = [r for r in rows if r["cat"] == cat]
        p1_ok = sum(1 for r in sub if r["p1"] == r["expect"])
        tc_ok = sum(1 for r in sub if r["tc"] == r["expect"])
        cat_ok[cat] = (p1_ok, tc_ok)
        worse = "—"
        if cat == "A":
            worse = "是(恶化)" if tc_ok < p1_ok else "否(未恶化)"
        print(f"{cat:<6}{cat_desc(cat):<26}{p1_ok}/{len(sub):<10}"
              f"{tc_ok}/{len(sub):<12}{worse}")

    # ---- 2. 边界带 (B+C) ----
    bc = [r for r in rows if r["cat"] in ("B", "C")]
    p1_bc = sum(1 for r in bc if r["p1"] == r["expect"])
    tc_bc = sum(1 for r in bc if r["tc"] == r["expect"])
    print(f"\n===== 2. 边界带(B+C)正确率 =====  (阶段一 {p1_bc}/{len(bc)} = "
          f"{p1_bc/len(bc)*100:.0f}% | toolcall {tc_bc}/{len(bc)} = "
          f"{tc_bc/len(bc)*100:.0f}%)")

    # ---- 3. 同口径延迟三场景 ----
    print("\n===== 3. 同口径延迟三场景 =====")
    # 该走 KB (A+B): 判定 + 生成两次调用, 生成基线共用
    kb_sub = [r for r in rows if r["cat"] in ("A", "B")]
    p1_kb = sum(r["p1_dt"] + r["gen_dt"] for r in kb_sub) / len(kb_sub)
    tc_kb = sum(r["tc_dt"] + r["gen_dt"] for r in kb_sub) / len(kb_sub)
    # 该走 WEB (D): 判定到发出搜索请求为止
    d_sub = [r for r in rows if r["cat"] == "D"]
    p1_d = sum(r["p1_dt"] for r in d_sub) / len(d_sub)
    tc_d = sum(r["tc_dt"] for r in d_sub) / len(d_sub)
    # 判定不足 (C): 只算判定
    c_sub = [r for r in rows if r["cat"] == "C"]
    p1_c = sum(r["p1_dt"] for r in c_sub) / len(c_sub)
    tc_c = sum(r["tc_dt"] for r in c_sub) / len(c_sub)

    print(f"{'场景':<18}{'阶段一(同口径)':<18}{'toolcall(同口径)':<18}{'比值':<8}")
    for name, p1, tc in [("该走KB(A+B)", p1_kb, tc_kb),
                         ("该走WEB(D)", p1_d, tc_d),
                         ("判定不足(C)", p1_c, tc_c)]:
        ratio = tc / p1 if p1 > 0 else float("inf")
        print(f"{name:<18}{p1:<18.2f}{tc:<18.2f}{ratio:<8.2f}x")
    print(f"  (注: KB 场景含生成基线; 生成耗时两方案相同, 差值即判定层差异)")
    print(f"  (注: 生成基线均值 {sum(r['gen_dt'] for r in kb_sub)/len(kb_sub):.2f}s)")

    # ---- 4. 判定理由 (B/C 全部) ----
    print("\n===== 4. 判定理由人工比对 (B/C 类) =====")
    for r in rows:
        if r["cat"] not in ("B", "C"):
            continue
        print(f"\n[{r['cat']}] {r['q']} (best={r['best']:.3f}, 预期 {r['expect']})")
        print(f"  阶段一  : {r['p1_reason']}")
        print(f"  toolcall: {r['tc_reason']}")

    # ---- 5. 结论 (停止条件) ----
    print("\n===== 5. 结论(按重新写死停止条件) =====")
    ratios = {"KB": tc_kb / p1_kb if p1_kb > 0 else float("inf"),
              "WEB": tc_d / p1_d if p1_d > 0 else float("inf"),
              "不足": tc_c / p1_c if p1_c > 0 else float("inf")}
    all_le2 = all(v <= 2.0 for v in ratios.values())
    a_p1, a_tc = cat_ok["A"]
    a_not_worse = a_tc >= a_p1
    print(f"边界带: 阶段一 {p1_bc}/{len(bc)} ({p1_bc/len(bc)*100:.0f}%) | "
          f"toolcall {tc_bc}/{len(bc)} ({tc_bc/len(bc)*100:.0f}%)")
    print(f"三场景延迟比值: KB {ratios['KB']:.2f}x / WEB {ratios['WEB']:.2f}x "
          f"/ 不足 {ratios['不足']:.2f}x (要求均<=2x)")
    print(f"A 类误判恶化: {'否' if a_not_worse else '是'} (阶段一 {a_p1}/5 vs "
          f"toolcall {a_tc}/5)")
    max_ratio = max(ratios.values())
    if tc_bc >= p1_bc and all_le2 and a_not_worse:
        print("[结论] 建议替换 tool calling——边界带正确率>=阶段一, 延迟同口径<=2x, "
              "A类未恶化 -> 但替换前停下等用户确认")
    elif tc_bc >= p1_bc and 2.0 < max_ratio <= 3.0:
        print(f"[结论] 边界情况: 边界带正确率更高({tc_bc}/{len(bc)} vs {p1_bc}/{len(bc)}) "
              f"但延迟略超2倍(最大 {max_ratio:.2f}x) -> 列出数据停下等用户定夺")
    else:
        print("[结论] 保留阶段一, tool calling 停用(当前设计下), README 结论维持'停用'")


def cat_desc(cat):
    return {"A": "高置信KB", "B": "低置信KB", "C": "灰区实时", "D": "无答案"}[cat]


if __name__ == "__main__":
    main()
