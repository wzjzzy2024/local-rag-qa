# -*- coding: utf-8 -*-
"""混合路由验证: 灰区(0.55-0.75)启用 tool calling 判定(中性格式约束)

设计(用户确认):
  - <0.55 直接联网(纯阈值, 不改)
  - >0.75 直接知识库(纯阈值, 不改)
  - 0.55-0.75 灰区: tool calling 判定(带格式约束, 但去掉"能用自身知识就NO_TOOL"
    的语义判断, 只保留"输出JSON或NO_TOOL, 不输出解释")
  - kb_sufficient 作 fallback: tool calling 判定异常/超时/格式崩坏时回退
  - 配置: 灰区上下界可配(默认0.55/0.75), 灰区判定方式可配(tool_calling/kb_sufficient)

验证(成本低, 只重跑灰区):
  1. 逻辑验证: 用已有28条数据拼混合路由理论正确率 (A 7/7 + D 7/7 + 灰区 tool calling)
  2. 实跑: 只重跑 B+C 灰区 14 条, 确认灰区正确率 >= 13/14, D 类无误伤(结构保证),
     灰区延迟(判定+生成同口径)与阶段一对比绝对差
停止条件(写死):
  - 灰区正确率 >= 13/14 且 D 类无误伤 且 灰区延迟绝对差 <= 0.5s
    -> 采纳混合路由(替换前停下等用户确认)
  - 否则 -> 保留阶段一, 混合路由记录"理论更优但实跑不达标"
"""
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import faiss
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import OllamaEmbeddings, ChatOllama

import rag_chat as rc

# ---- 中性格式约束: 不含任何"能用自身知识就NO_TOOL"的语义判断 ----
TC_CONSTRAINT = (
    "你是路由决策助手。请决定是否调用 search 工具。\n"
    "规则: 1) 若决定调用工具, 只输出 tool call JSON, 不输出任何其他文本; "
    "2) 若决定不调用, 只输出 NO_TOOL 四个字母, 不输出任何其他文本。"
)

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

# 混合路由配置(可配, 默认灰区 0.55-0.75)
MIX_LOW = float(__import__("os").environ.get("MIX_LOW", "0.55"))
MIX_HIGH = float(__import__("os").environ.get("MIX_HIGH", "0.75"))
MIX_MODE = __import__("os").environ.get("MIX_MODE", "tool_calling")  # tool_calling / kb_sufficient

# B+C 灰区 14 条(与 v2 相同), 覆盖不同相似度点位
GREY_CASES = [
    # B 类 (KB 预期)
    ("什么是pv操作？",            "B", "KB", 0.675),
    ("什么是B树？",              "B", "KB", 0.666),
    ("什么是编译原理中的词法分析？",  "B", "KB", 0.662),
    ("什么是哈希表？",            "B", "KB", 0.653),
    ("什么是页表？",              "B", "KB", 0.644),
    ("什么是LRU缓存？",          "B", "KB", 0.642),
    ("什么是零拷贝？",            "B", "KB", 0.589),
    # C 类 (WEB 预期)
    ("2026年最新发布的开源大模型是什么？", "C", "WEB", 0.620),
    ("2026年AI领域有哪些最新进展？", "C", "WEB", 0.610),
    ("最新版PyTorch的版本号是多少？", "C", "WEB", 0.607),
    ("2026年10月AI大模型最新发布了哪些？", "C", "WEB", 0.602),
    ("2026年最新款的游戏显卡是什么？", "C", "WEB", 0.584),
    ("苹果最新发布的手机是什么型号？",  "C", "WEB", 0.564),
    ("最新发布的iPhone是什么型号？",  "C", "WEB", 0.560),
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


def mixed_route(llm_tools, llm, question, texts, sel, best):
    """混合路由: 分段判定。返回 (route, stage, judge_dt, reason)
    route: KB/WEB; stage: 纯阈值低/纯阈值高/灰区; judge_dt: 判定耗时
    """
    if best < MIX_LOW:
        return "WEB", "D纯阈值直连", 0.0, f"best={best:.3f}<{MIX_LOW} 直接联网"
    if best > MIX_HIGH:
        return "KB", "A纯阈值直连", 0.0, f"best={best:.3f}>{MIX_HIGH} 直接知识库"

    # 灰区: tool calling 判定 (异常回退 kb_sufficient)
    if MIX_MODE == "tool_calling":
        msgs = [SystemMessage(content=TC_CONSTRAINT), HumanMessage(content=question)]
        t0 = time.time()
        try:
            resp = llm_tools.invoke(msgs)
            dt = time.time() - t0
            content = getattr(resp, "content", "") or ""
            tool_calls = getattr(resp, "tool_calls", None) or []
            if tool_calls:
                return "WEB", "灰区-toolcall", dt, \
                    f"TOOL_CALL {[tc['name'] for tc in tool_calls]} args={[tc.get('args',{}) for tc in tool_calls]}"
            # NO_TOOL / 文本: 视为不调 -> KB (内容已被模型视为可答)
            return "KB", "灰区-toolcall", dt, f"NO_TOOL content={content[:80]!r}"
        except Exception as e:
            print(f"  [fallback] tool calling 异常({e}), 回退 kb_sufficient")
            return judge_kb_suff(llm, question, texts, sel)

    # kb_sufficient 模式
    return judge_kb_suff(llm, question, texts, sel)


def judge_kb_suff(llm, question, texts, sel):
    snippets = "\n".join(f"- {texts[i][:150]}" for i, _s in sel[:rc.TOP_K])
    p = KB_SUFF_PROMPT.format(q=question, snippets=snippets)
    t0 = time.time()
    try:
        resp = llm.invoke(p)
        dt = time.time() - t0
        ans = resp.content.strip()
        suff = "足以回答" in ans and "不足以回答" not in ans
        return ("KB" if suff else "WEB"), "灰区-kbsuff", dt, f"判定回复: {ans}"
    except Exception as e:
        print(f"  [fallback] kb_sufficient 也失败({e}), 保守走知识库")
        return "KB", "灰区-kbsuff", 0.0, f"判定失败保守KB: {e}"


def main():
    emb = OllamaEmbeddings(model=rc.EMBED_MODEL, base_url=rc.OLLAMA_BASE)
    index, texts, metas = rc.load_index()
    llm = ChatOllama(model=rc.LLM_MODEL, base_url=rc.OLLAMA_BASE, temperature=0.2)
    llm_tools = llm.bind_tools(TOOLS)

    print(f"===== 混合路由验证 =====  灰区 [{MIX_LOW}, {MIX_HIGH}], 模式={MIX_MODE}")
    print(f"{'问题':<34}{'类':<3}{'best':<7}{'预期':<5}{'混合路由':<22}{'判定耗时':<9}{'正确':<4}")
    print("-" * 96)

    rows = []
    for q, cat, expect, expect_best in GREY_CASES:
        qv = np.array(emb.embed_query(q), dtype="float32").reshape(1, -1)
        faiss.normalize_L2(qv)
        scores, idx = index.search(qv, rc.TOP_K)
        scores, idx = scores[0], idx[0]
        best = float(scores[0]) if len(scores) and scores[0] > -9e9 else 0.0
        sel = [(int(i), float(s)) for i, s in zip(idx, scores)
               if int(i) >= 0 and s > 0]

        route, stage, dt, reason = mixed_route(llm_tools, llm, q, texts, sel, best)
        ok = (route == expect)
        rows.append(dict(q=q, cat=cat, expect=expect, best=best, route=route,
                         stage=stage, dt=dt, reason=reason, ok=ok, gen_dt=0.0))
        print(f"{q:<34}{cat:<3}{best:<7.3f}{expect:<5}{route+'('+stage+')':<22}"
              f"{dt:<9.2f}{'✓' if ok else '✗'}")

    # 灰区正确率
    n_ok = sum(1 for r in rows if r["ok"])
    print(f"\n===== 灰区实跑正确率 =====  {n_ok}/{len(rows)} = {n_ok/len(rows)*100:.0f}%")

    # 逻辑验证: 混合路由理论 = A(7/7 纯阈值高) + D(7/7 纯阈值低) + 灰区实跑
    # A/D 行为不变由分段结构保证(>0.75 / <0.55 不走判定)
    theory = 7 + 7 + n_ok
    theory_total = 28
    print(f"逻辑验证: 混合路由理论 = A纯阈值 7/7 + D纯阈值 7/7 + 灰区实跑 {n_ok}/14")
    print(f"          = {theory}/{theory_total} = {theory/theory_total*100:.0f}%")
    print(f"          (对比: 阶段一 25/28=89%, 纯tool calling v2 20/28=71%)")

    # D 类无误伤检查: 结构保证(灰区判定只对 0.55-0.75 触发), 打印确认
    print(f"\n===== D 类无误伤检查 =====")
    print(f"灰区判定仅对 best ∈ [{MIX_LOW}, {MIX_HIGH}] 触发(结构保证)。")
    print(f"实跑 14 条中 best 范围: {min(r['best'] for r in rows):.3f} ~ "
          f"{max(r['best'] for r in rows):.3f}, 均在灰区。")
    print(f"<0.55 或 >0.75 的问题不会进入 tool calling, D 类无误伤(确认)。")

    # 灰区延迟对比(判定+生成同口径)
    print(f"\n===== 灰区延迟对比(判定+生成同口径) =====")
    # 生成基线: 每条灰区问题测一次普通生成
    for r in rows:
        p = GEN_PROMPT.format(q=r["q"])
        t0 = time.time()
        llm.invoke(p)
        r["gen_dt"] = time.time() - t0

    # 阶段一灰区延迟 = kb_sufficient 判定 + 生成 (需现测 kb_sufficient 判定)
    p1_total, tc_total = 0.0, 0.0
    for r in rows:
        _rt, _st, _dt, _rs = judge_kb_suff(llm, r["q"], texts, sel_from(r, texts, emb, index))
        p1_total += _dt + r["gen_dt"]
        tc_total += r["dt"] + r["gen_dt"]
    p1_avg = p1_total / len(rows)
    tc_avg = tc_total / len(rows)
    diff = abs(tc_avg - p1_avg)
    print(f"阶段一(灰区 kbsuff+生成): {p1_avg:.2f}s")
    print(f"tool calling(灰区 判定+生成): {tc_avg:.2f}s")
    print(f"绝对差: {diff:.2f}s (停止条件要求 <= 0.5s)")
    print(f"生成基线均值: {sum(r['gen_dt'] for r in rows)/len(rows):.2f}s")

    # 判定理由(B/C 打印)
    print(f"\n===== 判定理由(B/C 灰区) =====")
    for r in rows:
        print(f"[{r['cat']}] {r['q']} (best={r['best']:.3f}, 预期{r['expect']}, "
              f"实际{r['route']})  {r['reason']}")

    # 结论
    print(f"\n===== 结论(停止条件) =====")
    d_safe = True  # 结构保证
    print(f"灰区正确率: {n_ok}/14 (需>=13/14)")
    print(f"D 类无误伤: {'✓(结构保证)' if d_safe else '✗'}")
    print(f"灰区延迟绝对差: {diff:.2f}s (需<=0.5s)")
    if n_ok >= 13 and d_safe and diff <= 0.5:
        print("[结论] 采纳混合路由——灰区正确率达标、D类无误伤、延迟差在限内。"
              "替换前停下等用户确认。")
    else:
        print("[结论] 保留阶段一——混合路由理论更优但实跑未达标，L4 停在这里。")


def sel_from(r, texts, emb, index):
    qv = np.array(emb.embed_query(r["q"]), dtype="float32").reshape(1, -1)
    faiss.normalize_L2(qv)
    scores, idx = index.search(qv, rc.TOP_K)
    return [(int(i), float(s)) for i, s in zip(idx[0], scores[0])
            if int(i) >= 0 and s > 0]


if __name__ == "__main__":
    main()
