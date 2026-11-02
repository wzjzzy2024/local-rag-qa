# -*- coding: utf-8 -*-
"""L3 阶段一验收: 二级判定路由
- 灰色地带问题(0.55-0.68): 知识库命中但答不了实时信息 -> 应回退联网
- 知识库内问题: 不应误回退（强制判定模式测误杀率）
- 统计 L2(纯阈值) vs L3(二级判定) 同组问题路由正确率
- 最后跑 1 条完整 ask() 验证回退链路端到端可用
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import faiss
from langchain_ollama import OllamaEmbeddings, ChatOllama

import rag_chat as rc

# ---- 加载索引一次，复用 ----
print("[L3] 加载索引与模型...")
emb = OllamaEmbeddings(model=rc.EMBED_MODEL, base_url=rc.OLLAMA_BASE)
index, texts, metas = rc.load_index()
llm = ChatOllama(model=rc.LLM_MODEL, base_url=rc.OLLAMA_BASE, temperature=0.2)

def retrieve(question):
    qv = np.array(emb.embed_query(question), dtype="float32").reshape(1, -1)
    faiss.normalize_L2(qv)
    scores, idx = index.search(qv, rc.TOP_K)
    scores, idx = scores[0], idx[0]
    best = float(scores[0]) if len(scores) and scores[0] > -9e9 else 0.0
    sel = [(int(i), float(s)) for i, s in zip(idx, scores) if int(i) >= 0 and s > 0]
    return best, sel

# (问题, 期望路由, 类别)
CASES = [
    ("2026年10月AI大模型最新发布",           "web", "灰区-实时"),
    ("DeepSeek 最新的模型是什么版本",         "web", "灰区-实时"),
    ("苹果最新的手机型号是什么",              "web", "灰区-实时"),
    ("2026年诺贝尔物理学奖得主是谁",          "web", "对照组低置信"),
    ("什么是递归？",                          "kb",  "知识库内"),
    ("什么是pv操作？",                        "kb",  "知识库内"),
    ("进程和线程的区别是什么？",              "kb",  "知识库内"),
    ("TCP三次握手的过程是什么？",             "kb",  "知识库内"),
]

print(f"\n===== L3 阶段一验收 =====  (CHECK_MAX={rc.SECONDARY_ROUTE_CHECK_MAX}, "
      f"ENABLED={rc.SECONDARY_ROUTE_ENABLED})")
print(f"{'问题':<32}{'best':<7}{'L2路由':<16}{'L3判定':<12}{'L3路由':<16}{'期望':<6}{'L2✓':<5}{'L3✓':<5}")
print("-" * 110)

stats = {"correct_l2": 0, "correct_l3": 0, "total": len(CASES),
         "corrected": 0, "misjudged_kb": 0}
for q, expect, cat in CASES:
    best, sel = retrieve(q)
    l2_route = "knowledge_base" if best >= rc.SCORE_THRESHOLD else "web_search"

    # L3: 默认配置下的路由
    if best < rc.SCORE_THRESHOLD:
        l3_route, judge = "web_search", "-"
    elif best >= rc.SECONDARY_ROUTE_CHECK_MAX:
        l3_route, judge = "knowledge_base", "-高置信跳过"
    else:
        t0 = time.time()
        suff = rc.kb_sufficient(llm, q, texts, sel)
        judge = "足以回答" if suff else "不足以回答"
        l3_route = "knowledge_base" if suff else "web_search(回退)"
        print(f"  [判定耗时 {time.time()-t0:.1f}s] {q} -> {judge}")

    ok2 = (l2_route == "knowledge_base") == (expect == "kb")
    ok3 = (l3_route == "knowledge_base") == (expect == "kb")
    stats["correct_l2"] += ok2
    stats["correct_l3"] += ok3
    if cat == "灰区-实时" and ok3 and not ok2:
        stats["corrected"] += 1
    print(f"{q:<32}{best:<7.3f}{l2_route:<16}{judge:<12}{l3_route:<16}"
          f"{expect:<6}{'✓' if ok2 else '✗':<5}{'✓' if ok3 else '✗':<5}")

print("\n===== 知识库内问题误杀率（强制判定，模拟 CHECK_MAX=∞）=====")
mis = 0
for q, expect, cat in CASES:
    if cat != "知识库内":
        continue
    best, sel = retrieve(q)
    suff = rc.kb_sufficient(llm, q, texts, sel)
    if not suff:
        mis += 1
        print(f"  [误判] {q} (best={best:.3f}) 判定为不足以回答")
print(f"知识库内 {4} 条中误回退 {mis} 条 -> 误杀率 {mis}/4")

print("\n===== 汇总 =====")
print(f"总案例 {stats['total']} 条 | L2 路由正确 {stats['correct_l2']} 条 "
      f"({stats['correct_l2']/stats['total']*100:.0f}%) | "
      f"L3 路由正确 {stats['correct_l3']} 条 "
      f"({stats['correct_l3']/stats['total']*100:.0f}%)")
print(f"灰区问题被 L3 纠正(应联网而纯阈值走知识库) {stats['corrected']} 条")

# ---- 端到端: 1 条灰区问题完整跑 ask() ----
print("\n===== 端到端: 灰区问题完整 ask() =====")
rc.ask("2026年10月AI大模型最新发布")
