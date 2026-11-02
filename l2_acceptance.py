# -*- coding: utf-8 -*-
"""
L2 验收:
1. 用 3 条问题跑完整 ask()，观察 L2 上下文结构（title/url/published_at/正文片段、去重、截断）
2. 打印实际喂给模型的联网上下文（L2 打印逻辑已在 rag_chat.py）
3. groundedness 检查: 记录答案中的具体断言，人工核对是否能在正文找到依据
"""
import sys
import re

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from rag_chat import ask, CONTEXT_BUDGET_TOKENS

QS = [
    "2026年诺贝尔物理学奖得主是谁？",
    "2026世界杯冠军是谁？",
    "华为mate70的发布时间是什么时候？",
]

for q in QS:
    print("\n\n" + "=" * 70)
    print("###### 问题:", q, " ######")
    ask(q)

print("\n\n===== L2 配置 =====\n"
      f"CONTEXT_BUDGET_TOKENS = {CONTEXT_BUDGET_TOKENS} (默认 13000 = 40% of 32k)\n"
      "clean_content: 去 HTML 残留/导航噪声/markdown 代码块/空白\n"
      "truncate_by_paragraphs: 按段落截断，不从句子中间切\n"
      "build_context: title + url + published_at + 清洗后正文，URL 去重")
