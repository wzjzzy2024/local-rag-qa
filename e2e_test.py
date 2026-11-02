# -*- coding: utf-8 -*-
"""真实问答链路验证: 默认 tavily(无key) -> 自动降级 bing -> 完整 ask() 流程"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from rag_chat import ask

for q in ["2026年诺贝尔物理学奖得主是谁？", "2026冬奥会的举办地在哪里？"]:
    print("\n\n###### 测试:", q, " ######")
    ask(q)
