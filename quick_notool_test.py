# -*- coding: utf-8 -*-
"""低成本验证: 给 tool calling 加输出约束, 重测"不该调"场景延迟

约束: "只输出 tool call JSON 或 'NO_TOOL'，不输出任何解释性文本"
目标: 验证 4.7s 延迟根因(生成完整解释文本)是否可修复。
结果处理(一次就够):
  - 不该调延迟 <= 1s  -> 根因确认可修复, L4 阶段二重估优先级上调
  - 仍 > 2s           -> 记录实测数字, 不改结论
测试对象: B 类 5 条 (对比中全部被判 NO_CALL 且生成 278-450 字解释)
"""
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama

LLM_MODEL = "qwen2.5:7b-instruct-q4_K_M"
OLLAMA_BASE = "http://localhost:11434"

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

CONSTRAINT = (
    "你是路由决策助手。判断当前问题是否需要联网搜索。\n"
    "规则: 1) 需要搜索时, 只输出 tool call JSON, 不要输出任何解释性文本; "
    "2) 不需要搜索(能用自身知识直接回答)时, 只输出 NO_TOOL 四个字母, "
    "不要输出任何解释性文本、答案或理由。"
)

# B 类 5 条: 对比实测全部 NO_CALL 且生成整篇解释
CASES = [
    "什么是pv操作？",
    "什么是编译原理中的词法分析？",
    "什么是哈希表？",
    "什么是页表？",
    "什么是零拷贝？",
]


def main():
    llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_BASE, temperature=0.2)
    llm_tools = llm.bind_tools(TOOLS)

    print(f"===== 约束后 '不该调' 延迟验证 =====  model={LLM_MODEL}")
    print(f"约束: {CONSTRAINT}\n")
    times = []
    for q in CASES:
        msgs = [SystemMessage(content=CONSTRAINT), HumanMessage(content=q)]
        t0 = time.time()
        resp = llm_tools.invoke(msgs)
        dt = time.time() - t0
        times.append(dt)
        content = getattr(resp, "content", "") or ""
        tool_calls = getattr(resp, "tool_calls", None) or []
        called = bool(tool_calls)
        kind = "TOOL_CALL" if called else "NO_TOOL/文本"
        print(f"{q:<32}{dt:<6.2f}s  {kind}  "
              f"content_len={len(content)}  首60字:{content[:60]!r}")

    avg = sum(times) / len(times)
    print(f"\n均值: {avg:.2f}s  (约束前 B 类 toolcall 均值 5.25s)")
    if avg <= 1.0:
        print("[结果] <= 1s -> 根因确认可修复: 延迟主要来自'生成完整解释文本'的生成行为,"
              " prompt 约束可压至可接受范围; L4 阶段二重估优先级上调")
    elif avg <= 2.0:
        print("[结果] 1-2s -> 明显改善但仍偏高, 记录数字, 延迟优化方向部分有效")
    else:
        print("[结果] > 2s -> 存在其他因素, 记录实测数字, 不改结论")


if __name__ == "__main__":
    main()
