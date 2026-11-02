# -*- coding: utf-8 -*-
"""L3 阶段二前置: qwen2.5:7b-instruct-q4_K_M 原生 tool calling 稳定性验证

只做验证，不做任何架构改动、不接 ReAct 循环。

方法:
  - 构造 1 个最小工具 search(query, top_k)（schema 简单）
  - 10 个不同问题（5 个该调工具 = 实时/外部信息; 5 个不该调 = 知识库/通用知识）
  - 让模型用 bind_tools 决定"调不调、调哪个、参数是什么"
  - 统计: 结构化输出成功率 / 参数正确率 / 格式崩坏数 / 幻觉工具名数 / 参数缺失数

判定规则（写死）:
  - 成功 = 该调则调且工具名==search 且 query 非空; 不该调则返回纯文本不调用
  - 参数正确 = query 为非空字符串; top_k 缺失时给默认值视为可接受(记入备注)
  - 决策规则: 成功率 >= 8/10 且参数正确率 >= 8/10 -> 可做最小 tool calling 原型(并存对比)
             否则 -> 不做阶段二, L3 停在阶段一, 本数据记入 README 作为"为什么不继续"依据
"""
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from langchain_ollama import ChatOllama

LLM_MODEL = "qwen2.5:7b-instruct-q4_K_M"
OLLAMA_BASE = "http://localhost:11434"

# ---- 最小工具定义（schema 简单） ----
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

# (问题, 期望: 1=该调工具, 0=不该调)
CASES = [
    ("2026年世界杯冠军是谁？", 1),
    ("今天北京的天气怎么样？", 1),
    ("华为mate70的发布时间是什么时候？", 1),
    ("2026年诺贝尔物理学奖得主是谁？", 1),
    ("最新版 PyTorch 的版本号是多少？", 1),
    ("什么是递归？", 0),
    ("Python 中如何读取文件？", 0),
    ("1+1等于几？", 0),
    ("解释什么是信号量", 0),
    ("进程和线程的区别是什么？", 0),
]


def main():
    llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_BASE, temperature=0.2)
    llm_tools = llm.bind_tools(TOOLS)

    n_success = 0          # 行为符合预期（该调则调+工具名对 / 不该调则不调）
    n_param_ok = 0         # 触发调用时 query 非空且为 str
    n_malformed = 0        # 输出无法解析 / 抛异常 / 空
    n_hallucinated = 0     # 调用不存在的工具名
    n_param_missing = 0    # 调用 search 但 query 缺失/空
    details = []

    for q, expect in CASES:
        t0 = time.time()
        try:
            resp = llm_tools.invoke(q)
            raw = getattr(resp, "content", "") or ""
            tool_calls = getattr(resp, "tool_calls", None) or []
            called = bool(tool_calls)
            names = [tc["name"] for tc in tool_calls]
            args_list = [tc.get("args", {}) for tc in tool_calls]
            dt = time.time() - t0

            # 判定
            if called and names == ["search"] and expect == 1:
                args = args_list[0]
                query = args.get("query")
                if isinstance(query, str) and query.strip():
                    n_success += 1
                    n_param_ok += 1
                    status = "OK-该调则调"
                else:
                    n_success += 0
                    n_param_missing += 1
                    status = "FAIL-query缺失/空"
            elif called and names == ["search"] and expect == 0:
                n_success += 0
                status = "FAIL-不该调却调"
            elif called and expect == 1:
                # 调了但工具名不对
                n_hallucinated += 1
                n_success += 0
                status = f"FAIL-幻觉工具名{names}"
            elif not called and expect == 0:
                n_success += 1
                n_param_ok += 1
                status = "OK-不该调则不调"
            elif not called and expect == 1:
                n_success += 0
                status = "FAIL-该调却没调"

            details.append((q, expect, called, names, args_list,
                            f"{dt:.1f}s", status, raw[:120]))
        except Exception as e:
            n_malformed += 1
            dt = time.time() - t0
            details.append((q, expect, "EXC", [], [],
                            f"{dt:.1f}s", f"FAIL-异常:{e}", ""))

    # ---- 打印明细 ----
    print(f"===== tool calling 稳定性验证 =====  model={LLM_MODEL}")
    print(f"{'问题':<30}{'期望':<6}{'调用':<6}{'工具':<10}{'参数':<40}{'耗时':<6}{'判定':<20}")
    print("-" * 120)
    for q, expect, called, names, args, dt, status, raw in details:
        arg_str = str(args)[:40] if args else "-"
        print(f"{q:<30}{'调' if expect else '不调':<6}"
              f"{'是' if called else '否':<6}{str(names):<10}{arg_str:<40}{dt:<6}{status:<20}")
        if status.startswith("FAIL"):
            print(f"    raw: {raw}")

    # ---- 汇总 ----
    print("\n===== 汇总 =====")
    print(f"结构化输出成功率(行为符合预期): {n_success}/10 = {n_success/10*100:.0f}%")
    print(f"参数正确率(query非空):          {n_param_ok}/10 = {n_param_ok/10*100:.0f}%")
    print(f"格式崩坏:                       {n_malformed} 次")
    print(f"幻觉工具名:                     {n_hallucinated} 次")
    print(f"query 参数缺失/空:              {n_param_missing} 次")

    # ---- 决策规则（写死） ----
    if n_success >= 8 and n_param_ok >= 8:
        print("\n[决策] 成功率与参数正确率均 >= 8/10 -> 可做最小 tool calling 原型"
              "（先并存对比，不替换阶段一 kb_sufficient）")
    else:
        print("\n[决策] 成功率或参数正确率 < 8/10 -> 不做阶段二，L3 停在阶段一；"
              "稳定性数据作为'为什么不继续'依据记入 README")
    return n_success, n_param_ok


if __name__ == "__main__":
    main()
