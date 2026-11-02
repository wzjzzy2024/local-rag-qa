# -*- coding: utf-8 -*-
"""验证2(替换后必做): fallback 路径主动触发测试。

主路径 14/14 只证明 tool calling 正常时能接住, 本脚本 mock 三种故障:
  故障1: tool calling 返回格式崩坏(非法 JSON content, 无结构化 tool_calls)
  故障2: tool calling 超时(invoke 抛 TimeoutError)
  故障3: tool calling 返回第三种输出(既非 TOOL_CALL 也非 NO_TOOL)

每种故障下验证:
  a. 正确回退到 kb_sufficient, 且 kb_sufficient 判定结果生效(足以->KB / 不足->WEB)
  b. kb_sufficient 也失败(抛异常)时保守走知识库(不误走 WEB)
  c. 打日志说明降级原因(检查 stdout 含降级说明)
"""
import io
import sys
from contextlib import redirect_stdout
from unittest.mock import patch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import rag_chat as rc


class FakeBound:
    """模拟 bind_tools 后的可调用对象"""
    def __init__(self, resp=None, exc=None):
        self._resp = resp
        self._exc = exc

    def invoke(self, msgs):
        if self._exc:
            raise self._exc
        return self._resp


class FakeChatOllama:
    """模拟 ChatOllama: bind_tools 返回 FakeBound"""
    def __init__(self, resp=None, exc=None, **kw):
        self._resp = resp
        self._exc = exc

    def bind_tools(self, tools):
        return FakeBound(resp=self._resp, exc=self._exc)


class FakeResp:
    def __init__(self, content="", tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


# 三种故障的 FakeChatOllama 工厂
FAULTS = {
    "故障1_格式崩坏": FakeChatOllama(resp=FakeResp(content='{"query": "abc"')),
    "故障2_超时":     FakeChatOllama(exc=TimeoutError("Ollama request timeout")),
    "故障3_第三种输出": FakeChatOllama(resp=FakeResp(content="我无法决定要不要搜索")),
}


def run_case(fault_name, fake_llm, kb_behavior):
    """执行一次灰区判定, 返回 (mode, route_detail, captured_log)
    kb_behavior: "enough"(kb 返回 True) / "not_enough"(kb 返回 False) / "fail"(kb 抛异常)
    """
    def fake_kb_sufficient(llm, question, texts, sel):
        if kb_behavior == "enough":
            return True
        if kb_behavior == "not_enough":
            return False
        raise RuntimeError("kb_sufficient LLM 调用失败")

    buf = io.StringIO()
    with patch.object(rc, "ChatOllama", lambda **kw: fake_llm), \
         patch.object(rc, "kb_sufficient", fake_kb_sufficient), \
         redirect_stdout(buf):
        mode, route_detail = rc._grey_route_judge(None, "什么是pv操作？", [], [])
    return mode, route_detail, buf.getvalue()


def main():
    print("===== 验证2: fallback 路径主动触发测试 =====")
    print(f"MIX_MODE={rc.MIX_MODE} (须为 tool_calling 才有主路径可故障)")
    if rc.MIX_MODE != "tool_calling":
        print("!! 当前 MIX_MODE 不是 tool_calling, 测试无意义, 停止")
        sys.exit(1)

    n_pass = 0
    total = 0
    for fault_name, fake_llm in FAULTS.items():
        print(f"\n-------- {fault_name} --------")
        # a. kb_sufficient 正常: 足以 -> KB
        mode, detail, log = run_case(fault_name, fake_llm, "enough")
        total += 1
        p1 = (mode == "knowledge_base") and ("kb_sufficient→KB" in detail)
        print(f"  a1. kb 足以回答   -> {mode} ({detail})  "
              f"{'✓' if p1 else '✗ 应回退kb_sufficient且判为KB'}")
        n_pass += int(p1)
        # a. kb_sufficient 正常: 不足 -> WEB
        mode, detail, log = run_case(fault_name, fake_llm, "not_enough")
        total += 1
        p2 = (mode == "web_search") and ("kb_sufficient→WEB" in detail)
        print(f"  a2. kb 不足以回答 -> {mode} ({detail})  "
              f"{'✓' if p2 else '✗ 应回退kb_sufficient且判为WEB'}")
        n_pass += int(p2)
        # b. kb_sufficient 也失败 -> 保守 KB
        mode, detail, log = run_case(fault_name, fake_llm, "fail")
        total += 1
        p3 = (mode == "knowledge_base") and ("kb_sufficient→KB" in detail)
        print(f"  b.  kb 也失败    -> {mode} ({detail})  "
              f"{'✓' if p3 else '✗ 应保守走知识库(不误走WEB)'}")
        n_pass += int(p3)
        # c. 降级日志
        total += 1
        has_log = ("[L3]" in log and ("回退 kb_sufficient" in log or "保守走知识库" in log))
        print(f"  c.  降级日志说明  -> {'✓' if has_log else '✗ 缺降级日志'}")
        if has_log:
            for line in log.splitlines():
                if "[L3]" in line:
                    print(f"       日志: {line.strip()}")
        n_pass += int(has_log)

    print(f"\n===== fallback 测试结果 =====  {n_pass}/{total} 通过")
    if n_pass == total:
        print("三种故障下均正确回退 kb_sufficient, kb_sufficient 失败时保守走知识库, 降级日志齐全。")
    else:
        print("!! 存在未通过项, 见上方 ✗")


if __name__ == "__main__":
    main()
