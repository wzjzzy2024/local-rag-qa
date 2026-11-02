# -*- coding: utf-8 -*-
"""
课设 RAG - 检索增强问答 + 联网搜索兜底
逻辑:
  1. 用户问题 -> bge-m3 向量化 -> FAISS 检索 top-k
  2. 若最高余弦相似度 >= SCORE_THRESHOLD: 用知识库内容生成答案
  3. 否则: 自动调用 Bing 联网搜索，抓取结果注入上下文生成答案
用法: python rag_chat.py
"""
import os
import re
import sys
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import faiss
import numpy as np
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

from web_providers import simplify_query, search_with_fallback

BASE = Path(__file__).resolve().parent
INDEX_DIR = BASE / "index"

EMBED_MODEL = "bge-m3"
LLM_MODEL = "qwen2.5:7b-instruct-q4_K_M"
OLLAMA_BASE = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

SCORE_THRESHOLD = 0.55
TOP_K = 5
MAX_SEARCH = 6

# ---- L2: 联网上下文预算 ----
# 小模型 context 32768，联网上下文总预算默认 40%（约 13k token）
# 中文/英文混合下 1 token 约 1.5~2 字符，按 1.5 保守换算为字符预算
CONTEXT_BUDGET_TOKENS = int(os.environ.get("WEB_CONTEXT_BUDGET_TOKENS", "13000"))
_CHARS_PER_TOKEN = 1.5
_CONTEXT_BUDGET_CHARS = int(CONTEXT_BUDGET_TOKENS * _CHARS_PER_TOKEN)

# ---- L3 阶段一: 知识库二级判定（已由混合路由取代，常量保留兼容历史验收脚本）----
# ask() 路由现由下方 MIX_* 体系控制。SECONDARY_ROUTE_CHECK_MAX=0.68 即旧灰区上界，
# 混合路由下灰区上界为 MIX_HIGH（默认 0.75）。l3_acceptance/l3_compare*.py 仍引用
# 此常量，故保留定义；MIX_MODE=kb_sufficient 即切回阶段一判定方式。
SECONDARY_ROUTE_ENABLED = os.environ.get("SECONDARY_ROUTE_ENABLED", "1") == "1"
SECONDARY_ROUTE_CHECK_MAX = float(os.environ.get("SECONDARY_ROUTE_CHECK_MAX", "0.68"))

# ---- L3: 混合路由（三段置信度分工，替换阶段一路由分支）----
#   best <  MIX_LOW            -> 直接联网（D 段纯阈值，不改原逻辑）
#   best >  MIX_HIGH           -> 直接知识库（A 段纯阈值，不改原逻辑）
#   MIX_LOW <= best <= MIX_HIGH -> 灰区判定:
#       MIX_MODE=tool_calling（默认）: qwen 原生 tool calling 判定（中性格式约束，
#           无"能用自身知识就NO_TOOL"语义判断）；异常/超时/第三种输出回退 kb_sufficient，
#           再失败保守走知识库
#       MIX_MODE=kb_sufficient: 切回阶段一判定方式（灰区走 kb_sufficient）
#   上下界可配、判定方式可配，A/D 两段行为不变。
MIX_LOW = float(os.environ.get("MIX_LOW", "0.55"))
MIX_HIGH = float(os.environ.get("MIX_HIGH", "0.75"))
MIX_MODE = os.environ.get("MIX_MODE", "tool_calling")
# 灰区 tool calling 判定超时（秒），超时抛异常回退 kb_sufficient
GREY_TC_TIMEOUT = float(os.environ.get("GREY_TC_TIMEOUT", "15"))

# 灰区 tool calling 判定用中性格式约束：只约束输出形态，不含任何语义诱导。
TC_CONSTRAINT = (
    "你是路由决策助手。请决定是否调用 search 工具。\n"
    "规则: 1) 若决定调用工具, 只输出 tool call JSON, 不输出任何其他文本; "
    "2) 若决定不调用, 只输出 NO_TOOL 四个字母, 不输出任何其他文本。"
)

SEARCH_TOOL_DEF = [{
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

SYSTEM_PROMPT = (
    "你是一个严谨的知识库问答助手。请基于提供的资料回答用户问题。\n"
    "要求：1) 回答准确、条理清晰，中文作答；2) 若资料不足，明确说明；"
    "3) 不要编造资料中没有的事实。"
)


def load_index():
    data = np.frombuffer((INDEX_DIR / "kb_index.faiss").read_bytes(), dtype="uint8")
    index = faiss.deserialize_index(data)
    with open(INDEX_DIR / "kb_metas.npy", "rb") as f:
        metas = np.load(f, allow_pickle=True)
    with open(INDEX_DIR / "kb_texts.npy", "rb") as f:
        texts = np.load(f, allow_pickle=True)
    return index, texts, metas


# ---- L2: 正文清洗规则（可扩展配置） ----
# 每条规则 = (类型, 模式, 说明)，主逻辑 clean_content 遍历执行。
#   类型 "line": 行内匹配该正则的部分被删除（可用 .* 吞掉整行）；行变空则丢弃
#   类型 "drop": 整行命中该正则则整行丢弃
# 遇到新噪声时，只需在下面列表追加一条规则（带说明），不要改 clean_content 主逻辑。
_CLEAN_RULES = [
    # ---- 浏览器/框架残留 ----
    ("line", r"Close Search.*",       "站点内嵌搜索框文本"),
    ("line", r"PyTorch Logo.*",       "站点 logo 文本残留"),
    ("line", r"^搜索\s*$",            "页面搜索栏占位文字"),
    ("line", r"^Menu\s*$",            "页面菜单占位文字"),
    ("line", r"Previous Next.*",      "分页器 Previous/Next"),
    ("line", r"^首页\s*资讯\s*军事.*", "门户网站顶部导航条"),

    # ---- 侧栏 / 推荐 / 相关阅读 ----
    ("line", r"热门新闻.*",           "侧栏热门新闻标题列表"),
    ("line", r"相关阅读.*",           "相关阅读/推荐阅读/延伸阅读"),
    ("line", r"继续阅读.*",           "继续阅读链接"),
    ("line", r"相同主题内容.*",       "相同主题内容链接"),
    ("line", r"相关资讯.*",           "相关资讯列表"),

    # ---- 版权 / 来源 / 投诉 ----
    ("line", r"©\s*(20\d\d|Reuters|[A-Z]).*", "版权声明或图片来源行"),
    ("line", r"Copyright.*",          "英文版权行"),
    ("line", r"版权所有.*",           "中文版权行"),
    ("line", r"责任编辑[:：].*",      "责任编辑署名"),
    ("line", r"文章投诉热线.*",       "文章投诉热线"),
    ("line", r"投诉邮箱.*",           "投诉邮箱"),

    # ---- 交互 / 分享 / 评论 ----
    ("line", r"查看所有评论.*",       "评论计数/查看评论"),
    ("line", r"评论区.*",             "评论区提示"),
    ("line", r"点赞.*",               "点赞提示"),
    ("line", r"分享\s*$",             "分享按钮文字"),
    ("line", r"Image\s*\d+\s*:.*",    "图片标注(Image N: ...)"),
    ("line", r"^主题分类\s*$",        "主题分类栏"),

    # ---- 财经/工具类导航 ----
    ("line", r"^工具\s+日历.*",       "财经工具导航(日历)"),
    ("line", r"^选股器.*",            "选股器等投资工具链接"),
    ("line", r"^教育\s+提醒\s+下载APP.*", "下载APP提示"),
    ("line", r"发布时间\s+\d{4}年.*", "页面发布时间戳(正文已含日期)"),
    ("line", r"更新时间\s+\d{4}年.*", "页面更新时间戳"),
    ("line", r"^关键词[:：].*",       "SEO 关键词行"),

    # ---- 纯栏目词单行 ----
    ("drop", r"^(教育|时尚|科技|财经|娱乐|体育|汽车|房产|家居|手机|数码|新闻)$",
             "门户频道导航单行"),

    # ---- 外文 / 符号残留 ----
    ("drop", r".*[\u0400-\u04FF\u0600-\u06FF\u0900-\u097F].*",
             "西里尔/阿拉伯/天城文残留行(如俄语)"),
    ("drop", r"^[\s:：|*#.\-–—\\/、，。;；'\"()（）\[\]{}<>«»]*$",
             "孤立符号行(无意义)"),
]

# 正文内嵌发布时间特征（RFI 等页面正文含"发表时间：06/10/2026 - 12:00"）
_PUB_IN_BODY_RE = re.compile(
    r"发表时间[:：]\s*(\d{1,2}[/-]\d{1,2}[/-]\d{4})|\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日"
)


def extract_pub_from_text(text):
    """从正文提取发布时间（Tavily basic 深度不返回 published_date 时的兜底）

    解析逻辑:
      1. 优先匹配 "YYYY年M月D日" 中文格式 -> 无歧义，直接返回
      2. 匹配 "发表时间：DD/MM/YYYY" 类数字格式 -> 无法确定月/日顺序
         （美式 MM/DD 与欧式 DD/MM 含义相反），保守处理: 保留原始字符串并
         标注"日期格式不确定"，不做强行解析——错误日期比没有日期更危险。
    """
    if not text:
        return None
    m = _PUB_IN_BODY_RE.search(text)
    if not m:
        return None
    raw = m.group(0).strip()
    # 去掉"发表时间："前缀，只留日期串
    raw = re.sub(r"^发表时间[:：]\s*", "", raw)
    # 数字斜杠/横杠格式（如 06/10/2026）: 月日顺序有歧义，保守标注
    if re.match(r"^\d{1,2}[/-]\d{1,2}[/-]\d{4}$", raw):
        return f"{raw}（日期格式不确定）"
    return raw


def clean_content(text):
    """清洗正文: 去 HTML 残留、按 _CLEAN_RULES 逐行去导航/广告/外文残留、
    markdown 标记、压缩空白。新噪声只需往 _CLEAN_RULES 加规则。"""
    if not text:
        return ""
    t = re.sub(r"<[^>]+>", "", text)          # HTML 标签
    t = re.sub(r"```", "", t)                  # markdown 代码块
    kept = []
    for line in t.splitlines():
        s = line.strip()
        if not s:
            continue
        # drop 规则: 整行命中即丢弃
        if any(re.match(pat, s) for typ, pat, _d in _CLEAN_RULES if typ == "drop"):
            continue
        # line 规则: 行内删除匹配部分
        for typ, pat, _d in _CLEAN_RULES:
            if typ == "line":
                s = re.sub(pat, "", s)
        s = s.strip()
        if s:
            kept.append(s)
    out = "\n".join(kept)
    out = re.sub(r"[ \t]+", " ", out)          # 行内多余空白
    return out.strip()


def truncate_by_paragraphs(text, budget_chars):
    """按段落截断（不从句子中间切），超出预算时保留前段并标注"""
    if not text:
        return ""
    if len(text) <= budget_chars:
        return text
    paras = [p.strip() for p in re.split(r"\n+", text) if p.strip()]
    out, used = [], 0
    for p in paras:
        if used + len(p) + 1 > budget_chars:
            break
        out.append(p)
        used += len(p) + 1
    if not out:
        out = [paras[0][: budget_chars]]   # 单段超长时从段内截（保底）
    return "\n".join(out) + "\n[已按上下文预算截断]"


def build_context(hits, best, kb_fallback=False):
    """L2: 联网上下文模板——保留 title/url/published_at/正文片段，多条去重，
    总长受 CONTEXT_BUDGET_TOKENS 约束（默认 40% 窗口）。
    kb_fallback=True 表示 L3 知识库二级判定回退场景（header 如实说明）。"""
    if kb_fallback:
        header = (f"（提示：本地知识库命中（最高相关度 {best:.2f}）但片段不足以"
                  f"回答该问题，以下为联网搜索结果）")
    else:
        header = (f"（提示：本地知识库最高相关度 {best:.2f} 低于阈值 "
                  f"{SCORE_THRESHOLD}，以下为联网搜索结果）")
    lines = [header, ""]

    seen_urls = set()
    dedup = []
    for r in hits:
        u = r.get("url", "")
        if u and u in seen_urls:
            continue
        seen_urls.add(u)
        dedup.append(r)

    used = len(header)
    for i, r in enumerate(dedup, 1):
        title = (r.get("title") or "").strip()
        url = (r.get("url") or "").strip()
        body_src = r.get("content") or r.get("snippet") or ""
        cleaned = clean_content(body_src)
        # 发布时间: provider 字段优先，为空则从正文提取兜底
        pub = (r.get("published_at") or extract_pub_from_text(body_src)
               or "未知").strip()
        body = truncate_by_paragraphs(cleaned,
                                      _CONTEXT_BUDGET_CHARS // max(len(dedup), 1))
        block = f"[{i}] {title}\n   来源: {url}\n   发布时间: {pub}\n   正文: {body}"
        if used + len(block) > _CONTEXT_BUDGET_CHARS:
            break
        lines.append(block)
        used += len(block)

    return "\n\n".join(lines)


def rewrite_query(llm, question):
    """用 LLM 生成高质量搜索词（行业标准做法）。

    实测发现: 规则精简对"事件结果"类问题有效（冠军/最新版本），
    但对"事件属性"类问题（举办地/发布时间/榜单）仍会被 Bing 误判，
    因此先用 LLM 改写为最适合搜索引擎的关键词。
    """
    prompt = (
        "你负责把用户的问题改写成 1 个最适合搜索引擎的中文关键词或短语。\n"
        "规则: 1) 只输出一行，不要解释；2) 保留年份等关键限定；"
        "3) 使用新闻/百科常用的搜索表达，如\"2024巴黎奥运会金牌榜\"；"
        "4) 去掉口语词（是什么/在哪里/怎么/呢）。\n"
        f"问题: {question}\n关键词:"
    )
    try:
        resp = llm.invoke(prompt)
        kw = resp.content.strip().strip('"\'，。').splitlines()[0].strip()
        if kw and len(kw) <= 60:
            return kw
    except Exception as e:
        print(f"[web] LLM 改写失败，回退规则精简: {e}")
    return simplify_query(question)


def kb_sufficient(llm, question, texts, sel):
    """L3 阶段一: 二级判定——知识库命中后判断 top-5 片段是否真能回答该问题。

    复用 is_relevant 的"LLM 判断相关性"思路，但判定对象从"搜索结果"改为
    "知识库命中片段 vs 问题"。返回 True=足以回答(走知识库) / False=回退联网。
    判定失败时保守返回 True（走知识库），避免把能答的问题误回退。
    """
    snippet_list = "\n".join(
        f"- {texts[i][:150]}" for i, _s in sel[:TOP_K]
    )
    p = (
        "判断下面的知识库片段是否足以回答用户的问题。\n"
        f"问题: {question}\n知识库片段:\n{snippet_list}\n"
        "如果片段包含答案（或能据此给出可靠答案），回复: 足以回答\n"
        "如果片段只是相关主题背景、缺少回答所需的关键信息"
        "（尤其是实时/最新信息），回复: 不足以回答\n"
        "只输出\"足以回答\"或\"不足以回答\"。"
    )
    try:
        resp = llm.invoke(p)
        ans = resp.content.strip()
        return "足以回答" in ans and "不足以回答" not in ans
    except Exception as e:
        print(f"[L3] 二级判定失败，保守走知识库: {e}")
        return True


def _kb_suff_route(llm, question, texts, sel):
    """灰区 kb_sufficient 判定（阶段一方式 / tool calling fallback 共用）:
    足以回答 -> KB；不足以回答 -> WEB；判定异常 -> 保守 KB（不误走 WEB）。"""
    try:
        if kb_sufficient(llm, question, texts, sel):
            return "knowledge_base", "灰区kb_sufficient→KB(足以回答)"
        return "web_search", "灰区kb_sufficient→WEB(不足以回答)"
    except Exception as e:
        print(f"[L3] kb_sufficient 判定异常({type(e).__name__}: {e})，"
              f"保守走知识库")
        return "knowledge_base", "灰区kb_sufficient→KB(判定异常保守)"


def _grey_route_judge(llm, question, texts, sel):
    """L3 混合路由灰区判定: tool calling 主路径 + kb_sufficient fallback。

    返回 (mode, route_detail)。mode ∈ {"web_search", "knowledge_base"}。
    故障降级链: tool calling 异常/超时/第三种输出 -> kb_sufficient -> 保守 KB。
    每次判定都打印降级原因，便于排查。
    """
    if MIX_MODE == "kb_sufficient":
        return _kb_suff_route(llm, question, texts, sel)

    # tool calling 主路径（中性格式约束）
    try:
        judge_llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_BASE,
                               temperature=0.2, request_timeout=GREY_TC_TIMEOUT)
        resp = judge_llm.bind_tools(SEARCH_TOOL_DEF).invoke(
            [SystemMessage(content=TC_CONSTRAINT), HumanMessage(content=question)])
        tool_calls = getattr(resp, "tool_calls", None) or []
        if tool_calls:
            return ("web_search",
                    f"灰区tool calling→WEB(调{','.join(tc.get('name','?') for tc in tool_calls)})")
        content = (getattr(resp, "content", "") or "").strip()
        if content.upper() == "NO_TOOL":
            return "knowledge_base", "灰区tool calling→KB(NO_TOOL)"
        # 第三种输出: 既非 TOOL_CALL 也非 NO_TOOL，格式未遵循约束
        print(f"[L3] tool calling 输出未遵循约束(非TOOL_CALL且非NO_TOOL): "
              f"{content[:80]!r}，回退 kb_sufficient")
    except Exception as e:
        print(f"[L3] tool calling 判定异常({type(e).__name__}: {e})，回退 kb_sufficient")
    return _kb_suff_route(llm, question, texts, sel)


def _web_search_branch(llm, question, best, kb_fallback=False):
    """联网搜索分支（L1/L2 逻辑原样迁入，供正常低置信度与 L3 回退共用）:
    改写搜索词 -> LLM 相关性判定 -> provider 搜索 -> 清洗构建上下文。
    返回 (context, source_note)。"""
    kw = rewrite_query(llm, question)
    print(f"[web] 搜索词: {kw}")

    def is_relevant(results, q, c):
        """让 LLM 判断搜索结果是否与问题相关"""
        snippet_list = "\n".join(
            f"- {r['title']} | {r['snippet'][:80]}" for r in results[:5]
        )
        p = (
            "判断下面搜索结果是否与问题相关。\n"
            f"问题: {q}\n搜索结果:\n{snippet_list}\n"
            "如果结果能回答问题（或包含问题答案线索），回复: 相关\n"
            "如果结果与问题无关（如年份百科、节假日、官网首页等泛化内容），回复: 不相关\n"
            "只输出\"相关\"或\"不相关\"。"
        )
        try:
            resp = llm.invoke(p)
            ans = resp.content.strip()
            return "相关" in ans and "不相关" not in ans
        except Exception as e:
            print(f"[web] 相关性判断失败，接受结果: {e}")
            return True

    hits, used_provider = search_with_fallback(
        question, top_k=MAX_SEARCH, search_word=kw,
        relevance_check=is_relevant)
    print(f"[web] 实际使用 provider: {used_provider}")
    if not hits:
        context = "联网搜索无可用结果，请基于常识作答并说明信息来源有限。"
        source_note = ""
    else:
        context = build_context(hits, best, kb_fallback=kb_fallback)
        source_note = "\n\n[联网来源] " + "、".join(r["url"] for r in hits)
        # L2 验收: 打印实际喂给模型的联网上下文（可截断，能看出结构）
        print(f"[L2] 联网上下文预算: {CONTEXT_BUDGET_TOKENS} token "
              f"(≈{_CONTEXT_BUDGET_CHARS} 字符)")
        preview = context if len(context) <= 1500 else context[:1500] + "...[截断展示]"
        print("[L2] 实际喂给模型的联网上下文:\n" + preview)
    return context, source_note


def ask(question, _cache=None):
    if _cache is not None:
        # 测试/批处理可复用预加载的 (emb, index, texts, metas)，避免重复加载 606MB 索引
        emb, index, texts, metas = _cache
    else:
        emb = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE)
        index, texts, metas = load_index()

    qv = np.array(emb.embed_query(question), dtype="float32").reshape(1, -1)
    faiss.normalize_L2(qv)
    scores, idx = index.search(qv, TOP_K)
    scores = scores[0]          # 余弦相似度
    idx = idx[0]

    best = float(scores[0]) if len(scores) and scores[0] > -9e9 else 0.0

    llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_BASE, temperature=0.2)

    sel = [(int(i), float(s)) for i, s in zip(idx, scores)
           if int(i) >= 0 and s > 0]

    # ---- L3 混合路由: 三段置信度分工 ----
    #   <MIX_LOW 直接联网; >MIX_HIGH 直接知识库; 灰区走判定(tool calling/kb_sufficient)
    if best < MIX_LOW:
        mode = "web_search"
        route_detail = f"D段纯阈值直连→WEB"
        kb_fallback = False
    elif best > MIX_HIGH:
        mode = "knowledge_base"
        route_detail = "A段纯阈值直连→KB"
        kb_fallback = False
    else:
        mode, route_detail = _grey_route_judge(llm, question, texts, sel)
        kb_fallback = (mode == "web_search")  # 灰区回退联网需标注"知识库命中但不足"

    if mode == "web_search":
        context, source_note = _web_search_branch(
            llm, question, best, kb_fallback=kb_fallback)
    else:
        ctx_parts, srcs = [], []
        for i, s in sel:
            ctx_parts.append(texts[i])
            srcs.append(metas[i].get("source", "?"))
        context = "\n\n---\n\n".join(ctx_parts)
        source_note = "\n\n[知识库来源] " + "、".join(sorted(set(srcs)))

    prompt = f"""{SYSTEM_PROMPT}

===资料===
{context}
===问题===
{question}"""

    print(f"\n[路由] {route_detail} (best={best:.3f})")
    print(f"[模式] {'知识库检索' if mode=='knowledge_base' else '联网搜索兜底'} "
          f"(最高相关度 {best:.2f})")
    print(f"[检索片段 top{min(3,len(scores))} 相似度] "
          + "、".join(f"{s:.2f}" for s in scores[:3] if s > 0))
    if mode == "knowledge_base":
        print("[前3条片段]")
        for k, (i, s) in enumerate(sel[:3], 1):
            print(f"  {k}. [{s:.2f}] {texts[i][:100]}...")
    print("-" * 50)

    resp = llm.invoke(prompt)
    print(resp.content)
    print(source_note)
    return mode, route_detail, best


def main():
    print("=" * 50)
    print("课设 RAG 问答系统（本地知识库 + 联网搜索兜底）")
    print("输入问题回车问答；输入 exit 退出")
    print("=" * 50)
    while True:
        try:
            q = input("\n> 问题: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q.lower() in ("exit", "quit", "退出"):
            break
        ask(q)


if __name__ == "__main__":
    main()
