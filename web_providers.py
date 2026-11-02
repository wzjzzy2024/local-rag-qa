# -*- coding: utf-8 -*-
"""
L1: WebSearchProvider 抽象 + 双 provider + 降级链

- BingProvider: 迁移自原 rag_chat.bing_search，行为基本不变，仅补 mkt/setlang 参数
- TavilyProvider: 官方 REST API (https://docs.tavily.com)，用 requests 直连，不装 SDK
- search_with_fallback: 主 provider 失败(缺 key/网络/429/5xx)时自动降级 Bing，再失败返回空

统一返回结构（每条 dict）:
  {
    "title": str, "url": str, "snippet": str,
    "content": str | "",          # 正文（Tavily 自带；Bing RSS 无正文，为空，L2 再抓）
    "published_at": str | None,   # 发布时间（RFC 或 ISO 字符串）
    "source": "tavily" | "bing",
  }
"""
import os
import re
import sys
import time
import html as html_lib
import urllib.parse
import urllib.request
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import requests


# ============================================================
# 0. 查询词处理（原 rag_chat 迁移，供 BingProvider 使用）
# ============================================================
def simplify_query(question):
    """把口语化问题转成适合搜索引擎的关键词。"""
    q = question.strip()
    q = re.sub(r"[？?]$", "", q)
    q = re.sub(r"^(请问|帮我|麻烦|请|告诉我|我想知道)", "", q)
    q = re.sub(r"(颁给了谁|颁给谁|是谁|是什么|是什么啊|怎么样|怎么办|怎么用|"
               r"如何|多少|有哪些|介绍下|介绍一下|讲讲|解释下|解释一下|是啥|啥)", "", q)
    q = re.sub(r"([\u4e00-\u9fa5]|[A-Za-z0-9])的([\u4e00-\u9fa5]|[A-Za-z0-9])",
               r"\1\2", q)
    q = re.sub(r"(?<!\d)(\d{4})(?!年)(?=[\u4e00-\u9fa5])", r"\1年", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q or question


def move_year_to_end(kw):
    """年份开头移到末尾（Bing 中文对年份开头查询召回差）"""
    m = re.match(r"^(\d{4}年?)(.*)$", kw)
    if m:
        year = m.group(1).rstrip("年")
        rest = m.group(2).strip()
        if rest:
            return f"{rest}{year}"
    return kw


_EN_MAP = {
    "奥运会": "Olympics", "冬奥会": "Winter Olympics", "世界杯": "World Cup",
    "金牌榜": "medal table", "冠军": "champion winner", "举办地": "host city venue",
    "手机": "phone smartphone", "模型": "model", "机器人": "robot",
}


def en_fallback(kw):
    """中文搜索词转英文兜底词"""
    out = kw
    for zh, en in _EN_MAP.items():
        out = out.replace(zh, en)
    m = re.search(r"(\d{4})", out)
    if m and m.group(1) in out:
        out = out.replace(m.group(1), m.group(1) + " ") if m.start() == 0 else out
    return out if out != kw else None


# ============================================================
# 1. Provider 接口
# ============================================================
class WebSearchProvider:
    name = "base"

    def search(self, query, top_k=5, relevance_check=None):
        """query: 搜索词；top_k: 返回条数；relevance_check: 可选相关性判定函数
        返回统一结构 list[dict]，失败抛异常由上层降级"""
        raise NotImplementedError


# ============================================================
# 1.5 API key 读取（优先环境变量，回退项目根 .env，不引 dotenv 依赖）
# ============================================================
def load_api_key(name="TAVILY_API_KEY"):
    """读取 API key: 优先 os.environ，回退项目根目录 .env 文件"""
    val = os.environ.get(name, "").strip()
    if val:
        return val
    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.exists():
        try:
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                if k.strip() == name:
                    return v.strip().strip('"').strip("'")
        except OSError:
            pass
    return ""


# ============================================================
# 2. BingProvider（迁移原 bing_search + 补 mkt/setlang）
# ============================================================
class BingProvider(WebSearchProvider):
    name = "bing"

    def __init__(self, mkt="zh-CN", setlang="zh", timeout=15, max_retries=2):
        self.mkt = mkt          # 可配，默认中文市场
        self.setlang = setlang  # 可配，默认中文界面
        self.timeout = timeout
        self.max_retries = max_retries

    def _build_url(self, kw):
        params = [
            ("q", kw),
            ("format", "rss"),
            ("mkt", self.mkt),
            ("setlang", self.setlang),
        ]
        return "https://www.bing.com/search?" + urllib.parse.urlencode(params)

    def _request_once(self, url):
        req = urllib.request.Request(url, headers={
            "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                           "AppleWebKit/537.36 (KHTML, like Gecko) "
                           "Chrome/120.0 Safari/537.36")
        })
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return resp.read().decode("utf-8", errors="ignore")

    def _parse(self, page, top_k):
        items = re.findall(r"<item>(.*?)</item>", page, re.S)
        results = []
        for it in items[:top_k]:
            t = re.search(r"<title>(.*?)</title>", it, re.S)
            l = re.search(r"<link>(.*?)</link>", it, re.S)
            d = re.search(r"<description>(.*?)</description>", it, re.S)
            pub = re.search(r"<pubDate>(.*?)</pubDate>", it, re.S)
            if not (t and l):
                continue
            results.append({
                "title": html_lib.unescape(t.group(1)).strip(),
                "url": l.group(1).strip(),
                "snippet": html_lib.unescape(d.group(1)).strip() if d else "",
                "content": "",          # RSS 无正文，L2 再抓
                "published_at": pub.group(1).strip() if pub else None,
                "source": self.name,
            })
        return results

    def search(self, query, top_k=5, relevance_check=None):
        # query 即搜索词（上层 search_with_fallback 已按 search_word 或 simplify 处理）
        kw = query
        candidates = [kw]
        moved = move_year_to_end(kw)
        if moved != kw:
            candidates.append(moved)
        en = en_fallback(kw)
        if en:
            candidates.append(en)

        last_err = None
        for c in candidates:
            url = self._build_url(c)
            attempt = 0
            while attempt <= self.max_retries:
                try:
                    page = self._request_once(url)
                    break
                except Exception as e:
                    last_err = str(e)
                    attempt += 1
                    if attempt > self.max_retries:
                        print(f"[bing] 请求失败(重试耗尽): {e}")
                        break
                    wait = min(2 ** attempt, 10)
                    print(f"[bing] 请求失败({e})，{wait}s 后重试 "
                          f"({attempt}/{self.max_retries})")
                    time.sleep(wait)
            else:
                continue  # while 正常结束（无 break）说明全失败

            results = self._parse(page, top_k)
            if not results:
                continue
            if relevance_check is not None:
                try:
                    ok = relevance_check(results, query, c)
                except Exception:
                    ok = True
                if not ok:
                    print(f"[bing] 候选词 \"{c}\" 结果与问题不相关，尝试下一个...")
                    continue
            if c != candidates[0]:
                print(f"[bing] 已自动切换搜索词: {candidates[0]} -> {c}")
            return results
        if last_err:
            print(f"[bing] 所有候选词均失败，最后错误: {last_err}")
        return []


# ============================================================
# 3. TavilyProvider（官方 REST，requests 直连）
# ============================================================
class TavilyProvider(WebSearchProvider):
    name = "tavily"

    API_URL = "https://api.tavily.com/search"

    def __init__(self, api_key=None, timeout=10, max_retries=2):
        self.api_key = (api_key or load_api_key("TAVILY_API_KEY")).strip()
        self.timeout = timeout
        self.max_retries = max_retries
        self.last_params = None   # 供验收打印请求参数

    def search(self, query, top_k=5, relevance_check=None):
        if not self.api_key:
            raise RuntimeError("TAVILY_API_KEY 未设置，无法调用 Tavily")

        body = {
            "query": query,
            "max_results": top_k,
            "search_depth": "basic",
        }
        self.last_params = {"query": query, "max_results": top_k,
                            "search_depth": "basic",
                            "api_key": self._mask_key()}

        attempt = 0
        while True:
            try:
                resp = requests.post(
                    self.API_URL,
                    json=body,
                    headers={"Authorization": f"Bearer {self.api_key}",
                             "Content-Type": "application/json"},
                    timeout=self.timeout,
                )
            except Exception as e:
                attempt += 1
                if attempt > self.max_retries:
                    raise RuntimeError(f"Tavily 请求异常(重试耗尽): {e}")
                wait = min(2 ** attempt, 10)
                print(f"[tavily] 请求异常({e})，{wait}s 后重试 "
                      f"({attempt}/{self.max_retries})")
                time.sleep(wait)
                continue

            if resp.status_code == 200:
                break
            # 429 或 5xx：退避重试
            if resp.status_code in (429,) or resp.status_code >= 500:
                attempt += 1
                if attempt > self.max_retries:
                    raise RuntimeError(
                        f"Tavily HTTP {resp.status_code}: {resp.text[:200]}")
                wait = min(2 ** attempt, 10)
                print(f"[tavily] HTTP {resp.status_code}，{wait}s 后重试 "
                      f"({attempt}/{self.max_retries})")
                time.sleep(wait)
                continue
            # 4xx（400/401/403 等）：不重试，直接失败
            raise RuntimeError(f"Tavily HTTP {resp.status_code}: {resp.text[:200]}")

        data = resp.json()
        results = []
        for r in data.get("results", [])[:top_k]:
            results.append({
                "title": r.get("title", "").strip(),
                "url": r.get("url", "").strip(),
                "snippet": (r.get("content") or "")[:300].strip(),
                "content": r.get("content") or "",
                "published_at": r.get("published_date") or None,
                "source": self.name,
            })
        return results

    def _mask_key(self):
        k = self.api_key
        if not k:
            return "<empty>"
        if len(k) <= 12:
            return k[:8] + "***"
        return k[:8] + "***" + k[-4:]


# ============================================================
# 4. 工厂 + 降级链
# ============================================================
def get_provider(name=None):
    """按配置创建 provider。name 未指定时读环境变量 WEB_PROVIDER（默认 tavily）"""
    name = (name or os.environ.get("WEB_PROVIDER", "tavily")).strip().lower()
    if name == "bing":
        return BingProvider()
    if name == "tavily":
        return TavilyProvider()
    print(f"[provider] 未知 provider: {name}，回退 tavily")
    return TavilyProvider()


def search_with_fallback(query, top_k=5, relevance_check=None, preferred=None,
                         search_word=None):
    """主 provider 优先，失败自动降级 Bing；Bing 也失败返回空列表。

    query: 原始用户问题（用于相关性判定/候选词生成）
    search_word: 已改写好的搜索词（LLM 改写），优先于规则精简
    返回 (results, used_provider_name)
    """
    provider = None
    try:
        provider = get_provider(preferred)
        print(f"[provider] 使用主 provider: {provider.name}")
        kw = search_word or simplify_query(query)
        results = provider.search(kw, top_k=top_k, relevance_check=relevance_check)
        if results:
            return results, provider.name
        print(f"[provider] {provider.name} 无结果，降级到 bing")
    except Exception as e:
        # provider 可能尚未创建成功（get_provider 本身抛异常），不能用 provider.name
        print(f"[provider] {getattr(provider, 'name', '?')} 失败: {e}，降级到 bing")

    try:
        bing = BingProvider()
        kw = search_word or simplify_query(query)
        results = bing.search(kw, top_k=top_k, relevance_check=relevance_check)
        return results, "bing"
    except Exception as e:
        print(f"[provider] bing 也失败: {e}，返回空列表")
        return [], "bing"
