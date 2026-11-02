# 课设 RAG 问答系统（本地知识库 + 联网搜索兜底）

对应课设第三部分**选项 B（LangChain 实现 RAG）**。

## 环境

| 组件 | 说明 |
|---|---|
| 推理框架 | Ollama（本地服务，默认 `http://localhost:11434`） |
| 生成模型 | `qwen2.5:7b-instruct-q4_K_M`（7.6B，中文问答） |
| 向量模型 | `bge-m3`（多语言 embedding，1024 维） |
| Python | 3.10+（开发/验收使用 3.13） |
| 关键依赖 | 见 `requirements.txt`：langchain-ollama / langchain-core / langchain-text-splitters / faiss-cpu / numpy / requests |

> 可用环境变量 `OLLAMA_BASE_URL` 覆盖默认 Ollama 地址。

## 目录结构

```
RAG/
├── kb_source/        # 知识库源文档（.md；不入库，需自备）
├── index/            # 构建好的 FAISS 索引（不入库，由 build_index.py 生成）
│   ├── kb_index.faiss   # 向量索引
│   ├── kb_metas.npy     # 每条向量的来源元数据
│   └── kb_texts.npy     # 每条向量的原文
├── .env.example      # 环境变量模板（复制为 .env，填入 key）
├── requirements.txt  # 第三方依赖（按实际 import 生成）
├── build_index.py    # 建库脚本（断点续传 + 自动重试）
├── web_providers.py  # L1: 搜索后端抽象（WebSearchProvider 接口 + Bing/Tavily 双 provider + 降级链）
├── rag_chat.py       # 问答脚本（L2 正文清洗/截断/上下文模板 + L3 二级判定路由）
├── l1_acceptance.py  # 验收：L1 provider 对比 / 降级 / 中文 / 切换（mock 为主，无需 key）
├── l1_retest.py      # 验收：L1 Tavily 真实 key 实测（条数/延迟/content/中文）
├── l2_acceptance.py  # 验收：L2 联网上下文结构 / groundedness（3 条实时问题）
├── l3_acceptance.py  # 验收：L3 阶段一 二级判定路由正确率 / 误杀率 / 端到端回退
├── l3_compare.py     # 对比：L3 阶段二并存对比 v1（20 条，无约束，结论延迟超限停用）
├── l3_compare_v2.py  # 对比：L3 阶段二并存对比 v2（28 条 + 输出约束，发现 D 类语义扩展副作用）
├── l3_mixed.py       # 验证：混合路由（灰区 tool calling 判定，中性约束，14/14=100%）
├── tool_call_stability.py # 验证：L3 阶段二前置 qwen tool calling 稳定性（只测能力，不改架构）
├── quick_notool_test.py   # 验证：prompt 约束后"不该调"延迟（5.25s→0.73s，L4 重估依据）
├── regression_clean.py # 回归：补1 清洗规则 / 补2 published_at 歧义
└── e2e_test.py       # 端到端问答冒烟
```

## 快速开始（首次运行）

前置条件与完整顺序：

1. 安装 [Ollama](https://ollama.com/download)，并确保服务已启动：
   ```bash
   ollama serve
   ```
2. 拉取所需模型：
   ```bash
   ollama pull qwen2.5:7b-instruct-q4_K_M
   ollama pull bge-m3
   ```
3. 安装 Python 依赖（建议使用虚拟环境）：
   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
4. 准备知识库源文档：把 `.md` 文件放入 `kb_source/`。本项目演示语料来自
   [LINJIANG12/video2book-courses](https://github.com/LINJIANG12/video2book-courses)：
   ```bash
   git clone https://github.com/LINJIANG12/video2book-courses.git kb_source
   ```
   （也可替换为自己的 md 文档；`kb_source/` 不入库。）
5. 构建索引（首次全量约 2 小时，支持断点续传）：
   ```bash
   python build_index.py
   ```
   快速试跑可用 `python build_index.py --sample 50`。
6. 配置联网搜索 key（可选）：不配置时 Tavily 不可用，会自动降级 Bing。
   ```bash
   cp .env.example .env
   # 编辑 .env，填入 TAVILY_API_KEY=tvly-...
   ```
7. 启动问答：
   ```bash
   python rag_chat.py
   ```
   输入问题回车问答，`exit` 退出。

> `index/` 与 `kb_source/` 都已加入 `.gitignore`，克隆仓库后必须自行准备源文档并构建索引。

## 使用方法

### 1. 启动问答（索引已构建好）

```bash
python rag_chat.py
```

交互式输入问题，`exit` 退出。

### 2. 切换搜索 provider（L1）

```bash
# 默认 tavily；切换为 bing：
export WEB_PROVIDER=bing        # Windows PowerShell: $env:WEB_PROVIDER = "bing"
# Tavily key 读取：环境变量优先，回退项目根目录 .env（已 gitignore）
python rag_chat.py
```

> provider 抽象见 `web_providers.py`：`WebSearchProvider.search(query, top_k) -> list[dict]`，
> 每条含 `title/url/snippet/content/published_at/source`。主 provider 失败（缺 key/网络/429/5xx）自动降级 Bing。
> key 打码规则：日志/验收输出只显示前 8 位 + 后 4 位。

### 3. 联网上下文预算（L2）

```bash
# 默认 13000 token（≈40% of 32k 窗口），可按需调：
export WEB_CONTEXT_BUDGET_TOKENS=16000   # Windows PowerShell: $env:WEB_CONTEXT_BUDGET_TOKENS = "16000"
```

### 4. L3 混合路由（三段置信度分工，替换阶段一）

```bash
# 三段路由:
#   best <  MIX_LOW            -> 直接联网（D 段纯阈值直连，无 LLM 判定）
#   best >  MIX_HIGH           -> 直接知识库（A 段纯阈值直连，无 LLM 判定）
#   MIX_LOW <= best <= MIX_HIGH -> 灰区判定（默认 tool calling）
# 灰区判定: MIX_MODE=tool_calling（默认）用 qwen 原生 tool calling（中性格式约束），
#   异常/超时/第三种输出 -> 回退 kb_sufficient -> 再失败保守走知识库；
#   MIX_MODE=kb_sufficient 即切回阶段一判定方式。
export MIX_LOW=0.55                 # 灰区下界（默认 0.55）
export MIX_HIGH=0.75                # 灰区上界（默认 0.75）
export MIX_MODE=tool_calling        # 灰区判定方式（tool_calling / kb_sufficient）
export GREY_TC_TIMEOUT=15           # 灰区 tool calling 判定超时秒数（默认 15）
export LLM_TIMEOUT=120              # 生成/改写/二级判定调用超时秒数（默认 120）
# Windows PowerShell 写法：$env:MIX_LOW = "0.55"（其余同理）
# 旧配置 SECONDARY_ROUTE_ENABLED / SECONDARY_ROUTE_CHECK_MAX 已不再参与路由，
# 常量保留仅为兼容历史验收脚本（l3_acceptance/l3_compare*.py）。
```

### 5. 重新构建索引（知识库更新后）

```bash
python build_index.py                 # 全量构建（约2小时）
python build_index.py --sample 50     # 仅前50个文件（调试）
python build_index.py --reset         # 清空缓存重来
```

### 6. 运行验收

| 脚本 | 用途 | 是否需要 Tavily key |
|---|---|---|
| `l1_acceptance.py` | L1：双 provider 对比 / 失败降级 / 中文查询 / 配置切换（mock 错 key 测降级） | 否（降级路径覆盖） |
| `l1_retest.py` | L1：Tavily 真实 key 实测（条数/相关性/延迟/content 长度 + 请求参数打码） | 是 |
| `l2_acceptance.py` | L2：联网上下文结构打印 / 3 条实时问题 groundedness | 是 |
| `l3_acceptance.py` | L3 阶段一：8 条问题路由正确率（L2 vs L3）/ 误杀率 / 端到端回退 | 是（回退联网时） |
| `tool_call_stability.py` | L3 阶段二前置：qwen tool calling 稳定性（10 条问题，成功率/参数/崩坏/幻觉） | 否（纯 Ollama 本地） |
| `l3_compare.py` | L3 阶段二并存对比 v1：20 条四类问题，无约束 tool calling（结论：延迟超限停用） | 否（纯 Ollama 本地） |
| `l3_compare_v2.py` | L3 阶段二并存对比 v2：28 条 + 输出约束重估（发现 D 类 0/7 约束副作用，边界情况待定夺） | 否（纯 Ollama 本地） |
| `l3_mixed.py` | 混合路由验证（替换前）：灰区 14 条 tool calling 判定（中性约束，14/14=100%，延迟差 0.35s） | 否（纯 Ollama 本地） |
| `l3_replace_e2e.py` | **替换后验证1**：端到端 28 条完整走 ask()（路由路径分布 / 拼合假设 / 边界 best） | 是（灰区WEB+D段联网时） |
| `l3_fallback_test.py` | **替换后验证2**：fallback 三故障主动触发（格式崩坏/超时/第三种输出，mock 12/12） | 否（纯 mock） |
| `l3_blind_test.py` | **收尾盲测**：20 条全新问题（未参与调参）端到端，20/20=100%，贴边 2 条记录 | 是（灰区WEB+D段联网时） |
| `l3_mode_kbsuff_test.py` | **收尾补测**：MIX_MODE=kb_sufficient 配置切换（灰区 14 条 11/14，与阶段一一致） | 是（回退联网时） |
| `quick_notool_test.py` | 低成本验证：prompt 约束后"不该调"延迟（5 条，5.25s→0.73s） | 否（纯 Ollama 本地） |
| `regression_clean.py` | 补1/补2 回归：清洗规则结果不变 / 歧义日期标注 | 否 |
| `e2e_test.py` | 端到端问答冒烟 | 是 |

## 工作流程

```
用户问题
   │
   ▼
bge-m3 向量化
   │
   ▼
FAISS 检索 top-5（余弦相似度）
   │
   ├── best < 0.55 ──► 直接联网兜底（D 段纯阈值直连）
   │
   ├── best > 0.75 ──► 直接知识库命中（A 段纯阈值直连）：拼入上下文 ──► 生成答案
   │
   └── 灰区 [0.55, 0.75] ──► qwen 原生 tool calling 判定
                              ├── 决定调用 search ──► 联网兜底
                              └── 输出 NO_TOOL ──► 知识库生成（标注来源）
                              （判定异常/超时/格式不符 → 回退 kb_sufficient
                                二级判定 → 仍失败则保守走知识库）
                                                    │
                             ┌──────────────────────┴──────────────────┐
                             ▼ (主 provider，默认 tavily)                ▼ (自动降级 bing)
                        Tavily REST API                          Bing RSS (mkt=zh-CN&setlang=zh)
                             │                                      │
                             └──────────────────┬──────────────────┘
                                                ▼
              LLM 改写搜索词 → 相关性判定 → L2 正文清洗/截断/模板
                    （title + url + published_at + 正文片段，去重，预算内）
                                                ▼
                                        注入上下文 ──► qwen2.5 生成答案（标注来源）
```

## 验收结果（实测）

| 场景 | 提问 | 最高相关度 | 走向 | 结果 |
|---|---|---|---|---|
| 知识库检索 | 什么是递归？ | 0.71 | 知识库 | 命中 CS50/数据结构笔记，答案完整，标注 5 个来源 |
| 联网搜索兜底 | 2026年诺贝尔物理学奖颁给了谁？ | 0.46 | 联网搜索 | 正确回答 Francis Halzen（冰立方中微子），标注 6 个来源 |
| L1 中文对比 | 2026冬奥会举办城市 | — | Tavily | **Tavily 全中**（百度文库/维基/百度百科/Olympics.com 均米兰-科尔蒂纳丹佩佐）；**Bing 全偏**（2026年百科/节假日）——Tavily 中文明显优于 Bing，无需加语言路由 |
| L1 降级 | — | — | mock 错 key | Tavily 401 → 自动降级 Bing，返回 5 条（PASS） |
| L2 联网正文 | 华为mate70发布时间 | 0.46 | Tavily 正文 | **精准回答 2024年11月26日发布**（L1 Bing 时代此问题搜不到）；groundedness 有依据 |
| L3 回退 | 2026年10月AI大模型最新发布 | 0.61 | 命中→判定不足→回退联网 | 答案详实（GPT-6 全量推送/Gemini 4 Argon/Claude Haiku 5.5 等 6 类进展），全部有来源 |

### L3 阶段一验收数据

| 指标 | 结果 |
|---|---|
| 路由正确率 | **L2（纯阈值）62% → L3（二级判定）100%**（8 条案例） |
| 灰区纠正 | 3/3 条灰色地带问题被正确纠正（0.606/0.667/0.581 均判定"不足以回答"→回退联网） |
| 误杀率 | **0/4**（强制判定模式下，递归 0.714 / PV 操作 0.675 / 进程线程 0.796 / TCP 三次握手 0.791 均判定"足以回答"，无误回退） |
| 端到端 | 灰区问题完整链路：命中(0.61) → 判定不足 → 回退 → Tavily 正文 → 详实答案+来源 |

> **验收局限（在案记录）**：误杀率 0/4 含金量有限——4 条知识库内问题相似度为 0.714 / 0.675 / 0.796 / 0.791，
> 其中 3 条高于 `SECONDARY_ROUTE_CHECK_MAX=0.68` 直接跳过二级判定，误杀率天然为 0；强制判定模式（CHECK_MAX=∞）
> 的验证方向正确，但样本仍集中在强信号区，**未覆盖 0.55–0.68 真正的边界带**（误杀风险在"知识库确实有答案、
> 但相似度仅 0.58 左右"的问题上）。样本量 8 条偏小，结论待扩充验证。

## 关键设计决策

1. **Embedding 模型选 bge-m3**：知识库为中文内容，bge-m3 多语言检索效果优于 nomic-embed-text。
2. **索引用内积 + L2 归一化**：等价于余弦相似度，阈值可解释（0~1）。
3. **相关度阈值 0.55**：实测真实命中的相关度 >0.6，弱相关碎片 0.45~0.5；0.55 能正确区分"知识库有答案"与"需要联网"。
4. **搜索后端抽象（L1）**：定义 `WebSearchProvider` 接口，Bing RSS 与 Tavily API 都是它的实现；`WEB_PROVIDER` 环境变量控制主 provider（默认 tavily，可切 bing），上层问答逻辑零改动。主 provider 失败自动降级到 Bing，再失败返回空列表由上层处理，不编造结果。
5. **Tavily 接入（L1）**：官方 REST API 直连（不装 SDK，复用已装 requests）；key 从 `TAVILY_API_KEY` 读取——环境变量优先，回退项目根 `.env`（自写解析，不引 dotenv）；超时 10s；429/5xx 指数退避重试最多 2 次；4xx 直接失败并降级。日志/输出 key 一律打码（前 8 后 4），`.env` 已入 `.gitignore`。
6. **Bing RSS 保留为 fallback（L1）**：迁移原实现行为不变，仅补 `mkt=zh-CN&setlang=zh` 参数（可配，默认中文），并新增解析 RSS `<pubDate>` 作为 `published_at`。
7. **LLM 查询改写**：先让 qwen 把口语化问题改写成新闻/百科常用搜索表达（如"2024巴黎奥运会金牌榜第一名"），比纯规则精简更通用。
8. **搜索词自动纠偏（通用规则，非针对个例）**：实测发现 Bing 中文对"年份开头"的查询有系统性召回缺陷（"2024奥运会金牌榜"会返回2024年百科/统计年鉴等泛化页）。因此：
   - 年份开头自动移到末尾（"2024巴黎奥运会金牌榜" → "巴黎奥运会金牌榜2024"），实测召回精准；
   - 中文结果不理想时自动追加英文兜底词；
   - **LLM 相关性判定**：让 qwen 判断搜索结果是否真能回答问题，不相关就自动换下一个候选词，全部不相关才放弃。实测"2026冬奥会举办地"经此机制从泛化页自动切换到精准结果（米兰-科尔蒂纳丹佩佐）。
9. **L2 走规则 A（轻量路线，不实现 fetch_page）**：判定依据——Tavily 3 条实时查询平均 content 长度 1110/1312/1096 字符（均远超 300），3/3 覆盖核心信息，噪声可清洗。因此直接依赖 provider 的 content，做清洗+截断+模板升级。
10. **正文清洗（L2）**：逐行处理——去 HTML 标签、导航/版权/图片标注/工具链接/投诉信息等噪声行、非中英文残留行（俄语等）、孤立符号行；保留段落结构。
11. **发布时间兜底（L2）**：Tavily basic 深度不返回 `published_date`，改从正文提取（"发表时间：06/10/2026" 或 "2026年11月18日"）。
12. **上下文预算（L2）**：`WEB_CONTEXT_BUDGET_TOKENS` 环境变量可配，默认 13000 token（≈40% of 32k 窗口）；按 1.5 字符/token 换算字符预算；正文按段落截断（不从句子中间切），多条结果按 URL 去重。
13. **L3 阶段一：二级判定路由（不引入 tool calling）**：知识库命中且相关度落在灰区 `[0.55, 0.68)` 时，由 LLM 判断 top-5 片段是否真能回答该问题（复用 is_relevant 的"LLM 判相关性"思路，判定对象改为知识库片段 vs 问题）；判定"不足以回答"则回退联网。配置：`SECONDARY_ROUTE_ENABLED`（默认 1）/ `SECONDARY_ROUTE_CHECK_MAX`（默认 0.68，高置信跳过省一次 LLM 调用）。判定失败时保守走知识库（避免误回退）。不改 rewrite_query / is_relevant / provider 接口，只加路由层。
14. **断点续传 + 自动重试**：155159 块向量化约 2 小时，过程中 Ollama 偶发 socket 错误，脚本每批落盘、失败指数退避重试，中断可续传。
15. **中文路径兼容**：FAISS 的 C++ 层无法打开含中文的路径，改用 Python 层 `serialize_index` 读写。
16. **L3 混合路由（替换阶段一，✅ 已实施）**：路由分三段——`<MIX_LOW(0.55)` 直连 WEB、`>MIX_HIGH(0.75)` 直连 KB、`[0.55, 0.75]` 灰区用 qwen 原生 tool calling 判定（`SEARCH_TOOL_DEF` + 中性格式约束 `TC_CONSTRAINT`，无"能用自身知识就 NO_TOOL"语义诱导）。降级链：tool calling 异常/超时/第三种输出 → `_kb_suff_route`（kb_sufficient）→ 再失败保守 KB。判定对象与生成共用主 LLM，灰区判定用独立 `ChatOllama(request_timeout=GREY_TC_TIMEOUT)` 防止挂起。`MIX_MODE=kb_sufficient` 切回阶段一。`ask()` 增 `_cache` 参数复用预加载索引。

## 已知局限（报告可写）

- **LLM 知识截止**：qwen2.5:7b 训练数据有截止时间，当搜索无结果且问题涉及训练截止后的事实时，模型可能给出"尚未发生"的错误判断（如 2024 奥运会），无法完全避免。
- **Bing 中文索引覆盖**：部分实体（如特定年份的金牌榜）在 Bing 中文索引中无有效内容，搜索兜底会失败并如实告知"信息来源有限"。
- **Bing 中文召回缺陷（L1 实测确认）**：加 `mkt/setlang` 后，"2026冬奥会举办城市"裸查询仍返回 2026 年百科/节假日泛化页——`mkt/setlang` 不解决年份开头缺陷，依赖候选词切换 + LLM 相关性判定纠正（真实链路已验证）。
- **硬阈值无二级判定（L0 薄弱点 2，L3 阶段一已解）**：相关度 ≥0.55 即走知识库，"2026年10月AI大模型最新发布"（0.60）被知识库截获、答不出实时信息。L3 阶段一加入二级判定：知识库命中后由 LLM 判断 top-5 片段是否真能回答，不足则回退联网（见下"L3 阶段一"）。
- **正文清洗边界（补1）**：`_CLEAN_RULES` 规则列表基于有限样本（RFI/中华网/合成噪声等）迭代，**未覆盖所有网页结构**；遇到新噪声时向列表追加一条带注释的规则即可（类型 `line` 行内删除 / `drop` 整行丢弃），主逻辑 `clean_content` 不需要改。已知未覆盖样例：部分页面的"您尝试访问的内容不存在或不再可用"等错误提示行、特殊排版（表格/列表项前缀）。
- **日期格式歧义（补2）**：`extract_pub_from_text` 对 `06/10/2026` 这类数字日期不做月/日顺序推断（美式 MM/DD 与欧式 DD/MM 相反，无法确定）——保守策略：保留原始字符串并标注"（日期格式不确定）"，不做强行解析，避免"旧闻当新闻"；仅 `YYYY年M月D日` 中文格式直接采用。已知影响：RFI 等欧式站点日期会带"不确定"标注，不参与时效性推理。
- **延迟 trade-off（补3）**：实测 Tavily 单次 2.3–3.1s vs Bing 0.5s（RSS 更快）。当前默认 Tavily 是**拿延迟换召回质量**（content 完整、中文召回优、可降级）。若后续优化交互体验，候选方案（**L4，现在不做**）：按查询类型路由——英文/技术类查询走 Tavily、简单事实类查询走 Bing，减少常态延迟。
- **盲测未覆盖模糊问题（收尾记录）**：盲测 20 条问题边界清晰（每条明确对应一个类别），**未覆盖"用户自己也说不清该走知识库还是联网"的模糊问题**；真实使用中此类问题的路由可靠性未经测试。盲测样本同时未覆盖多轮追问、反问、口语化残缺问题等真实输入形态。

## L4 候选清单（现在不做，按需挑）

| 候选 | 说明 | 触发条件 |
|---|---|---|
| 扩充 L3 验收样本（目标 20+ 条，重点覆盖 0.55–0.68 边界带） | 当前 8 条样本的误杀率结论局限于高置信区；需在边界带验证"知识库有答案但相似度 ~0.58"的误回退风险 | 需要更可信的路由正确率/误杀率结论时 |
| 按查询类型路由（英文/技术→Tavily，简单事实→Bing） | 减少常态延迟（Tavily 2.3–3.1s vs Bing 0.5s） | 交互体验优先、接受少量召回质量下降时 |
| 多引擎并行搜索（SearXNG + Tavily）后合并去重 | 提升召回广度 | 长尾/冷门查询召回不足时 |
| 混合重排序（BM25 + 语义 + cross-encoder） | 对抓回的正文分块后重排再喂模型 | 正文质量/相关度排序不理想时 |
| 搜索能力封装 MCP Server | 本地 RAG 与联网搜索作为并列 MCP 工具接入 | 需要对外暴露工具接口时 |
| 联网结果 groundedness 自动检查 | 答案断言能否在抓到的正文中找到依据 | 需要自动化的幻觉检测时 |
| **L3 阶段二：Qwen2.5 tool calling + ReAct** | **混合路由已实测（结论见数据）**：灰区（0.55–0.75）tool calling 判定 14/14 = 100%、D 类无误伤、延迟差 0.35s——按停止条件**采纳混合路由**，替换阶段一的路由分支（替换前等用户确认）。未接 ReAct 多轮循环 | 用户确认采纳混合路由后实施替换 |

### L3 阶段二前置：tool calling 稳定性验证（实测）

验证脚本 `tool_call_stability.py`（只测能力，不改架构、不接 ReAct）。构造最小工具 `search(query, top_k)`，10 个问题（5 该调 / 5 不该调）测 `bind_tools` 结构化输出：

| 指标 | 结果 |
|---|---|
| 结构化输出成功率 | **10/10 = 100%**（该调则调且工具名正确、不该调则不调） |
| 参数正确率 | **10/10 = 100%**（query 均为非空中文关键词） |
| 格式崩坏 | 0 次 |
| 幻觉工具名 | 0 次 |
| query 参数缺失/空 | 0 次 |
| 平均延迟 | 该调 ~0.9s / 不该调 ~4.7s（纯文本生成） |

**结论（按约定规则 ≥8/10 判定）**：可做最小 tool calling 原型——但先做**并存对比**（同一组问题分别跑阶段一判定 vs tool calling 判定，对比路由正确率与延迟），数据说话再决定是否替换；当前不替换 `kb_sufficient`。阶段二未正式实施。

### L3 阶段二并存对比 v2（重估：28 条 + 输出约束，实测）

问题集扩充至 28 条（每类 7 条）：A 高置信KB（0.750–0.791）/ B 低置信KB（0.589–0.675）/ C 灰区实时（0.560–0.620）/ D 无答案（0.447–0.495）。tool calling 带约束"只输出 tool call JSON 或 NO_TOOL，不输出任何解释性文本"。

| 类别 | 阶段一路由正确 | tool calling 路由正确 |
|---|---|---|
| A 高置信KB | 7/7 | **7/7**（无约束版曾误判二叉树遍历，约束后修复） |
| B 低置信KB | **4/7**（哈希表/LRU/零拷贝 3 条被误杀） | **7/7** |
| C 灰区实时 | 7/7 | **6/7**（PyTorch 0.607 被压成 NO_TOOL） |
| D 无答案 | 7/7 | **0/7** ← 约束致命副作用 |
| 总计 | 25/28 | 20/28 |
| **边界带（B+C）** | **11/14 (79%)** | **13/14 (93%)** |

**同口径延迟三场景**（KB = 判定+生成两次调用，生成基线共用 1.16s；WEB/C = 判定到发出搜索请求）：

| 场景 | 阶段一 | tool calling | 比值 |
|---|---|---|---|
| 该走 KB（A+B，同口径） | 1.34s | 1.44s | **1.07x** |
| 该走 WEB（D） | 0.00s（纯阈值直连，无 LLM） | 0.10s（但全判错，延迟比较无意义） | inf* |
| 判定不足（C） | 0.34s | 0.69s | **2.06x** |

> *WEB 场景 inf 为除零（阶段一 D 类纯阈值无 LLM 调用，0.00s），绝对值差异仅 0.1s。
> 同口径后 KB 场景延迟差距基本消失（1.07x）——证实此前 32.6x 是口径偏差，不是真实差距。

**关键副作用（实现归因修正）**：v2 脚本实现时把约束扩展成了"能用自身知识直接回答时输出 NO_TOOL"——**这是实现者加入的语义判断，不是原始格式约束**（原始约束只是"只输出 tool call JSON 或 NO_TOOL，不输出解释"）。该语义扩展诱导模型过度自信：D 类 7 条实时问题（世界杯冠军/NBA/苹果发布会/mate70/诺贝尔奖等）全部被模型认为"自身知识能答"而输出 NO_TOOL，**该联网却不联网，0/7 全灭**（无约束版 D 类 5/5 全对）。模型会把参数记忆里的过期/幻觉信息当作"能直接回答"。**归因：是语义扩展所致，不是格式约束/架构问题。**

**判定理由比对（B/C）**：B 类双方一致——阶段一"足以回答"、tool calling 输出 `NO_TOOL`（约束有效且正确）；C 类 6/7 一致，1 条（PyTorch 0.607）tool calling 误判 NO_TOOL。

**结论（按重新写死的停止条件）**：边界带正确率 tool calling 更高（93% vs 79%）✓，KB 延迟同口径 1.07x ✓，A 类未恶化 ✓，但**判定不足场景 2.06x 略超 2 倍**（2.0–2.1 边界）且 **D 类 0/7 系统性误判**（停止条件未覆盖的新发现）→ **边界情况：列出数据停下等用户定夺**。约束方案本身已证实不可直接采用（D 类副作用），如需推进 tool calling 应换思路（如：只对"判定是否调用"的 prompt 加约束、但对"已决定调用"的结果不加压；或用 stop token 替代 prompt 约束；或对 D 类低置信区间保留纯阈值直连、仅在 0.55–0.68 灰区启用 tool calling 判定）。

### L3 阶段二：混合路由（✅ 已实施替换，三项验证全过）

**方案（用户确认）**：路由分三段，按置信度区间分工（不是二选一）：
- `<0.55` 直接联网（纯阈值直连，不改 A/D 行为）
- `>0.75` 直接知识库（纯阈值直连，不改）
- `0.55–0.75` 灰区启用 **tool calling 判定**（中性格式约束，**去掉"能用自身知识就 NO_TOOL"语义判断**，只保留"输出 JSON 或 NO_TOOL，不输出解释"）；`kb_sufficient` 作 fallback（判定异常/超时/格式崩坏时回退，再不行保守走知识库）
- 配置项：灰区上下界可配（`MIX_LOW`/`MIX_HIGH` 默认 0.55/0.75）、灰区判定方式可配（`MIX_MODE` 默认 tool_calling，可切 kb_sufficient 对比）

**已实施**：`rag_chat.py` 的 `ask()` 路由分支已替换为混合路由（`_grey_route_judge` + `_kb_suff_route`），A/D 两段行为不变；`SECONDARY_ROUTE_*` 保留为兼容常量不再参与路由；`ask()` 支持 `_cache` 参数复用预加载索引（批处理/测试用）。实现见 [关键设计决策](#关键设计决策)。

**验证 1：端到端 28 条实跑（`l3_replace_e2e.py`，完整走 ask()，真实联网）**：

| 类别 | 预期 | 实际路由路径 | 正确 |
|---|---|---|---|
| A 高置信（>0.75） | KB | 7/7 全部 **A段KB直连**（不经过灰区判定）✓ | 7/7 |
| D 无答案（<0.55） | WEB | 7/7 全部 **D段WEB直连**（不经过灰区判定）✓ | 7/7 |
| B 灰区低置信 | KB | 7/7 全部 **灰区tool calling→KB(NO_TOOL)** | 7/7 |
| C 灰区实时 | WEB | 7/7 全部 **灰区tool calling→WEB(调search)** | 7/7 |
| **总计** | — | — | **28/28 = 100%** |

灰区 14 条全部走 tool calling（fallback 0 条触发，与 `l3_mixed.py` 一致）；总正确率与拼合数据 28/28 一致，**拼合假设成立，无漏洞**。

**验证 2：fallback 三故障主动触发（`l3_fallback_test.py`，mock 12/12 通过）**：

| 故障 | 回退 kb_sufficient 且判定生效 | kb_sufficient 也失败→保守 KB | 降级日志 |
|---|---|---|---|
| ①格式崩坏（非法 JSON content） | ✓（足以→KB / 不足→WEB） | ✓ 保守 KB 不误走 WEB | ✓ |
| ②超时（TimeoutError） | ✓ | ✓ | ✓ |
| ③第三种输出（非 TOOL_CALL 非 NO_TOOL） | ✓ | ✓ | ✓ |

> 该验证发现并修复一个防御缺口：`_kb_suff_route` 原本未兜底 `kb_sufficient` 自身抛异常的情况（真实 `kb_sufficient` 内部已捕获，但防御应在路由层再兜一层），已加 try/except 保守走 KB。

**验证 3：灰区边界脆弱性（实测）**：A 类最低样本"什么是数据库事务？"实跑 best = **0.7503**，距上界 0.75 仅 **0.0003**——**贴边**。向量库任何微小变化（重建索引、换 embedding 模型、样本扩充）都可能把它推入灰区。**当前不改上界**（灰区判定能处理它），但需注意：**0.75 上界基于 28 条样本，A 类最低样本 0.750 贴边，边界余量小；样本扩充或向量模型更换后需重校准上下界。**

**结论**：三项验证全过 → 混合路由替换完成，系统处于最终形态。

**盲测（收尾，`l3_blind_test.py`，20 条全新问题未参与调参）**：28 条同时是调参集与验证集——0.55/0.75 边界按这 28 条的相似度分布确定，再用这 28 条验证 28/28，存在乐观偏差。为排除偏差，造 20 条全新问题（四类各 5，未参与任何调参）端到端复验：

| 类别 | 预期 | 盲测结果 | 实际路由路径 |
|---|---|---|---|
| A（应 KB） | KB | 5/5 ✓ | 1 条直连（中序遍历 0.753），4 条实际落灰区但 tool calling 正确判 KB |
| B（灰区应 KB） | KB | 5/5 ✓ | 1 条直连（分页存储 0.757），4 条灰区判 KB |
| C（灰区应 WEB） | WEB | 5/5 ✓ | 1 条实际 <0.55 直连 WEB（诺贝尔化学奖 0.466），4 条灰区判 WEB |
| D（应 WEB） | WEB | 5/5 ✓ | 全部 D 段直连 |
| **总计** | — | **20/20 = 100%** | **B/C 灰区 10/10 = 100%** |

**盲测贴边条目（2/20）**：二叉树中序遍历 best=**0.7534**（距上界 0.0034）、2026 双十一成交额 best=**0.5518**（距下界 0.0018）——本轮均判对，但边界脆弱性在全新样本上复现，**上下界重校准需求依然成立**。

**如实记录（最终口径）**：20 条未参与调参的盲测样本全部路由正确（A 5/5、B 5/5、C 5/5、D 5/5），但**样本量有限，不代表真实场景 100% 正确**；盲测中 2 条贴边（中序遍历 0.7534 距上界 0.0034、双十一 0.5518 距下界 0.0018）复现了边界脆弱性，上下界需在样本扩充后重校准。**调参集 28/28 为自测口径，盲测 20/20 为独立口径，两者都不能作为最终正确率承诺。**

**补测：MIX_MODE=kb_sufficient 配置切换（`l3_mode_kbsuff_test.py`）**：验证 1/2 均在默认 tool_calling 下跑过，补测切换路径——灰区 14 条 **11/14 = 79%**，与阶段一灰区参考**完全一致**（B 4/7 + C 7/7，哈希表/LRU/零拷贝误杀条同源），配置切换无损坏，`MIX_MODE=kb_sufficient` 正确复现阶段一行为。

**"保守走 KB"方向选择的理由（验证 2 遗留）**：kb_sufficient 异常时保守走 KB——**理由是异常概率极低且 KB 延迟更低；代价是灰区 C 类问题（该联网的）在极端情况下会误答**（拿知识库硬答而非联网取真实信息），可接受。此选择对 B 类（该走 KB）正确、对 C 类（该走 WEB）不完美，属知情取舍，不改为 WEB 兜底。

**复杂度成本（替换后新增，务必知悉）**：混合路由相比阶段一，从"两段阈值 + 一个判定"变成"三段路由 + tool calling 判定 + kb_sufficient fallback + 故障降级"：
- **新增故障面**：tool calling 格式崩坏、超时、第三种输出（均已有降级路径，但每个都是新检查点）；
- **调试难度上升**：路由错误要从四层（阈值 / tool calling / fallback / 保守回退）逐层排查，`[路由]` 打印行是主要定位手段；
- **维护成本**：灰区上下界需随样本重校准（见验证 3 贴边问题），tool calling 约束 prompt 需随 qwen 模型版本重测（格式约束有效性可能随版本漂移）。
- 收益是路由正确率从阶段一 89% → 混合路由 100%（28 条样本口径），代价是上述维护复杂度。此段写给未来的维护者，勿让"100%"掩盖"更难维护"。

## 部署过程（对应课设报告素材）

1. 安装 Ollama，`ollama pull qwen2.5:7b-instruct-q4_K_M`、`ollama pull bge-m3`
2. `pip install -r requirements.txt` 安装 Python 依赖
3. `git clone https://github.com/LINJIANG12/video2book-courses.git kb_source` 准备知识库源文档
4. `python build_index.py` 构建 FAISS 索引
5. `cp .env.example .env` 并填入 `TAVILY_API_KEY`（可选，不填自动降级 Bing）
6. `python rag_chat.py` 问答验证
