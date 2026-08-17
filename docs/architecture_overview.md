# CityU-Agent 系统架构概述

> 版本：v1（对应 `agent/graph.py` 中的 "rule-based RAG graph (v1)"）
> 范围：从用户提问（User Query）到最终回答（Final Response）的完整链路，
> 覆盖 LangGraph 工作流、RAG pipeline、数据模型与端到端执行案例。

---

## 1. 系统架构总览

CityU-Agent 是一个面向香港城市大学（CityUHK）授课型硕士课程的问答/推荐助手。
它不是一个让 LLM 自由调用工具的 ReAct agent，而是一个**规则路由 + 检索增强生成（RAG）**的确定性图（graph）：
LLM 只在最后一个节点负责"根据检索到的证据组织语言生成答案"，而"该查哪个索引"由一个基于关键字的规则路由器决定。

### 1.1 组件分层

| 层 | 模块 | 文件 | 职责 |
|---|---|---|---|
| 入口 | CLI | `agent/cli.py` | one-shot / REPL 交互，调用 `app.invoke()` |
| 图定义 | Graph + State | `agent/graph.py` | 用 LangGraph 定义节点、边、条件路由，编译为 `app` |
| 节点层 | Nodes | `agent/node/*.py` | router / 三个 retriever / generator / citation |
| RAG 层 | 检索与解析 | `rag/*.py` | 解析 markdown、构建文档、解析结构化字段、向量检索 |
| 存储层 | 向量库 | `rag/vectorstore/` | Chroma 持久化索引（3 个 collection） |
| 数据层 | 原始数据 | `data/markdown/`, `data/programmes.json` | 64 个课程 markdown 及其解析后的结构化 JSON |
| 提示词/模型 | LLM | `agent/node/answer_node.py` | `ChatOpenRouter(deepseek/deepseek-v4-flash, temperature=0)` |

### 1.2 从 User Query 到 Final Response 的完整流程

```
 User Query
     │
     ▼
 app.invoke({"query": ...})                agent/cli.py
     │
     ▼
 ┌─────────────────────────────────────────────────────────────┐
 │  router_node                                                 │
 │  classify_query(query)  →  intent ∈ {summary, metadata,     │
 │                             section}                        │
 └─────────────────────────────────────────────────────────────┘
     │  intent (写入 shared state)
     ▼  (conditional edge)
 ┌─────────────────────────────────────────────────────────────┐
 │  三条检索路径之一：                                          │
 │  • summary  → summary_retriever   (programme_summaries)     │
 │  • metadata → metadata_retriever  (结构化精确查找/元数据索引)│
 │  • section  → section_retriever   (programme_sections)      │
 └─────────────────────────────────────────────────────────────┘
     │  写入 state["evidence"] = [Evidence, ...]
     ▼
 ┌─────────────────────────────────────────────────────────────┐
 │  generator (answer_node)                                    │
 │  format_evidence(evidence) → prompt → LLM → state["answer"] │
 └─────────────────────────────────────────────────────────────┘
     │
     ▼
 ┌─────────────────────────────────────────────────────────────┐
 │  citation (citation_formatter)                              │
 │  answer + "Sources:" 块 → state["final_response"]           │
 └─────────────────────────────────────────────────────────────┘
     │
     ▼
 Final Response（answer + 带 id 锚点的来源列表）
```

一个查询只会走**一条**检索路径（图是单路径 DAG，不是循环 agent）。每个节点只读/写共享 state，LLM 仅出现在 generator 节点。

---

## 2. LangGraph 工作流分析

### 2.1 Graph 结构（Graph Structure）

`agent/graph.py` 中的 `build_graph()` 用 `StateGraph(AgentState)` 声明图，然后编译为 `app`：

```
START
  │
  ▼
router  (classify_query: summary | metadata | section)
  │
  ├── summary  ──► summary_retriever  ──┐
  ├── metadata ─► metadata_retriever ───┼──► generator ──► citation ──► END
  └── section  ──► section_retriever ───┘
```

关键点：

- **6 个节点**：`router`, `summary_retriever`, `metadata_retriever`, `section_retriever`, `generator`, `citation`。
- **无循环**：单次查询严格按 `router → 一个 retriever → generator → citation → END` 顺序执行。
- **无 ToolNode / 无 agentic loop**：没有 LLM 自主多步推理、没有 `bind_tools`、没有 `ToolMessage` 往返。

### 2.2 State 对象（State Object）

`AgentState` 是一个 `TypedDict`，是贯穿所有节点的唯一共享上下文：

```python
class AgentState(TypedDict):
    query: str            # 用户原始提问（入口写入）
    intent: str           # 路由结果（router 写入）
    evidence: list        # 检索到的证据列表 [Evidence, ...]（retriever 写入）
    answer: str           # LLM 生成的正文（generator 写入）
    citations: list       # 结构化引用列表（citation 写入）
    final_response: str   # 最终输出（citation 写入）
    programme_id: str     # 解析出的课程 id（仅 metadata 路径写入，如 "P66"）
    programme_name: str   # 解析出的课程名（仅 metadata 路径写入）
```

注意：`programme_id` / `programme_name` 只有 metadata 精确查找路径会写入，且 `programme_name` 下游（citation 节点）并未消费——citation 自己通过 `get_programmes()` 重建 id→name 映射，因此这两个字段目前主要用于调试和测试断言。

### 2.3 Nodes（节点）

| 节点名 | 文件 | 输入（读 state） | 输出（写 state） | 核心逻辑 |
|---|---|---|---|---|
| `router` | `agent/node/router_node.py` | `query` | `intent` | 调 `rag.router.classify_query` |
| `summary_retriever` | `agent/node/summary_retriever_node.py` | `query` | `evidence` | `retrieve_summary(query, k=5)` → 5 条 `Evidence` |
| `metadata_retriever` | `agent/node/metadata_retriever_node.py` | `query` | `evidence`, `programme_id`, `programme_name` | 结构化精确查找；失败则向量回退 |
| `section_retriever` | `agent/node/section_retriever_node.py` | `query` | `evidence` | `retrieve_section(query, k=5)` → 5 条 `Evidence` |
| `generator` | `agent/node/answer_node.py` | `query`, `evidence` | `answer` | `format_evidence` + `SystemMessage`/`HumanMessage` → LLM |
| `citation` | `agent/node/citation.py` | `answer`, `evidence` | `citations`, `final_response` | 组装 `Sources:` 块 |

### 2.4 Edges 与 Conditional Routing

```python
graph.add_edge(START, "router")

graph.add_conditional_edges(
    "router",
    lambda state: state["intent"],
    {
        "summary":  "summary_retriever",
        "metadata": "metadata_retriever",
        "section":  "section_retriever",
    },
)

graph.add_edge("summary_retriever", "generator")
graph.add_edge("metadata_retriever", "generator")
graph.add_edge("section_retriever", "generator")
graph.add_edge("generator", "citation")
graph.add_edge("citation", END)
```

- **条件路由**只有一个分支点：`router` 之后，根据 `state["intent"]` 的值从映射表中选一个 retriever。
- 三个 retriever **汇聚**到同一个 `generator` 节点（`add_edge` 三条汇聚边），下游共享 generator + citation。
- 路由函数是一个纯 lambda（读 `state["intent"]`），不调用 LLM。

### 2.5 Message Passing / Context 在节点间的传输

本图不使用 LangChain 的 message list 在节点间传递，而是采用 **LangGraph 的 state-reducer 语义**：

1. 每个节点函数接收完整 `state`，返回一个**部分 dict**（只包含自己要更新的字段）。
2. LangGraph 运行时把返回值**合并**进共享 state（`AgentState` 字段默认 last-write-wins，非 append）。
3. 下游节点从合并后的 `state` 读取上游写入的字段。

因此 context 的流动是：

```
state["query"]                     # CLI 入口注入，全图只读
        │
state["intent"]                    # router 写，供 conditional edge 消费
        │
state["evidence"]                  # retriever 写，供 generator 与 citation 消费
        │
state["answer"]                    # generator 写，供 citation 消费
        │
state["citations"] + ["final_response"]  # citation 写，最终输出
```

> `HumanMessage` / `SystemMessage` 只在 `generator` 节点**局部**构造，不进入 state——这是"消息在节点间传输"与本设计最大的区别：节点间传的是结构化字段（尤其 `evidence: list[Evidence]`），而不是聊天消息列表。

### 2.6 Agent 如何决定什么时候调用 RAG

- 决策点是**确定性的规则路由器** `rag/router.py::classify_query`，不是 LLM 自主决策。
- 三个意图，按优先级顺序匹配（**metadata 优先**，因为事实字段信号最强）：

  ```python
  METADATA_KEYWORDS = ["fee", "tuition", "deadline", "duration", "credit", "mode"]
  SUMMARY_KEYWORDS  = ["recommend", "suggest", "suitable", "which programme",
                       "best programme", "what should i study", "choose"]
  # 命中 metadata 关键字 → "metadata"
  # 否则命中 summary 关键字 → "summary"
  # 否则 → "section"
  ```

- 命中即路由到对应索引，**始终会调用一次 RAG**（本设计没有"无需检索直接回答"的分支，也没有检索质量反馈/重试循环）。

### 2.7 工具调用是如何处理的（Tool Invocation Flow）

这是本系统与典型 "agentic RAG" 最大的区别：

- **运行时没有 LLM 工具调用**。没有 `@tool` 装饰器、没有 `bind_tools`、没有 `ToolNode`、没有 `ToolMessage` 往返。
- `test/callTool.py` 里演示了 `bind_tools + ToolMessage` 的实验性写法，但它**未接入图**，只是早期原型探索。
- 实际的"工具"是节点内直接调用的普通 Python 函数：

  | 函数 | 调用位置 | 作用 |
  |---|---|---|
  | `classify_query` | router 节点 | 决定检索意图 |
  | `extract_programme_ref` / `extract_field` / `find_programme` | metadata 节点 | 从自然语言里抽课程 id/名称与字段名 |
  | `value_to_text` / `_render_fee` | metadata 节点 | 结构化值 → 文本 |
  | `retrieve_summary` / `retrieve_section` / `retrieve_metadata` | 各 retriever 节点 | Chroma 向量检索 |
  | `model.invoke(messages)` | generator 节点 | 唯一一次 LLM 调用 |

  即：**"什么时候用哪个工具"由规则路由器预先决定，"工具结果"以 `Evidence` 对象塞进 state，最后交给 LLM 一次性生成**。

---

## 3. RAG Pipeline 分析

### 3.1 Data Ingestion（数据摄入）

摄入链路由两条独立的批处理脚本组成（注意是**两个**入口，不是一个统一管道）：

```
data/markdown/p02.md … p99.md   (64 个 scraped markdown)
        │  rag/parser.py
        ▼
data/programmes.json            (64 个结构化 Programme 对象)
        │
        ├─► rag/document_builder.py   → section 子文档（195 B + 143 C = 338 篇）
        ├─► rag/summary_builder.py    → 每课程 1 篇 summary（64 篇）
        └─► rag/metadata_builder.py   → 每课程 1 篇 metadata 文档（64 篇）
                    │
                    ▼
        rag/ingest.py          → 写 programme_sections + programme_summaries
        rag/metadata_ingest.py → 写 programme_metadata
```

`rag/parser.py` 是纯规则解析器，四个部分：

1. `ProgrammeParser` — 解析前置元数据（id、中英文名、Apply Now 链接）、`##` 结构化字段、学费、联系方式。
2. `OutlineExtractor` — 读取 `### Programme Outlines` 下的目录（TOC），作为正文分段的地面真值。
3. `SectionSegmenter` — 用 TOC 标题切分正文为 section chunk，并剥离文件末尾的 `†`/`^` 脚注。
4. `parse()` / `parse_all()` — 编排，输出 `data/programmes.json`。

文档按 schema（`docs/programme_schema.md`）分为三类：

- **A 类（结构化元数据）**：课程 id、学费、截止日期、学制、学分等 14 个字段——**不嵌入正文索引**，而是解析进 JSON，供精确查找。
- **B 类（检索知识）**：课程目标、入学要求、课程描述等——嵌入为 section 子文档。
- **C 类（可选信息）**：奖学金、认证、职业前景等——也嵌入，但打上 `optional: true` 标签。

### 3.2 Indexing（索引）

- 嵌入模型：`BAAI/bge-large-en-v1.5`（通过 `langchain_huggingface.HuggingFaceEmbeddings`）。
- 向量库：Chroma 持久化在 `rag/vectorstore/`，共 3 个 collection：

  | Collection | 文档数 | 内容 | 用途 |
  |---|---|---|---|
  | `programme_sections` | 338 | 每个课程每个 section 一篇 | 详细事实 QA |
  | `programme_summaries` | 64 | 每课程一篇 summary（目标+课程+亮点） | 课程推荐 |
  | `programme_metadata` | 64 | 每课程一篇结构化字段文档 | 精确事实的向量回退 |

- 每个 Document 携带丰富 metadata（见 §4.4），但检索时**未使用 metadata 过滤**。

### 3.3 Retrieval（检索）

检索由 `rag/retriever.py` 提供三个函数，均返回 `list[(Document, score)]`：

```python
retrieve_summary(query, k=5)  → programme_summaries.similarity_search_with_score
retrieve_section(query, k=5)  → programme_sections.similarity_search_with_score
retrieve_metadata(query, k=5) → programme_metadata.similarity_search_with_score
```

三条检索路径的实际行为：

1. **summary（推荐）**：直接对 `programme_summaries` 做语义检索 top-5。
2. **section（详细 QA）**：直接对 `programme_sections` 做语义检索 top-5。
3. **metadata（精确事实）**：**先做结构化精确查找**，命中才走向量库：
   - `extract_programme_ref(query)` 用 `P\d{2,3}` 正则找课程 id，找不到再用课程名最长匹配；
   - `extract_field(query)` 用关键字映射到字段 key（如 "tuition fee" → `tuition_fee`）；
   - `find_programme(ref)` 从 `data/programmes.json` 精确解析出该课程对象；
   - 命中则直接从 JSON 取值构造 `Evidence`（`score=1.0`，URL 拆到 `metadata["url"]` 不进 LLM content）；
   - **未命中**（查询里没有可解析的课程）才回退到 `programme_metadata` 向量检索。

> 检索是**懒加载**的：`rag/retriever.py` 用 `@lru_cache` 包裹 embedding 与 Chroma client，`import agent` 时不再加载 400MB 的 BGE 权重（见 §6）。

### 3.4 Generation（生成）

`agent/node/answer_node.py::generate_answer`：

1. `format_evidence(evidence)` 把每条 `Evidence` 渲染成 `[P66 | Tuition Fee]\n<content>` 块；
2. 拼接 `SystemMessage`（`ANSWER_PROMPT`）+ `HumanMessage`（`User query: … \n\n Retrieved context: …`）；
3. `ChatOpenRouter(model="deepseek/deepseek-v4-flash", temperature=0).invoke(messages)`；
4. 返回 `state["answer"]`。

生成是**单轮、无历史、无检索反馈**的：LLM 只看到当前查询 + 本次检索证据，温度 0（确定性）。

### 3.5 当前 RAG 设计的弱点

**检索质量**

1. **score 语义混乱（最重要的缺陷）**。Chroma 默认用 L2 距离，`similarity_search_with_score` 返回的 score 是**距离（越小越相关）**；但代码把它原样写进 `Evidence.score`，citation 又把它显示为 `Confidence: {score}`。结果：精确查找是 `1.0`（高=好），向量检索却是 `0.45` 表示最相关（低=好），两个量纲在同一份回答里并列，语义完全矛盾。没有做距离→相似度的归一化（如 `1/(1+dist)`），也没有设定相关性阈值。
2. **无重排 / 无混合检索**。纯稠密向量检索，没有 BM25/关键字召回，也没有 cross-encoder 重排。
3. **无元数据过滤**。Document 上明明有 `programme_id` / `section` / `category` / `optional`，但检索时全不用；`optional: true`（C 类可选信息）标签在 `document_builder` 里声明"供检索 boost/suppress"，实际上**没有任何代码消费它**——C 类文档与 B 类同等权重混在索引里。
4. **chunk 粒度过粗**。按"section"整块切分，Course Description 这类大 section 可能上千字符，向量表示被稀释，也无法定位到具体课程条目。
5. **无跨课程去重与结果聚合**。top-5 可能都是同一课程的不同 section，也可能散在不同课程，没有按 programme 聚合或多样性控制。

**路由**

6. **关键字路由脆弱**。`fee` 会误中 "feedback"（"fee" 是其前缀，单词边界无法排除）、`credit`/`mode` 语义重叠；多意图查询（"想学 AI，学费多少？"）只能命中 metadata，推荐诉求被丢弃（`test_graph.py` Test D 已记录此行为）。没有歧义消解、没有 LLM 辅助路由、没有多路径并行。
7. **metadata 字段抽取与路由关键字是两套独立、手写的关键词表**（`router.py` 与 `programme_resolver.py` 各自维护），易漂移。

**生成**

8. **无空证据处理**。检索返回 0 条时，generator 仍会带着空 context 生成，可能编造内容；没有 "I don't know" 兜底。
9. **引用是"后贴"而非"内联"**。LLM 生成正文时不被强制逐句溯源，`Sources:` 块由 citation 节点在事后拼接，正文与引用可能不对应。
10. **无对话历史 / 无多轮**。每个查询独立处理，无法追问或澄清。
11. **无评测闭环**。没有 retrieval recall/precision、answer faithfulness、citation accuracy 的自动化评测。

**工程结构**

12. **摄入管道分裂**。`rag/ingest.py`（sections + summaries）与 `rag/metadata_ingest.py`（metadata）是两个独立脚本，重建完整索引需要手动跑两次；两个脚本各自重复定义 `EMBEDDING_MODEL`、collection 名、`VECTOR_PATH`。
13. **死代码与不一致**。`rag/metadata_builder.py::build_field_document` 无人调用且其 docstring 声称"metadata retriever 使用它"（实际 retriever 内联重写了渲染逻辑）；`Evidence.to_citation()` 原本未被 citation 节点复用；`test/check_db.py` 引用了不存在的 `rag.retriever.vectorstore`（陈旧测试）。
14. **`AgentState.programme_name` 写了不读**。citation 用 `get_programmes()` 重新查名，state 里的 `programme_name` 形同虚设。

---

## 4. Data Model 分析

### 4.1 Programme（`data/programmes.json`，摄入中间产物）

```jsonc
{
  "programme_id": "P66",                 // 课程代码
  "name": "MSc Mechanical Engineering",  // 英文名
  "name_zh": "理學碩士(機械工程學)",       // 中文名
  "apply_now_url": "/en/.../apply-now",  // 申请链接
  "metadata": {                          // A 类结构化字段（14 个）
    "year_of_entry": "2026",
    "application_deadline": {"iso": [...], "raw": [...]},
    "mode_of_study": "Combined",
    "mode_of_funding": "Non-government-funded",
    "indicative_intake_target": "185",
    "minimum_no_of_credits_required": "30",
    "class_schedule": "...",
    "normal_study_period": "...",
    "maximum_study_period": "...",
    "mode_of_processing": "...",
    "tuition_fee": {"local": "...", "non_local": "...", "source": "http://..."},
    "programme_website": null,
    "intermediate_award": null
  },
  "contacts": [                          // 联系人（role/name/qualification/email/phone/fax）
    {"role": "Programme Leader", "name": "Prof LI You Fu", "email": "...", ...}
  ],
  "source_file": "data/markdown/p66.md",
  "outline": ["Programme Content", "Entrance Requirements", ...],  // TOC
  "sections": [                          // B/C 类正文分段
    {"title": "Entrance Requirements", "category": "B", "content": "...", "char_count": 1234}
  ],
  "footnotes": ["† Combined mode: ..."]
}
```

字段覆盖情况（64 个课程）：14 个 metadata 字段中 9 个 100% 覆盖，`class_schedule` 94%（缺 4 个），`programme_website` 78%（缺 14 个），`intermediate_award` 仅 16%（10 个）。

### 4.2 Evidence（运行时证据对象，`rag/evidence.py`）

```python
@dataclass
class Evidence:
    id: str            # 稳定锚点，如 "P66-tuition_fee"，用于引用/调试
    programme_id: str  # "P66"
    section: str       # 展示名，如 "Tuition Fee"
    content: str       # 给 LLM 看的文本（URL 等非 LLM 细节不进 content）
    score: float       # 检索得分（1.0=结构化精确命中）
    metadata: dict | None = None   # 额外字段，如 {"url": "..."}

    def render(self) -> str: ...      # 渲染为 LLM context 块
    def to_citation(self) -> dict: ...# id/programme_id/section/score
```

`Evidence` 是 retriever → generator → citation 之间流动的**统一单位**，取代了原始的 `Document`/dict。

### 4.3 AgentState（图共享状态）

见 §2.2。

### 4.4 Chroma Document 元数据（三类 collection 各自不同）

**programme_sections**（每 section 一篇）：

```python
{"id": "P66_Entrance Requirements", "programme_id": "P66",
 "programme_name": "...", "section": "Entrance Requirements",
 "category": "B", "source": "p66.md"}          # category=C 时额外 "optional": True
```

**programme_summaries**（每课程一篇）：

```python
{"id": "P66_summary", "type": "summary", "programme_id": "P66",
 "programme_name": "...", "source": "p66.md"}
```

**programme_metadata**（每课程一篇）：

```python
{"id": "P66_metadata", "programme_id": "P66", "programme_name": "...",
 "type": "metadata"}
```

### 4.5 数据流关系

```
data/markdown/p*.md  ──parser──►  Programme (JSON)
                                       │
              ┌────────────────────────┼─────────────────────┐
              ▼                        ▼                     ▼
        Section 文档             Summary 文档            Metadata 文档
        (sections[] 每项)        (Aims+Courses+Highlights) (metadata{} 全字段)
              │                        │                     │
              └────────────► Chroma 3 collections ◄──────────┘
                                      │  retrieve_* → (Document, score)
                                      ▼
                                 Evidence 列表（state["evidence"]）
                                      │  generator + citation
                                      ▼
                                 Final Response
```

---

## 5. Query 执行案例（End-to-End Example）

以 `"What is the tuition fee of MSc Mechanical Engineering?"` 为例，逐步追踪：

| 步骤 | 位置 | 发生的事 | state 变化 |
|---|---|---|---|
| 1 | `agent/cli.py` | `app.invoke({"query": "What is the tuition fee of MSc Mechanical Engineering?"})` | `query` 已设置 |
| 2 | `router_node` | `classify_query` 小写化后先查 metadata 关键字：命中 `"tuition"` 与 `"fee"` → 返回 `"metadata"` | `intent = "metadata"` |
| 3 | conditional edge | `lambda s: s["intent"]` 返回 `"metadata"` → 进入 `metadata_retriever` | — |
| 4 | `metadata_retriever_node` | ① `extract_programme_ref`：无 `P\d+` 代码，改走课程名最长匹配 → `{"programme_name": "MSc Mechanical Engineering"}`；② `extract_field`："tuition"/"fee" → `"tuition_fee"`；③ `find_programme` → 命中 `P66` 对象 | — |
| 5 | 同上 | `raw = P66.metadata["tuition_fee"]`（dict），`_render_fee` 拆成 `Local Students:\nHK$8,100 per credit\n\nNon-local Students:\nHK$8,100 per credit`（不含 Source），`url` 从 `raw["source"]` 取出 | 构造 `Evidence(id="P66-tuition_fee", programme_id="P66", section="Tuition Fee", content=…, score=1.0, metadata={"url": "…tpg/P66/index.htm"})` |
| 6 | 同上 | 返回 | `evidence=[该 Evidence]`，`programme_id="P66"`，`programme_name="MSc Mechanical Engineering"` |
| 7 | `generator` | `format_evidence` 渲染成 `[P66 | Tuition Fee]\nLocal Students: …`；拼 `SystemMessage(ANSWER_PROMPT)` + `HumanMessage(query + context)`；`deepseek-v4-flash` 生成正文 | `answer = "The tuition fee for MSc Mechanical Engineering is HK$8,100 per credit for both local and non-local students."` |
| 8 | `citation` | `_name_map()` 得 `P66 → MSc Mechanical Engineering`；`e.to_citation()` + 名/content 组装 `citations`；拼接 `Sources:` 块 | `citations=[{id:"P66-tuition_fee", …, score:1.0, content:…}]`；`final_response = answer + "\n\nSources:\n[P66-tuition_fee]\nMSc Mechanical Engineering > Tuition Fee\nConfidence: 1.0"` |
| 9 | `agent/cli.py` | `pretty_print` 打印 `intent`、evidence 数、`final_response` | 输出给用户 |

最终输出形态：

```
[intent] metadata  |  evidence 1  |  3.2s

[answer]

The tuition fee for MSc Mechanical Engineering is HK$8,100 per credit
for both local and non-local students.

Sources:

[P66-tuition_fee]
MSc Mechanical Engineering > Tuition Fee
Confidence: 1.0
```

**另两条路径的对比：**

- `"I am a computer science student interested in AI. Recommend suitable master programmes."`
  → `intent="summary"` → `summary_retriever` 对 `programme_summaries` 取 top-5 → top hit `P75 MSc Artificial Intelligence`。
- `"What are the entrance requirements of MA International Accounting?"`
  → `intent="section"` → `section_retriever` 对 `programme_sections` 取 top-5 → top hit `P02 Entrance Requirements`（score 0.45，注意这是 L2 距离，越小越相关，与 §3.5-1 的缺陷对应）。

---

## 6. 本次代码结构优化记录

| # | 文件 | 改动 | 原因 |
|---|---|---|---|
| 1 | `rag/retriever.py` | embedding 与 Chroma client 改为 `@lru_cache` 懒加载（`get_embedding` / `get_vectorstore`） | `import agent` 从 ~14s 降到 ~4s，BGE 权重只在真正检索时才加载 |
| 2 | `rag/router.py` | `_has_keyword` 对单词关键字统一用 `\bkw\w*` 单词边界匹配 | 修复 `coffee`/`model`/`accredited` 等被 `fee`/`mode`/`credit` 子串误判 |
| 3 | `agent/node/citation.py` | 复用 `Evidence.to_citation()`，删除冗余的 `"programme"` 键 | 消除与 `Evidence` 的重复实现，保持引用结构与证据对象一致 |
| 4 | `agent/node/answer_node.py` | `format_evidence` 删除 legacy dict 分支，只保留 `Evidence.render()` | 所有节点已统一产出 `Evidence`，旧分支为死代码 |

> 未改动部分（在 §3.5 弱点中记录，留待后续）：score 归一化、元数据过滤、`optional` 标签消费、摄入管道合并、`AgentState.programme_name` 消费、`build_field_document` 死代码清理等。
