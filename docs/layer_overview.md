# 系统分层与一次 Query 的完整旅程

> 定位：本文档以"一条用户 query 从入口到最终回答"的段落式叙述，逐阶段梳理当前代码（v2：`router → dispatcher`）的实现细节与数据结构流转。与 `architecture_overview.md`（静态架构）互为补充——那份讲"系统长什么样"，这份讲"一条请求具体怎么走"。
>
> 建议配合阅读：`agent/graph.py`（图与入口）、`agent/nodes/*.py`（各阶段节点）、`rag/retriever.py`（检索）、`data/sample4multiQuery.json`（真实请求的 checkpoint 样本）。

---

## 0. 总览：一次 query 的六个阶段

当前图是一条**固定线性流程**，共 6 个节点，无任何条件分支边（路由决策不在边上做，而是下沉到 dispatcher 节点内部）：

```
START
  │
  ▼
input_adapter  阶段 0：提取最后一条用户消息 → state["query"]
  │
  ▼
router         阶段 1：LLM 语义路由 → 每个子问题一条决策（RouterDecisionList）
  │
  ▼
dispatcher     阶段 2：按子决策分派到三个 retriever，合并证据（按 id 去重）
  │              ├─ metadata_retriever  精确事实：结构化解析直取 JSON 值
  │              ├─ section_retriever   详细内容：向量检索（可限定课程）
  │              └─ summary_retriever   课程推荐：向量检索整篇摘要
  │
  ▼
generator      阶段 3：证据渲染进 prompt → LLM 生成回答正文
  │
  ▼
citation       阶段 4：组装结构化 citations + confidence
  │
  ▼
output_adapter 阶段 5：包装为带 citations 的 AIMessage → END
```

State（`agent/state/schema.py`）按三层拆分：`InputState`（外部只传 `messages` 或 `query`）、`OutputState`（外部只收 `messages` / `final_response` / `citations`）、`AgentState`（内部全量，额外承载 `intent` / `decisions` / `programme_ref` / `resolved_programme_ref` / `evidence` / `answer` 等中间产物）。节点之间不传聊天消息列表，而是通过 LangGraph 的 state-reducer 语义互相读写这些**结构化字段**——每个节点收到完整 state，返回一个只含自己更新字段的 dict，运行时合并进共享 state。

下文按阶段展开，每阶段先讲"做什么"，再讲"代码里具体怎么做"。

---

## 1. 阶段 0：入口与 `input_adapter`

一条 query 有三种进入方式，最终都归一为 `state["query"]`：

- **CLI**（`agent/cli.py`）：直接 `app.invoke({"query": ...})` 或 `app.stream(..., stream_mode="values")`；
- **LangGraph server / chat UI**：以聊天消息形式传入，state 里是 `messages` 列表（含历史轮次）；
- **程序化调用**：`app.invoke({"messages": [...]})`。

`input_adapter` 负责从入口形态里**提取出"最新一轮的用户问题文本"**，只写 `query` 一个字段，逻辑分三层：

1. **形态兜底**：若 `messages` 为空但 `query` 已存在（CLI / 直接调用），原样透传；否则循环解包嵌套 list（`[[...]]`），取 `messages[-1]` 作为最后一条消息。
2. **消息类型兼容**：最后一条消息可能是 LangChain `BaseMessage`（读 `.content`），也可能是 dict（读 `["content"]`）。
3. **content 归一化**：`content` 可能是纯字符串，也可能是 content block 列表（chat 模型的多模态格式，形如 `[{"type": "text", "text": "..."}]`）——只拼接 `type == "text"` 的块，非 text 块（如图片）丢弃。

**关键细节**：该节点**只写 `query`**，不碰 router 上一轮留下的 `intent` / `retrieval_type` / `field` / `programme_ref` 等字段。因此多轮对话里，`input_adapter` 执行后存在一个"陈旧窗口"：`query` 已更新，但路由相关字段还是上一轮的旧值。当前图靠"router 每轮无条件全量覆写"兜住了这个问题（见 `multi_turn_conversation_analysis.md` §2.3/§4.5）；若未来在 `input_adapter` 与 `router` 之间插入新节点，需注意它会读到过期决策。

---

## 2. 阶段 1：`router` —— LLM 语义路由

router 的职责是：**把当前 query（连同完整对话历史）翻译成一份"检索计划"**——每个子问题一条决策，每条决策指明走哪条检索路径、查哪个字段、用什么问法。

### 2.1 输入与模型

输入是 `[SystemMessage(ROUTER_PROMPT), *messages]`——注意是**完整对话历史**，不是只有最新一轮；这也是多轮指代能工作的前提（模型从历史里推断省略的课程名）。模型是 `agent/llm.py` 里的 `ChatOpenRouter(model="deepseek/deepseek-v4-flash", temperature=0)`，通过 `with_structured_output(RouterDecisionList)` 强制输出 JSON（Pydantic schema 定义在 `agent/state/router_schema.py`）：

- 顶层：`intent`（qa / recommendation / comparison）+ `programme_ref`（整轮共享的课程引用）+ `decisions[]`；
- 每条 `RouterSubDecision`：`retrieval_type`（metadata / section / summary）、`field`（tuition_fee / deadline / duration / credit / study_mode / entrance_requirement / curriculum）、`sub_query`（该子问题独立问法）、`programme_ref`（仅跨课程复合问题才填，否则继承顶层）。

`ROUTER_PROMPT` 约束模型：只考虑最新一轮消息、不重发已答过的历史问题、最多 4 条决策、`sub_query` 不得夹带字段名/课程代号等注释。

### 2.2 三层可靠性防线

Phase 0 实测发现 deepseek-v4-flash 的嵌套 list 结构化输出不稳定：`field` 约 25–33% 概率解码为 None，且会把 JSON 泄漏进自由文本 `sub_query`（如 `"…? field: tuition_fee, programme: …"`）。因此 router 节点不信任 schema 约束，叠加三层防线（`router_node.py`）：

1. **确定性修复（`_prepare`，零 LLM 成本）**：`field` 缺失时用 `FIELD_KEYWORDS` 关键词表从 `sub_query` 文本反查（如 "english requirement" → `entrance_requirement`）；子决策 `programme_ref` 为空时用 `extract_programme_ref(sub_query)` 重新抽取；顶层 `programme_ref` 为空时借用第一条决策的引用。决策数超过 4 条截断。
2. **有效性校验 + 一次盲重试**：`_is_valid` 检查 `sub_query` 长度、是否含泄漏的 `"field:"` / `"programme:"`、字段与检索类型是否匹配（metadata 字段必须配 metadata 类型等）。校验不过就重调一次。
3. **规则路由兜底（`_fallback_decision_list`）**：两次都失败时，退回到 v1 的关键字路由器——`rag/router.py::classify_query` 判意图（metadata 关键字优先，其次 summary，其余 section），`rag/programme_resolver.py::extract_field` 抽字段（metadata 意图但抽不出字段时降级为 section），最终产出单决策计划。保证**任何输入下图都能拿到可用计划**。

### 2.3 输出

返回三个字段：`intent`、`programme_ref`（dict 化）、`decisions`（`model_dump()` 后的 dict 列表）。这就是"检索计划"，供 dispatcher 消费。

---

## 3. 阶段 2：`dispatcher` —— 按子决策分派检索

dispatcher 把 router 的计划**逐条执行**：对每条子决策，构造一个"子状态"（`{**state, query: sub_query, field, programme_ref}`），调用对应的 retriever 节点，最后把所有证据合并成一个列表。它是 v2 取代 v1 `add_conditional_edges` 的关键节点——路由分派从"图的边"下沉到"节点内的循环"。

**为什么每个子问题要带独立的 `sub_query`？** 因为 section 检索是对 `sub_query` 做向量搜索。若喂整条复合 query，"tuition fee" 的 token 会污染 "English requirements" 的检索结果。所以每条决策携带自己的问法文本，dispatcher 用 `dec.sub_query` 覆盖 `state["query"]` 后再调用 retriever。

**合并规则**：`evidence` 按 `e.id` 去重（两个子问题可能命中同一证据，如 "fee" 与 "cost" 都解析到 `P53-tuition_fee`）。`resolved_programme_ref` 采用"先到先得"——任何 retriever 成功解析出课程后写回，后续子决策与下一轮对话直接复用（多轮指代的关键，见 §7.3）。

下面分别看三条检索路径的实现细节。

### 3.1 metadata 路径：结构化精确查找（先查后搜）

`metadata_retriever_node.py`。精确事实（学费、截止日期、学分……）不是语义知识，所以这条路径**优先走结构化解析，不经过向量库**：

1. **字段确定**：`field = state.get("field") or extract_field(query)`——router 已给就用 router 的，否则从 query 文本用关键词表现抽（"tuition fee" → `tuition_fee`）。
2. **课程解析**：调用 `resolve_programme_ref(query, programme_ref, resolved_ref, messages)`，这是四层候选链（详见 §7.2）：本轮 router 引用 → 上轮已确认引用 → 当前 query 文本规则（`P\d{2,3}` 正则 / 课程名最长匹配）→ 最近若干条 human 消息文本。命中即返回完整课程对象。
3. **解析成功 → 直接取值**：`raw = programme["metadata"][field]`。`tuition_fee` 是 dict（local / non_local / source），用 `_render_fee` 渲染成 `Local Students:\n…\n\nNon-local Students:\n…`，`source` 拆出来放进 `evidence.metadata["url"]`（**URL 不进 LLM content**，只留给引用层）；其它字段用 `value_to_text`（deadline 优先取人类可读的 `raw` 而非 `iso`）。`field` 为 None 时退而渲染整篇 metadata 文档（`build_metadata_document`）。构造 `Evidence(id=f"{pid}-{field}", score=1.0, source_type="metadata")`——**score 固定 1.0，表示结构化精确命中**。同时写回 `resolved_programme_ref`（含真实 `programme_id`）。
4. **解析失败 → 向量回退**：query 里解析不出课程（如"有哪些项目收 30 学分？"）时，才在 `programme_metadata` collection 上做 top-5 语义检索，把每篇文档包成 `Evidence(id=f"{pid}-metadata-{i}", source_type="metadata")`。

### 3.2 section 路径：向量检索（可限定课程）

`section_retriever_node.py`。详细内容（入学要求、课程设置……）走向量库：

1. 与 metadata 路径共用同一个 `resolve_programme_ref` 四层链解析课程（保持两条路径对称，多轮省略指代时能互相抢救）。
2. 解析出课程 → `retrieve_section(sub_query, programme_id=pid, k=5)`，Chroma 加 `filter={"programme_id": pid}`，**把检索限定在该课程的 section 文档内**（这正是多轮场景证据全部是 `P53-*` 的原因）；解析不出 → 全库检索。
3. 每条结果包成 `Evidence(id=f"{pid}-{section}", score=<L2距离>, source_type="retrieval")`。解析成功同样写回 `resolved_programme_ref`。

**已知限制**（`multi_turn_conversation_analysis.md` §4.4）：`field`（如 `entrance_requirement`）在这条路径里**不参与过滤**——`retrieve_section` 只按课程过滤，不做 section 过滤，证据里会混入 Course Description / Useful Links 等无关段落，靠 generator 自己筛。这是检索精度上最直接的优化点。

### 3.3 summary 路径：课程推荐（整篇摘要检索）

`summary_retriever_node.py`。推荐类问题（"我感兴趣 AI，推荐项目"）对 `programme_summaries` collection 做 top-5 检索——每课程一篇规则拼装的摘要文档（目标 + 课程 + 亮点，`summary_builder.py`），按整篇课程排名而非片段。每条包成 `Evidence(id=f"{pid}-summary", section="Programme Summary")`，`source_type` 走默认值（`retrieval`）。这条路径不解析课程引用。

### 3.4 检索侧的共性问题：score 的语义

三个 collection 都用 Chroma 默认的 **L2 距离**，`similarity_search_with_score` 返回的是"**分数越低越相似**"的距离值。精确命中（metadata 结构化）是 `score=1.0`（高=好），向量检索却是 `0.45`（低=好）——两个量纲在 citation 层被同等对待，语义并不一致（详见 `architecture_overview.md` §3.5-1）。后续优化可考虑 `1/(1+dist)` 归一化或换 cosine。

---

## 4. 阶段 3：`generator` —— 证据驱动生成

`answer_node.py`。生成器**不自由发挥**，只做三件事：

1. **按 intent 选提示词**：`qa` 用 `QA_PROMPT`（逐子问题回答、事实直给、不臆造、证据不足明说）；`recommendation` / `comparison` 用 `SUMMARY_PROMPT`（说明匹配理由、多选项比较）。
2. **渲染证据**：`format_evidence` 把每条 `Evidence` 渲染成带标签的上下文块 `[P66 | Tuition Fee]\n<content>`，块间空行分隔；`content` 已经过 retriever 层"去杂质"（URL 等非 LLM 细节不在其中）。
3. **调用模型**：`model.invoke([SystemMessage(prompt), HumanMessage(f"User query:\n{query}\n\nRetrieved context:\n{context}")])`，temperature=0 保证确定性，返回 `{"answer": response.content}`。

复合问题的回答也是**一次生成**（dispatcher 已合并证据，generator 不按子问题循环），证据块自带 `[课程 | 章节]` 标签，统一 QA prompt 据此"对号入座"逐子问题作答。

---

## 5. 阶段 4：`citation` —— 结构化引用

`agent/nodes/citation.py`。生成完正文后，把 `evidence` 转成结构化的 `citations` 列表，供前端展示与溯源：

1. **课程名映射**：`_name_map()` 从 `data/programmes.json` 建 `programme_id → name` 映射（evidence 里只有 id，名字由这里补全，查不到就用 id 兜底）。
2. **置信度计算（`_confidence`）**：`source_type == "metadata"` 直接给 `"High"`（结构化精确命中）；向量检索按 score 阈值映射——`score < 0.3 → High`、`< 0.5 → Medium`、否则 `Low`。注意这里延续了 §3.4 的 L2 距离语义（低分=高相似），阈值方向没错但偏紧：样例里真正命中的 Entrance Requirements（0.647）也被标成 `Low`。
3. **组装**：每条 citation 含 `id / programme_id / programme_name / section / source_type / content / confidence / url`（url 取自 `evidence.metadata["url"]`，仅结构化路径带）。
4. **返回**：`final_response = state["answer"]`（正文原文），`citations` 单独成字段。

> ⚠️ **实现与文档的差异**：`citation_formatter` 的 docstring 描述"会在回答后追加 `Sources:` 块"，README 示例也展示了该输出——但当前代码**并没有拼接这个文本块**，`final_response` 就是 answer 原文。引用以结构化列表随 state 与 `AIMessage.additional_kwargs` 返回，由前端负责渲染（`sample4multiQuery.json` 里 ai 消息的 `additional_kwargs.citations` 即为该结构）。docstring 与 README 属早期版本残留。

---

## 6. 阶段 5：`output_adapter` —— 出参包装

`agent/graph.py::output_adapter`：把 `final_response` 与 `citations` 包装成一条 `AIMessage(content=final_response, additional_kwargs={"citations": citations})`，经 `add_messages` reducer 追加进 `state["messages"]`。这样：

- 走 `OutputState` 的调用方拿到 `messages`（含最终 AIMessage）、`final_response`、`citations` 三个公开字段；
- 内部中间态（`intent` / `decisions` / `evidence` / `answer` / `programme_ref`）**不进公开输出**，调试时经 `stream_mode="values"` 或 `get_state()` 观测。

---

## 7. 三个端到端案例（把上面串起来）

### 7.1 单事实查询："What is the tuition fee of MSc Computer Science?"

1. `input_adapter` → `query = "What is the tuition fee of MSc Computer Science?"`
2. `router` → `intent=qa`，单决策 `{retrieval_type: metadata, field: tuition_fee, sub_query: <原问法>}`，顶层 `programme_ref = {programme_id: None, programme_name: "MSc Computer Science"}`（**注意 id 是 None**，课程编号是下游解析出来的，不是 router 给的）。
3. `dispatcher` → 走 metadata 路径：`resolve_programme_ref` 命中顶层引用 → `find_programme("MSc Computer Science")` → P53 对象；`raw = P53.metadata["tuition_fee"]` → `_render_fee` 出 `Local Students: HK$7,600 per credit` / `Non-local Students: HK$9,100 per credit`，source 拆进 `url`；构造 `Evidence(P53-tuition_fee, score=1.0, source_type=metadata)`；写回 `resolved_programme_ref={programme_id: "P53", ...}`。
4. `generator` → QA prompt，答案形如 "Local Students: HK$7,600 per credit…"（样例见 `sample4multiQuery.json`）。
5. `citation` → `source_type=metadata` → `confidence: High`，url 一并进 citations；`final_response = answer`。
6. `output_adapter` → AIMessage + citations。

### 7.2 复合查询："What are the English requirements and tuition fee of MSc Computer Science?"

1. `router` → `intent=qa`，**两条**决策：`{section, entrance_requirement, "What are the English requirements of …?"}` 与 `{metadata, tuition_fee, "What is the tuition fee of …?"}`。
2. `dispatcher` → 循环两条：section 路径在 `programme_sections` 上做向量检索并加 `programme_id=P53` filter；metadata 路径结构化取值。证据合并，若两条命中同一 id 则去重。
3. `generator` → 证据块带 `[P53 | Entrance Requirements]`、`[P53 | Tuition Fee]` 标签，一次生成回答两个子问题。
4. `citation` / `output_adapter` → 两条 citation（entrance 那条是向量检索，按距离阈值多半是 `Low`；tuition 那条 `High`）。

### 7.3 多轮追问："what about English requirement?"（上一轮已问过学费）

这是 `data/sample4multiQuery.json` 记录的真实场景。核心机制有三层，缺一不可（详见 `multi_turn_conversation_analysis.md`）：

1. **checkpoint 续跑**：LangGraph server 按 `thread_id` 持久化 checkpoint，第二轮请求从第一轮终态恢复——`messages=[human1, ai1]`、`resolved_programme_ref`（第一轮已回填的 `P53`）全都在。
2. **router 全量重路由**：router 基于完整对话历史推断省略指代，重新输出 `programme_ref = {programme_name: "MSc Computer Science"}`——这轮的四字段整体覆写旧值，"清除旧状态"的本质就是覆写。
3. **检索限定课程**：section 路径命中 `resolved_programme_ref`（第一轮写回的 P53），`retrieve_section(Q2, programme_id="P53")` 过滤检索，证据全部是 `P53-*`。

`dispatcher` 在构造子状态时用 `{**state, ...}` 原样透传 `messages` 与 `resolved_programme_ref`，所以这三层机制在 v2 下依然成立。

---

## 8. 横切机制（贯穿多阶段）

### 8.1 `Evidence`：跨阶段统一数据单元（`rag/evidence.py`）

`@dataclass`：`id`（稳定锚点，如 `P53-tuition_fee`）、`programme_id`、`section`（展示名）、`content`（给 LLM 的纯文本，URL 等细节已剥离）、`score`（1.0=结构化命中，否则为 L2 距离）、`source_type`（metadata / retrieval）、`metadata`（携带 url 等仅供引用层的字段）。它是 retriever → generator → citation 之间唯一的流通对象，取代了早期直接用 `Document`/dict 传值的做法。

### 8.2 课程引用解析的四层 fallback 链（`rag/programme_resolver.py::resolve_programme_ref`）

metadata 与 section 两个 retriever 共用同一解析函数，候选按优先级：
**本轮 router 的 `programme_ref` → 上一轮已确认的 `resolved_programme_ref`（复用 id，免重复推断）→ 当前 query 文本规则（`P\d{2,3}` 正则 / 课程名最长匹配）→ 最近若干条 human 消息文本**。任一候选能被 `find_programme` 解析即返回。这条链是多轮指代、以及"P53 tuition fee"这种短 query 能命中的保障。

### 8.3 懒加载（`rag/retriever.py`）

embedding（`BAAI/bge-large-en-v1.5`，约 400MB 权重）与三个 Chroma client 都用 `@lru_cache` 包成工厂函数，首次真正检索时才加载。效果：`import agent` 从约 14s 降到约 4s，CLI / server 冷启动不背模型权重。

### 8.4 checkpoint 与多轮持久化

图本身（节点/边）不跨请求记忆任何东西，唯一的"记忆"是 LangGraph server 按 `thread_id` 持久化的 checkpoint 状态。分析样本 `sample4multiQuery.json` 含 10 个 checkpoint、2 次 run，第二次请求是在第一次的终态 checkpoint 之上续跑。`resolved_programme_ref` 是其中唯一被刻意设计为"对话级持久"的语义字段。

---

## 9. 相关文件索引

| 关注点 | 文件 |
|---|---|
| 图定义、input_adapter / output_adapter | `agent/graph.py` |
| 三层状态 schema、公开边界 | `agent/state/schema.py` |
| 路由 schema（RouterDecisionList / RouterSubDecision） | `agent/state/router_schema.py` |
| LLM 语义路由 + 三层防线 + 规则兜底 | `agent/nodes/router_node.py`、`rag/router.py` |
| 分派与证据合并 | `agent/nodes/dispatcher_node.py` |
| metadata 结构化检索 / section 向量检索 / summary 推荐 | `agent/nodes/metadata_retriever_node.py`、`section_retriever_node.py`、`summary_retriever_node.py` |
| 三个 Chroma 检索函数、懒加载 | `rag/retriever.py` |
| 课程引用四层解析链、字段抽取 | `rag/programme_resolver.py` |
| 证据对象 / 生成 / 引用 | `rag/evidence.py`、`agent/nodes/answer_node.py`、`agent/nodes/citation.py` |
| 数据管线（解析 → 建文档 → 入库） | `rag/parser.py`、`rag/document_builder.py`、`rag/summary_builder.py`、`rag/metadata_builder.py`、`rag/ingest.py`、`rag/metadata_ingest.py` |
| 多轮对话 checkpoint 分析 | `docs/multi_turn_conversation_analysis.md`、`data/sample4multiQuery.json` |
