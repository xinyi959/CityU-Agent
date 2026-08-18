# 多轮对话指代解析机制分析

> 样本：`data/sample4multiQuery.json`（agent-chat-ui 前端 → LangGraph 后端，同一 thread 两次请求）
> 场景：第一轮 "What is the tuition fee of MSc Computer Science?" → 第二轮 "what about English requirement?"
> 核心问题：系统**如何**让第二轮省略了指代（"English requirement" 暗指 "MSc Computer Science"）的 query 被正确清除旧状态、重新路由并落到 P53 的检索上。
> 结论先行：**系统没有专门的"指代消解/状态清除"节点**，它靠的是"router 每轮无条件基于完整对话历史全量重路由 + 下游 `find_programme` 名字匹配回填 programme_id + 向量库 programme_id 过滤"三层机制隐式完成的。

---

## 1. 样本数据概览

`sample4multiQuery.json` 是 LangGraph server 的 thread run 历史（checkpoint 流），共 **10 个 checkpoint**、分属 **2 个 run（2 次 HTTP 请求）**，thread_id 相同（`01a014ad-baf7-7d21-8298-9f6e4f52b31d`），因此第二次请求是在第一次请求的 checkpoint 之上**续跑**。

| 维度 | Run 1（学费） | Run 2（英语要求） |
|---|---|---|
| run_id | `01a014ad-baff-7d23-939b-5ed27b4a97c2` | `01a014ae-6ead-7492-9a29-efc0c2d8f7c2` |
| 可见 checkpoint step | 5, 6（样本只截取了末尾） | 7 – 14（完整） |
| 用户消息 | "What is the tuition fee of MSc Computer Science?" | "what about English requirement?" |
| 最终路由结果 | `qa / metadata / tuition_fee` | `qa / section / entrance_requirement` |
| `programme_ref` | `{programme_id: None, programme_name: "MSc Computer Science"}` | `{programme_id: None, programme_name: "MSc Computer Science"}` |

注意：**两个 run 的 `programme_ref.programme_id` 都是 `None`**，router 两次都只返回了课程名字。P53 这个 id 是下游 `find_programme()` 靠名字匹配出来的，不是 router 直接给的。

### 1.1 逐 checkpoint 时间线

| step | checkpoint_id（截断） | 执行节点 | 状态变化 | run |
|---|---|---|---|---|
| 5 | `1f19af9d-947e…` | `output_adapter` | 生成 ai1（学费回答 + citations） | 1 |
| 6 | `1f19af9d-9487…` | （终态） | `query=Q1, retrieval_type=metadata, field=tuition_fee, programme_ref={name:MSc CS}`，messages=[human1, ai1] | 1 |
| 7 | `1f19af9f-1bd3…` | `__start__` | **追加** human2；`query` 与 router 四字段仍是 Q1 的旧值 | 2 |
| 8 | `1f19af9f-1bd6…` | `input_adapter` | 仅覆写 `query:=Q2`；**intent/retrieval_type/field/programme_ref 保持 Q1 旧值（陈旧窗口）** | 2 |
| 9 | `1f19af9f-1bdc…` | `router` | **全量重路由**：`qa / section / entrance_requirement / programme_ref={name:MSc CS}` —— 四字段被整体覆写 | 2 |
| 10 | `1f19af9f-3bf5…` | `section_retriever` | `find_programme(name)→P53`，`retrieve_section(Q2, programme_id=P53)` 过滤检索，产出 5 条 P53 证据 | 2 |
| 11 | `1f19af9f-3c2b…` | `generator` | 生成 answer（英语要求） | 2 |
| 12 | `1f19af9f-7d6f…` | `citation` | 组装 citations + final_response | 2 |
| 13 | `1f19af9f-7d73…` | `output_adapter` | 生成 ai2（含 citations） | 2 |
| 14 | `1f19af9f-7d78…` | （终态） | messages=[human1, ai1, human2, ai2] | 2 |

---

## 2. 核心机制：第二轮 query 是如何被"清除"并正确重路由的

### 2.1 跨请求状态持久化（续跑的前提）

LangGraph server 按 `thread_id` 持久化 checkpoint。第二次请求进来时，图从 Run 1 的终态 checkpoint（step 6）恢复：`messages=[human1, ai1]`、`query=Q1`、以及 router 上一轮产出的四个字段全部还在。**图结构本身（节点/边）不跨请求记忆任何东西，唯一的"记忆"就是这条 checkpoint 状态**。

### 2.2 `__start__`：只追加消息，不重算任何决策（step 7）

`__start__` 的 task result 只是 `{'messages': [human2]}`，由 `AgentState.messages` 上的 `add_messages` reducer 追加到历史。此时 state 里 `query` 仍是 Q1，router 四字段也仍是 Q1 的 —— 一切"语义决策"都还没动。

### 2.3 `input_adapter`：只提取 query，**不**清除 router 旧字段（step 8）⚠️

`agent/graph.py` 的 `input_adapter`：

```python
def input_adapter(state: AgentState):
    messages = state.get("messages", [])
    ...
    last_message = messages[-1]        # 取最后一条 human 消息
    ...  # 解析 content block -> query 文本
    return {"query": query}            # 只写 query
```

它的返回**只有 `query`**。因此 step 8 之后存在一个状态窗口：`query=Q2`（新），但 `intent=qa, retrieval_type=metadata, field=tuition_fee, programme_ref={name:MSc CS}`（旧，来自 Q1）。这个"陈旧窗口"是设计使然——因为后续 `router` 会无条件覆写这四个字段；但如果任何新插入的节点在这之前读这些字段，就会读到 Q1 的过期决策（详见 §5.4）。

### 2.4 `router`：全量重路由 = 真正的"清除"机制（step 9）

`agent/nodes/router_node.py`：

```python
def router_node(state):
    messages = state["messages"]                 # 完整对话历史
    decision = router_llm.invoke(
        [SystemMessage(content=ROUTER_PROMPT), *messages]
    )
    ...
    return {
        "intent": decision.intent,
        "retrieval_type": decision.retrieval_type,
        "field": decision.field,
        "programme_ref": programme_ref,          # 四字段整体覆写
    }
```

关键点：

1. **Router 每次请求都会无条件重新执行**（`START → input_adapter → router` 是固定边），并且把 `intent / retrieval_type / field / programme_ref` 四个字段**整体覆写**。所谓"清除第二次 query 的旧状态"，本质就是：上一轮的 router 输出在这一轮被新一轮 LLM 决策整个替换掉。系统里不存在一个显式的 reset 节点，清除是"覆写式"的。
2. **Router 只看 messages，看不到上一轮的 router 决策、evidence 或 citations**。它拿到的上下文是 `[system prompt, human1, ai1, human2]`。其中 ai1 的 `additional_kwargs.citations`（含 `programme_id: P53`）**不会进入 LLM 调用**——`langchain_openai` 的 `_convert_message_to_dict` 只透传 `name/tool_calls/function_call/audio` 等白名单键，自定义的 `citations` 键被丢弃（见 `agent/llm.py` 使用的 `ChatOpenRouter` 底层转换逻辑）。而 ai1 的 content 文本（"Local Students: HK$7,600 per credit…"）也**没有提及课程名**。所以 router 推断"指代对象是 MSc Computer Science"的唯一信息源，是 **human1 的消息文本**。
3. `RouterDecision` 的 `programme_ref` 字段在 prompt 里**没有任何说明**（见 §5.1），router LLM 输出 `{programme_id: None, programme_name: "MSc Computer Science"}` 属于 schema 驱动下的涌现行为：它正确地把省略指代关联到了上一轮的课程名，但**没有回填 P53 编号**。

### 2.5 引用解析链路：LLM 只给名字，代码负责补全 id（step 10）

`agent/nodes/section_retriever_node.py`：

```python
programme_ref = state.get("programme_ref")
if programme_ref:
    programme = find_programme(programme_ref)   # 名字 -> 全量课程对象
    if programme:
        programme_id = programme["programme_id"] # 回填 P53

docs = retrieve_section(query, programme_id=programme_id, k=5)
```

`rag/programme_resolver.py::find_programme` 支持 `programme_id` 或 `programme_name` 任一命中即返回课程对象；`rag/retriever.py::retrieve_section` 在拿到 `programme_id` 后对 Chroma 加 `filter={"programme_id": programme_id}`，**把向量检索限定在 P53 的 section 文档内**。样本中 5 条证据全部是 `P53-*`，正是这层过滤的产物。

### 2.6 检索 → 生成 → 引用（steps 11–13）

- `generator` 按 `retrieval_type=section` 选择 `SECTION_PROMPT`，把 5 条 P53 证据渲染进 prompt，生成英语要求回答。
- `citation` 把 evidence 转成带 confidence 的 citations；`output_adapter` 包装成 ai2 消息，完成第二轮。

---

## 3. 多轮指代为什么能工作的三层保障

| 层 | 机制 | 代码位置 | 样本证据 |
|---|---|---|---|
| LLM 层 | router 基于完整对话历史推断省略指代，输出 `programme_name` | `agent/nodes/router_node.py` | step 9 输出 `programme_ref.name = "MSc Computer Science"` |
| 解析层 | `find_programme` 把"只有名字的引用"解析为课程对象并回填 id | `rag/programme_resolver.py` | step 10 得到 `P53` |
| 检索层 | `retrieve_section` 的 `programme_id` filter 保证证据只来自 P53 | `rag/retriever.py` | step 10 证据全部为 `P53-*` |

三层中任何一层失效（LLM 没推断出课程名 / 名字匹配失败 / filter 未生效），第二轮就会退化为全库检索，可能给出跨课程的混合答案。

---

## 4. 暴露的问题与脆弱点（多轮对话优化的抓手）

### 4.1 Router prompt 没有显式的"指代消解/引用提取"指令

`ROUTER_PROMPT` 只讲了 intent 和 retrieval_type 的分类标准，**完全没有提 programme_ref**。`programme_ref` 的提取完全是 `RouterDecision` schema 里挂了个字段、模型自发完成的。这带来两个后果：

- 模型行为不稳定：换模型/换温度/换措辞，`programme_ref` 可能返回 `None`，且没有任何兜底（见 4.3）。
- 模型**从不回填 `programme_id`**（两次 run 都是 `None`），课程编号的确定始终依赖名字匹配这条脆弱的链。

### 4.2 只信名字匹配，不信任上一轮的解析结果

Run 1 的 step 10 其实已经通过 `find_programme` 把名字解析成了 P53（evidence id 为 `P53-tuition_fee`），但这个解析结果**没有写回 state**（`metadata_retriever_node` 只返回了 `programme_id/pid` 作为普通 state 字段，且 `AgentState` 里没有对应的持久化设计；`programme_ref` 字段始终是 router 的原始输出）。于是 Run 2 的 router 无法复用"上一轮已确认 P53"这个事实，必须重新靠 LLM 从对话文本里再推断一遍名字。样本中成功了，但这是可避免的风险。

### 4.3 `section_retriever` 没有 fallback，与 `metadata_retriever` 不对称

`metadata_retriever_node` 有兜底：`ref = state.get("programme_ref") or extract_programme_ref(query)`——即使 router 没给引用，还能从 query 文本里用正则/名字匹配抢救（"P53 tuition fee" 这种 query 也能命中）。而 `section_retriever_node` **没有这一行**：`programme_ref` 为 `None` 就直接全库检索。对多轮场景这是最危险的分支——省略指代越严重（"what about English?"、"那雅思呢？"），router 越可能漏掉引用，section 路径就完全裸奔。

### 4.4 `field` 在 section 路径中被忽略

`RouterDecision.field` 把两类词汇合并成一个 Literal：metadata 字段（`tuition_fee/deadline/duration/credit/study_mode`）和 section 字段（`entrance_requirement/curriculum`）。但 `section_retriever_node` **根本不读 `field`**，`retrieve_section` 只是对 P53 全部 section 做 top-k 相似检索。样本里 router 明明输出 `field=entrance_requirement`，最终 evidence 却包含 Course Description / Useful Links / Did You Know? / Scholarship 等无关 section（score 0.93–1.09 的都比真正命中的 Entrance Requirements 0.647 高，靠的是 Chroma 距离语义，见 4.6），全靠 generator 自己筛。也就是说 **"entrance_requirement" 这个 router 输出目前是装饰性的**，没有参与检索过滤——这是检索精度上最直接的优化点。

### 4.5 陈旧状态窗口（`input_adapter` 与 `router` 之间）

step 8 的 checkpoint 里 `query=Q2` 但 `retrieval_type/field/programme_ref` 还是 Q1 的（`metadata/tuition_fee`）。当前图里 router 紧跟其后、必然覆写，所以无害；但后续如果有人在 input_adapter 之后、router 之前加节点，或改成条件边跳过 router，就会读到过期决策。建议要么让 `input_adapter` 顺带把 router 四字段清空为 `None`，要么在 `AgentState` 上明确区分"router 临时输出"与"对话级持久状态"两类字段。

### 4.6 顺带：confidence 的语义（与 Chroma 距离有关）

`citation_formatter._confidence` 的阈值是"分数越低越可信"：`score < 0.3 → High, < 0.5 → Medium, else → Low`。这与 `langchain_chroma` 的 `similarity_search_with_score` 返回 **L2 距离**（"Lower score represents more similarity"）一致，所以方向没错；但阈值很紧，样本中真正命中的 Entrance Requirements（0.647）也被标成 `confidence: Low`，5 条检索 citation 全部 Low。这不是 bug，但会影响前端展示的可信度观感，值得在优化时校准。

---

## 5. 多轮对话优化建议（按性价比排序）

1. **给 router 显式的引用提取指令 + 上轮已解析引用作为输入**
   在 `ROUTER_PROMPT` 中增加一段："若 query 未提及课程，必须依据对话历史推断其指代对象；优先沿用上一轮已确认的 programme"。同时在 `AgentState` 增加 `resolved_programme_ref`（带 `programme_id`），由各 retriever 在解析成功后写回，router 将其作为候选传入 prompt。这能把"指代正确性"从模型涌现行为变成受控流程。

2. **`section_retriever` 补齐与 `metadata_retriever` 对称的 fallback** ✅ 已实施
   `rag/programme_resolver.py::resolve_programme_ref` 提供四层候选链（router 引用 → 上轮已确认引用 → query 文本规则 → 最近若干条 human 消息文本），两个 retriever 共用；`section_retriever_node` 已接入（省略指代失败时至少能按文本规则抢救，"MSc Computer Science" 出现在第一轮 human 消息里也能命中）。

3. **让 `field` 真正参与 section 检索**
   在 `retrieve_section` 增加可选 `section` 过滤（`filter={"programme_id": P53, "section": "Entrance Requirements"}`），或对 evidence 做 field 相关的后置重排，去掉 Course Description / Useful Links 这类无关段落，减少 generator 的噪声与 Low-confidence 引用。

4. **消除陈旧窗口**：`input_adapter` 返回时把 `intent/retrieval_type/field/programme_ref` 置空，或在图上明确"router 之前无任何读取这些字段的节点"的约束并加注释/断言。

5. **把上一轮的 P53 解析结果写回 state** ✅ 已实施
   `AgentState` 新增 `resolved_programme_ref`，`metadata/section` 两个 retriever 在 `find_programme` 解析成功后写回（带 `programme_id`）；第二轮可直接复用 id，无需重复"名字 → id"推断。

6. **校准 confidence 阈值或改用 cosine**：若改用 `collection_metadata={"hnsw:space": "cosine"}`（分数越高越相似），同步修正 `_confidence` 方向，并调整阈值使真正命中的检索证据能显示 Medium/High。

---

## 6. 相关代码位置索引

| 关注点 | 文件 |
|---|---|
| 图结构、input_adapter（只写 query）、output_adapter | `agent/graph.py` |
| router 全量重路由、prompt 与覆写逻辑 | `agent/nodes/router_node.py` |
| `RouterDecision` schema（`field` 双词汇合并、`programme_ref`） | `agent/state/router_schema.py` |
| 名字→id 解析（`find_programme` / `extract_programme_ref`） | `rag/programme_resolver.py` |
| section 检索与 programme_id filter | `agent/nodes/section_retriever_node.py`、`rag/retriever.py` |
| metadata 检索及 fallback（对比参考） | `agent/nodes/metadata_retriever_node.py` |
| confidence 阈值与 citations 组装 | `agent/nodes/citation.py` |
| LLM 消息序列化（additional_kwargs 白名单，citations 不进入模型上下文） | `.venv/.../langchain_openai/chat_models/base.py::_convert_message_to_dict` |
| 样本数据 | `data/sample4multiQuery.json` |

---

## 7. v2 变更附注（Phase 1–3 之后）

本文档分析的是 v1 时期捕获的样本（router 输出单决策）。v2 起：

- router 输出 `RouterDecisionList`（每个子问题一条 decision），图改为 `router → dispatcher → generator`（见 `architecture_overview.md` §2.8）。
- 多轮指代解析的三层机制（全量重路由 + 名字匹配回填 id + programme_id 过滤）**不变**：dispatcher 仍把 `messages` / `resolved_programme_ref` 传入每个子决策的检索子状态，省略指代的第二轮照常落到已确认课程的检索上。
- 新增约束：router prompt 只切分**最新一轮**的子问题，禁止把历史已答问题重新吐成 decision（Phase 0 探针 D2 用例暴露的过度切分）。
- `RouterDecision` schema 相关描述已过时：字段定义见 `agent/state/router_schema.py`（`RouterDecisionList` / `RouterSubDecision`）。
