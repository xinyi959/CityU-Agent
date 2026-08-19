# 复合问题（Compound Query）实现工作记录

> 对应 commit（本分支 `sub-query4multi-question`）：
> `bc97eba`（Phase 1）→ `ddc9e65`（Phase 2）→ `7010a7b`（Phase 3）→ `596f534`（Phase 4）
>
> 解决的问题：一条复合 query（如 "What are the tuition fee and English requirement of
> MSc Computer Science?"）如何被**拆成多个子决策 → 按序触发不同 retriever → 合并证据 →
> 交给 generator 一次生成**。
>
> 本文是工作记录：包含真实运行 trace、代码位置与设计取舍理由，方便回看。

---

## 1. 背景与目标

改造前：router 输出单决策，复合问题只取**一个**字段——"英语要求 + 学费"只会答其中
一个（Phase 0 前系统实测行为）。目标：复合问题的**每个子问题都被回答**。

方案选型（最初的权衡）：

| 方案 | 合并位置 | LLM 调用 | 结论 |
|---|---|---|---|
| A. router 输出 list + dispatcher 扇出 | 证据层（evidence 合并后一次生成） | router + 1 次生成 = 2 次 | ✅ 采用 |
| B. 引入 sub-agent | 答案层（每子问题独立生成 + 聚合） | N 次生成 + 1 次聚合 = N+2 次 | 边界场景再上 |

选 A 的理由（依据领域形状）：复合问题几乎都是"**同一门课、多个独立字段、单跳查找**"
（fee + requirement + deadline…），没有子问题间依赖，不需要各自独立推理上下文；且下游
（citation / output_adapter）只消费 `evidence`，证据层合并后**整条下游零改动**。

---

## 2. 四个 commit 的分工

| Commit | 阶段 | 改动 | 交付物 |
|---|---|---|---|
| `bc97eba` | P1 schema + router | `RouterDecisionList` 取代单决策；新 prompt；field repair / 重试 / 规则兜底 | `agent/state/router_schema.py`、`agent/nodes/router_node.py`、`agent/state/schema.py`（+`decisions`） |
| `ddc9e65` | P2 图改线 | conditional edges 删除；新增 `dispatcher` 节点按子决策扇出 | `agent/nodes/dispatcher_node.py`、`agent/graph.py` |
| `7010a7b` | P3 生成器 | metadata/section 提示词合并为统一 QA_PROMPT，按子问题对号入座 | `agent/nodes/answer_node.py` |
| `596f534` | P4 文档+修复 | 文档 v2 化；`FIELD_LABELS` 补 router Literal 别名 | `docs/*.md`、`rag/metadata_builder.py` |

每个 commit 单独可跑、不破坏单问题路径（P1 期间 graph 仍走 `decisions[0]`，行为与改造前
逐字节一致）。

---

## 3. 端到端流水线：一条复合 query 的旅程

以真实 query **"What are the tuition fee and English requirement of MSc Computer Science?"**
为例（后续所有输出均为真实运行 trace）。

### 3.1 input_adapter

从 `messages` 提取最后一条用户文本写入 `state["query"]`。此后 `query` 全图只读。

### 3.2 router —— 拆分（Phase 1）

调用 `model.with_structured_output(RouterDecisionList)`，prompt 关键规则：

- 只切分**最新一轮**消息（禁止把历史已答问题重新吐成 decision）；
- 每个子问题一条 decision，保持顺序；
- `sub_query` 只写问题本身（禁止夹带 `field:`/`programme:` 注解）；
- per-decision `programme_ref` 仅在跨课程时填写，否则继承顶层。

真实输出：

```
ROUTER PLAN: qa [('metadata', 'tuition_fee'), ('section', 'entrance_requirement')]
```

```json
{
  "intent": "qa",
  "programme_ref": { "programme_name": "MSc Computer Science" },
  "decisions": [
    { "retrieval_type": "metadata", "field": "tuition_fee",
      "sub_query": "What is the tuition fee of MSc Computer Science?", "programme_ref": null },
    { "retrieval_type": "section", "field": "entrance_requirement",
      "sub_query": "What is the English requirement of MSc Computer Science?", "programme_ref": null }
  ]
}
```

**可靠性三层（不信任 schema 约束）**——Phase 0 探针发现 deepseek 的嵌套 list 结构化输出
不稳定，`field` 约 25–33% 概率解码为 None 且把 JSON 泄漏进 `sub_query`。因此：

1. **确定性修复**（零 LLM）：`field=None` 时用关键词表从 `sub_query` 反查（
   `_repair_field`）；`programme_ref` 为空时用 `extract_programme_ref(sub_query)` 反查
   （`_repair_programme_ref` / `_repair_top_programme_ref`）。
2. **一次盲重试**：修复后仍有非法 decision（含泄漏 JSON）→ 重调一次。
3. **规则兜底**：仍失败 → 退回 v1 关键字路由（`rag/router.py::classify_query` +
   `extract_field`），保证图总能拿到可用 plan。

router 写 state：`intent` / `retrieval_type`+`field`（= decisions[0] 的值，向后兼容）/
`programme_ref` / `decisions`。

### 3.3 dispatcher —— 按序执行（Phase 2）

```python
# agent/nodes/dispatcher_node.py —— 单个图节点，内部 for 循环
for dec in decisions:
    sub_state = {**state,
                 "query": dec["sub_query"],          # 子问题单独传给检索器
                 "retrieval_type": dec["retrieval_type"],
                 "field": dec["field"],
                 "programme_ref": dec["programme_ref"] or state["programme_ref"]}
    out = RETRIEVER_NODES[dec["retrieval_type"]](sub_state)   # 顺序调用
    # 合并 evidence，按 id 去重
```

**关键点：**

- **顺序执行，不是并行**。两个 retriever 在同一个节点函数里先后跑完（真实耗时 5.50s，
  无重叠）；证据顺序 = decisions 顺序，确定性。
- **每个 decision 带独立 `sub_query`**：section 检索是对 `sub_query` 做向量搜索，若喂整条
  复合 query，"tuition fee" 的 token 会污染 "English requirement" 的检索。
- **合并发生在节点内部**（本地 list + 按 id 去重，如 "fee" 与 "cost" 都命中
  `P53-tuition_fee`），dispatcher **只返回一次** partial state：
  `{"evidence": [...], "resolved_programme_ref": {...}}`。

真实证据顺序（metadata 在前，因 decisions[0] 是 metadata）：

```
['Tuition Fee', 'Entrance Requirements', 'Course Description', 'Did You Know?',
 'Useful Links', 'Programme Aims and Objectives']
```

> **关于 reducer（常被问到）**：`AgentState.evidence` 是裸 `list`，**没有** `Annotated`
> reducer（全 state 只有 `messages` 挂了 `add_messages`）。因为合并发生在 dispatcher 单节点
> 内部，`evidence` 只被写**一次**，LangGraph 默认 last-write-wins 就是简单赋值——不存在
> 并行写入竞争。若改成 Send API 并行分支（两个分支各写 `evidence`），默认 reducer 下后写
> 覆盖先写、丢一路证据，才必须上 `Annotated[list, operator.add]`。当前设计用"单节点单写入"
> 绕开了 reducer 复杂度。

### 3.4 generator —— 一次生成（Phase 3）

generator 收到的 state（真实值）：

| 字段 | 值 | 说明 |
|---|---|---|
| `query` | 原始完整复合问句 | dispatcher 不改写它；`sub_query` 只活在传给 retriever 的局部子状态里 |
| `evidence` | 6 条（1 精确 metadata + 5 向量 section） | 合并后一次性写入 |
| `retrieval_type` / `field` | `metadata` / `tuition_fee` | decisions[0] 的值；dispatcher 不返回它们，各 retriever 自己返回的 `retrieval_type` 被 dispatcher 吞掉（只透传 evidence / resolved_programme_ref） |
| `resolved_programme_ref` | `{programme_id: 'P53', ...}` | 第一个解析成功的 retriever 回填 |
| `intent` / `decisions` / `programme_ref` | router 原值 | 原样保留 |

提示词从"按 retrieval_type 选提示词"改为**按 intent 选**：`qa` 用统一 QA_PROMPT
（逐子问题对号入座证据块 `[P53 | Entrance Requirements]`），recommendation/comparison 仍用
SUMMARY_PROMPT。原因：混合证据下按 decisions[0] 选提示词是错的（此前实跑是
"section 提示词答全部"，碰巧能用但语义不严谨）。

真实答案（覆盖两问，事实准确）：

```
**English requirements**:
- TOEFL (Internet-based): 79
- IELTS (Academic): overall band score of 6.5
- CET-6: 450

**Tuition fee:**
- Local students: HK$7,600 per credit
- Non-local students: HK$9,100 per credit
```

### 3.5 citation —— 来源（零改动）

遍历合并后的 6 条 evidence 生成 citations：`[P53-Entrance Requirements]`（section 向量，
confidence Low）与 `[P53-tuition_fee]`（metadata 精确，confidence High）各带 URL/来源。
`FIELD_LABELS` 补了 router Literal 别名（`study_mode→Mode of Study` 等），保证展示名正确。

---

## 4. 设计决策记录

| # | 决策 | 理由 |
|---|---|---|
| 1 | 证据层合并（list + dispatcher）而非答案层（sub-agent） | 领域形状：同课程独立字段、单跳；零额外 LLM 调用；下游零改动。sub-agent 留作跨 intent 复合的触发条件 |
| 2 | 每个 decision 带独立 `sub_query` | section 向量检索不能被其它子问题的 token 污染（实测必要性） |
| 3 | per-decision `intent` 从 schema 去掉 | Phase 0 探针显示是结构化输出解码不稳定的主因；turn 级意图本来就一个 |
| 4 | 合并放 dispatcher 单节点内（非并行分支） | 避免 reducer 竞争（§3.3 说明）；证据顺序确定性；代价是两路检索不重叠，量级可忽略 |
| 5 | `programme_ref` 有确定性修复层 | 模型偶尔整体丢弃 ref（A3/E1 用例暴露），从 sub_query 文本反查零成本修复 |
| 6 | 可靠层 = 修复 + 一次重试 + 规则兜底 | 不信任 temperature=0 下的结构化输出稳定性（实测抖动）；兜底保证图永不崩溃 |
| 7 | 顶层 `retrieval_type`/`field` 保留为 decisions[0] 的值 | 向后兼容旧单决策路径 / 调试观察 |

---

## 5. Phase 0 探针结论（设计依据）

12 条复合用例 × 3 轮 live 探针（`test/test_router_compound.py`，未入库）：

| 失败模式 | 表现 | 对策 |
|---|---|---|
| 模式 1：结构化输出解码不稳定 | `field` 解码为 None（25–33%），JSON 泄漏进 sub_query（"…? field: tuition_fee, programme: …"） | 修复层 + 重试 + 泄漏检测（`_is_valid`） |
| 模式 2：多轮过度切分 | 把历史已答问题重新吐成 decision（D2 用例三轮全败） | prompt 规则 1：只切分最新一轮 |
| 模式 3：P-code 不映射课程名 | `P53` → programme_ref=None | 无害：下游 `resolve_programme_ref` 用 `_ID_RE` 按 code 解析 |

修复后回归：**12/12 通过**（含跨课程 E1、跨 intent E2、四元素上限 E3、多轮指代 D2）。

---

## 6. 已知边界与后续

- **跨课程复合**（"X 的学费 + Y 的英语要求"）：schema（per-decision programme_ref）与
  dispatcher（per-decision ref 优先）已支持，但**未做端到端验证**（回归只验到 router 输出）。
- **跨 intent 复合**（"推荐一个项目 + 告诉我它的学费"）：router 能识别 summary + metadata
  两条 decision，但合并后的证据交给单一生成器，提示词人格不统一——这是将来上 **sub-agent**
  （每子问题独立走 路由→检索→生成，再聚合）的触发条件。
- **延迟**：router 触发重试时单次可达 30–60s（deepseek 抖动，B1 用例最坏 65s）；可优化为
  带失败反馈的重试或下调 `MAX_DECISIONS`。
- **并行化**：若需真正并行，两条路——dispatcher 内 `ThreadPoolExecutor`（需评估
  Chroma/embedding 线程安全 + 按 decision 下标重排证据）或 Send API + `evidence` 挂 reducer
  （需处理去重/顺序）。

---

## 7. 文件索引

| 文件 | 职责 |
|---|---|
| `agent/state/router_schema.py` | `RouterDecisionList` / `RouterSubDecision` / `ProgrammeRefModel` |
| `agent/nodes/router_node.py` | LLM 路由 + `_repair_field` / `_repair_programme_ref` / 重试 / `_fallback_decision_list` |
| `agent/nodes/dispatcher_node.py` | 按 decisions 顺序调 retriever、合并去重、resolved_ref 回填 |
| `agent/nodes/answer_node.py` | 统一 QA_PROMPT（按 intent 选提示词） |
| `agent/graph.py` | 线性图：input_adapter → router → dispatcher → generator → citation → output_adapter |
| `agent/state/schema.py` | `decisions` 字段（P1 新增） |
| `rag/metadata_builder.py` | `FIELD_LABELS` router Literal 别名 |
| `docs/architecture_overview.md` | 架构 v2 化（§1.2 / §2 / §2.8 复合问题 / §5.1 复合案例） |
| `test/test_router_compound.py`（未入库） | 12 用例 router 回归（真实 router_node） |
| `test/test_agent_live.py`（未入库） | 5 条端到端（含 2 条复合） |
