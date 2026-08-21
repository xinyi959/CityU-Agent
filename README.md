# CityU-Agent

> 香港城市大学（CityUHK）授课型硕士课程问答与推荐助手 —— 基于 LangGraph 的规则路由 + RAG 确定性图

CityU-Agent 是一个面向 CityUHK 授课型硕士（Taught Postgraduate）课程信息的问答/推荐系统。它不是让 LLM 自由调用工具的 ReAct agent，而是一条 **规则路由 + 检索增强生成（RAG）** 的确定性 LangGraph 流水线：由 LLM 语义路由器把用户问题拆分为子决策，调度器按子决策扇出到对应的向量检索路径，最后由生成器基于检索到的证据组织回答，并附带结构化引用来源。

支持**复合问题**（一条查询包含多个子问题，如 "英语要求和学费"）与**多轮对话**（省略指代时自动继承上文课程引用）。

## ✨ 核心特性

- 🧠 **LLM 语义路由**：`deepseek/deepseek-v4-flash`（OpenRouter，temperature=0）结构化输出路由计划，每个子问题一条决策
- 🔀 **复合问题支持**：router 输出决策列表（最多 4 条），dispatcher 扇出多条检索路径并合并证据（按 id 去重）
- 🗂️ **三类检索路径**：结构化精确查找（metadata）/ 详细内容向量检索（section）/ 课程推荐（summary）
- 📚 **证据驱动回答**：所有回答基于检索到的 `Evidence` 生成，不臆造事实，回答附带结构化引用来源（citations：id / 课程 / 章节 / 置信度 / url，随 API 返回）
- 💬 **多轮对话**：`resolved_programme_ref` 回填课程 id，后续轮次省略指代时自动复用
- 🛡️ **多层可靠性防线**：字段确定性修复 → 盲重试 → 规则路由兜底，保证图总能拿到可用计划
- ⚡ **懒加载优化**：BGE 嵌入权重与 Chroma 客户端缓存到首次检索时才加载，`import agent` 保持轻量

## 🏗️ 系统架构

```
 User Query
     │
     ▼
input_adapter     提取最后一条用户消息 → state["query"]
     │
     ▼
router            LLM 语义路由 → RouterDecisionList
                  (intent + programme_ref + decisions[]，每个子问题一条)
     │
     ▼
dispatcher        按 retrieval_type 扇出到对应 retriever，合并 evidence 并去重
     │  ├─ metadata → 结构化精确查找 (data/programmes.json) / programme_metadata 索引
     │  ├─ section  → programme_sections 索引（按 programme_id 过滤）
     │  └─ summary  → programme_summaries 索引（课程推荐）
     ▼
generator         format_evidence → LLM 生成回答（qa / summary 两套提示词）
     │
     ▼
citation          组装结构化 citations（id / 课程 / 章节 / 置信度 / url）
     │
     ▼
output_adapter    包装为带 citations 的 AIMessage → Final Response
```

图是**无循环 DAG**（`input_adapter → router → dispatcher → generator → citation → output_adapter`），
LLM 只出现在 router 与 generator 两个节点；三个 retriever 由 dispatcher 以普通函数形式调用。

## 🛠️ 技术栈

| 类别 | 技术 |
|---|---|
| 工作流 | [LangGraph](https://langchain-ai.github.io/langgraph/)（`langgraph-cli[inmem]` 提供本地 server） |
| LLM | OpenRouter `deepseek/deepseek-v4-flash`（temperature=0） |
| 向量库 | [Chroma](https://www.trychroma.com/)（持久化于 `rag/vectorstore/`，3 个 collection） |
| 嵌入模型 | `BAAI/bge-large-en-v1.5`（[sentence-transformers](https://www.sbert.net/)） |
| 数据源 | 64 份 CityUHK 课程 Markdown（`data/markdown/`） |

## 📁 项目结构

```
CityU-Agent/
├── agent/                    # LangGraph 图与节点
│   ├── graph.py              # 图定义（6 个节点、state schema、编译为 app）
│   ├── cli.py                # 命令行入口（one-shot / REPL）
│   ├── llm.py                # OpenRouter 模型初始化
│   ├── state/                # InputState / OutputState / AgentState / RouterDecisionList / Citation
│   └── nodes/                # router / dispatcher / 三个 retriever / answer / citation
├── rag/                      # RAG 数据管线
│   ├── parser.py             # Markdown → data/programmes.json（规则解析器）
│   ├── document_builder.py   # section 子文档构建
│   ├── summary_builder.py    # 每课程一篇 summary
│   ├── metadata_builder.py   # 每课程一篇 metadata 文档
│   ├── ingest.py             # 写入 programme_sections + programme_summaries
│   ├── metadata_ingest.py    # 写入 programme_metadata
│   ├── retriever.py          # 三个 Chroma 检索函数（懒加载）
│   ├── programme_resolver.py # 课程引用/字段抽取与解析（P-code、课程名、关键词表）
│   ├── router.py             # 规则路由兜底（classify_query）
│   └── evidence.py           # Evidence 数据类（retriever → generator → citation 的统一单位）
├── data/
│   ├── markdown/             # 64 份课程原始 Markdown（p02.md … p99.md）
│   ├── programmes.json       # 解析后的结构化课程数据
│   └── sample4multiQuery.json# 多轮对话 LangGraph checkpoint 样本
├── docs/                     # 详细文档（见下方"文档"）
├── test/                     # 测试与调试脚本
├── langgraph.json            # LangGraph server 配置（graph: cityu_agent）
└── requirements.txt
```

## 🚀 快速开始

### 环境要求

- Python 3.12+
- 一个 [OpenRouter](https://openrouter.ai/) API Key（用于 LLM 调用）
- 一个 [HuggingFace](https://huggingface.co/) Token（用于下载 BGE 嵌入模型权重）

### 1. 安装依赖

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 配置环境变量

创建 `.env` 文件（参考仓库根目录的 `.env.example`）：

```bash
OPENROUTER_API_KEY="sk-or-v1-..."
HF_TOKEN="hf_..."
```

> ⚠️ `.env` 已被 `.gitignore` 忽略，**切勿提交任何密钥**。

### 3. 构建向量索引（首次运行必需）

`rag/vectorstore/` 已被 gitignore，克隆仓库后需要先重建索引：

```bash
python rag/parser.py            # （可选）从 Markdown 重新解析 data/programmes.json
python rag/ingest.py            # 写入 programme_sections (338) + programme_summaries (64)
python rag/metadata_ingest.py   # 写入 programme_metadata (64)
```

### 4. 运行

**方式一：命令行（CLI）**

```bash
# one-shot 单次查询
python -m agent.cli "What is the tuition fee of MSc Computer Science?"

# 交互式 REPL（输入 'exit' 退出）
python -m agent.cli
```

**方式二：LangGraph 本地 server（供聊天 UI 接入）**

```bash
langgraph dev
```

server 启动后可通过 LangGraph API 以 `messages`（聊天历史）或 `query` 形式调用图 `cityu_agent`。

## 📖 使用示例

```bash
$ python -m agent.cli "What is the tuition fee of MSc Computer Science?"

[intent] qa  |  evidence 1  |  3.2s

[answer]

Answer:

- Local Students: HK$7,600 per credit
- Non-local Students: HK$9,100 per credit
```

> CLI 只打印回答正文；结构化引用（id / 课程 / 章节 / 置信度 / url）由 `result["citations"]` 与 `AIMessage.additional_kwargs.citations` 携带，供程序化调用与聊天 UI 渲染，不再以文本块形式拼进回答。

**支持的提问类型：**

| 类型 | 示例 | 检索路径 |
|---|---|---|
| 精确事实 | "What is the tuition fee of MSc Computer Science?" | metadata（结构化精确查找） |
| 详细信息 | "What are the entrance requirements of MA International Accounting?" | section |
| 课程推荐 | "I am interested in AI, recommend programmes" | summary |
| 复合问题 | "What are the English requirements and tuition fee of MSc Computer Science?" | section + metadata（dispatcher 扇出） |
| 多轮追问 | 第一轮问学费，第二轮 "what about English requirement?" | 自动继承上文课程引用 |

## 🔌 程序化调用（LangGraph API）

```python
from agent import app

# 方式一：直接传 query
result = app.invoke({"query": "What is the tuition fee of MSc Computer Science?"})

# 方式二：传聊天消息（chat UI 入口）
result = app.invoke({
    "messages": [{"type": "human", "content": "What is the tuition fee?"}]
})

# 返回结构（OutputState）
#   messages:       包含最终 AIMessage（additional_kwargs.citations 携带结构化引用）
#   final_response: 最终回答文本
#   citations:      结构化引用列表（id / programme_id / section / source_type / content / confidence / url）
```

调试内部状态（路由意图、证据等）可用 `stream_mode="values"` 观察。

## 🧪 测试

测试脚本位于 `test/`，从仓库根目录运行：

```bash
python test/test_graph.py            # 图结构 + 端到端流程
python test/test_router_compound.py  # 复合问题路由回归测试（真实 LLM 调用）
python test/test_retrieval.py        # 检索链路
python test/test_splitter.py         # 文档切分
python test/test_rag.py              # 检索结果展示
```

> `test/test_router_compound.py` 等会真实调用 OpenRouter，需要有效的 `OPENROUTER_API_KEY`。

## 📚 文档

| 文档 | 内容 |
|---|---|
| [docs/architecture_overview.md](docs/architecture_overview.md) | 系统架构总览、LangGraph 工作流、RAG 管线、数据模型与端到端案例 |
| [docs/layer_overview.md](docs/layer_overview.md) | 一条 query 的完整旅程：入口 → router → 三条检索路径 → generator → citation 的分阶段实现细节 |
| [docs/programme_schema.md](docs/programme_schema.md) | 课程页面信息 schema（A/B/C 三类内容分类） |
| [docs/compound_query_implementation.md](docs/compound_query_implementation.md) | 复合问题实现的四个阶段工作记录 |
| [docs/recommendation_scope_propagation.md](docs/recommendation_scope_propagation.md) | 推荐→费用复合问题的作用域传递（Plan A）工作记录 |
| [docs/multi_turn_conversation_analysis.md](docs/multi_turn_conversation_analysis.md) | 多轮对话指代解析机制分析 |
| [docs/git_workflow_guide.md](docs/git_workflow_guide.md) | 提交规范与分支管理手册（Contributing 必读） |

## 🤝 贡献

欢迎提交 Issue 与 Pull Request。请先阅读 [docs/git_workflow_guide.md](docs/git_workflow_guide.md) 了解本仓库的约定：

- **原子提交**：一次 commit 只做一件事，commit message 遵循 `<type>(<scope>): <subject>` 格式
- **短命分支**：从最新 master 拉出，合入后立即删除
- **master 可部署**：任何时刻 master 上的代码都应可运行、可测试
- **无敏感信息**：`.env`、密钥、token 一律不得提交

## 📄 License

本项目代码仅供学习与研究使用，未指定开源许可证。课程信息版权归香港城市大学（CityUHK）所有。

---

**免责声明**：本项目的回答基于 CityUHK 官网公开的课程信息生成，仅供申请参考；请以 [CityUHK 官网](https://www.cityu.edu.hk/) 发布的最新信息为准。
