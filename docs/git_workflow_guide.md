# CityU-Agent Git 使用手册（初学者版）

> 适用范围：所有向本仓库提交代码的开发者（含新手）
> 目标：统一 commit 与分支的**必要条件**和**命名规范**，让历史可追溯、可回滚、可协作
> 一句话总结：**一次提交只做一件事；一个分支只做一个任务；master 永远可运行。**

---

## 0. 四条核心原则（先读这一段）

1. **原子提交**：一个 commit 只包含一个逻辑单元（一个功能 / 一个修复 / 一次文档更新），不要混装无关改动。
2. **短命分支**：分支从最新的 master 拉出，几天内完成、合入、删除，不要长期挂起。
3. **master 可部署**：master 上的任何时刻，代码都应该是可运行、可测试的状态。所有开发都在分支上进行。
4. **合完即删**：分支合并进 master 后立即删除，避免 stale 分支堆积造成混淆。

---

## 1. Commit 规范

### 1.1 什么时候可以提交（必要条件）

以下条件**全部满足**才可以 commit：

| # | 必要条件 | 说明 | 检查命令 |
|---|---|---|---|
| 1 | **一个逻辑单元** | 只包含一件事：一个功能 / 一个修复 / 一次文档更新 | `git diff` 预览改动范围 |
| 2 | **代码可运行** | 改动涉及的代码无语法错误、测试通过 | `pytest` / 运行对应模块 |
| 3 | **无意外文件** | `.env`、`.venv`、`vectorstore/`、`__pycache__` 不得出现在改动中；出现即说明 `.gitignore` 有问题 | `git status` |
| 4 | **无敏感信息** | 不含 API key、token、密码、本地绝对路径、个人邮箱 | `git diff` 肉眼检查 |
| 5 | **信息符合规范** | 提交信息按 §1.2 的格式书写 | — |

> ⚠️ 出现以下情况**不要**提交：
> - 工作做到一半、代码跑不起来
> - 临时调试代码（`print` 调试、注释掉的死代码）
> - 与当前任务无关的格式改动或重构
> - 密钥、`.env`、大文件、生成物

### 1.2 提交信息格式

```text
<type>(<scope>): <subject>

<body>

<footer>
```

| 部分 | 规则 |
|---|---|
| `type` | 必填，见 §1.3 的 type 表 |
| `scope` | 可选，改动模块，见 §1.4；不填时省略括号 |
| `subject` | 必填，一句话概括改动。祈使句、英文、≤ 50 字符、结尾不加句号 |
| `body` | 可选，说明**为什么**这么做、影响范围 |
| `footer` | 可选，`BREAKING CHANGE: ...` 或关联 issue（`Closes #12`） |

**示例：**

```text
feat(router): add LLM semantic router

Replace the keyword-based rule router with an LLM-based
semantic router to support intent classification.
```

### 1.3 type 一览表（必须使用这些值）

| type | 含义 | 示例 |
|---|---|---|
| `feat` | 新功能 | `feat(router): emit sub-decisions for compound queries` |
| `fix` | 修 bug | `fix(rag): correct programme_ref resolution` |
| `docs` | 文档 | `docs: add git workflow guide` |
| `refactor` | 重构，行为不变 | `refactor(state): split AgentState into Input/Output` |
| `perf` | 性能优化 | `perf(retriever): cache vector search results` |
| `test` | 增改测试 | `test(router): add unit tests` |
| `chore` | 杂务：依赖、配置 | `chore: bump langgraph version` |
| `style` | 格式调整，不影响逻辑 | `style: reformat with black` |
| `build` | 构建相关 | `build: update requirements.txt` |
| `ci` | CI 配置 | `ci: add pytest workflow` |
| `revert` | 回滚某次提交 | `revert: restore rule-based router` |

> ❌ **禁止**把 `add`、`modify`、`implement`、`checkpoint`、`update`、`align` 这类模糊动词当 type 用——它们不表达"这是哪种改动"。

### 1.4 scope 一览表（本仓库常用模块）

| scope | 范围 |
|---|---|
| `state` | AgentState 定义 |
| `router` | 路由节点 |
| `graph` | LangGraph 图定义 |
| `rag` | 解析 / 构建 / 检索 |
| `cli` | 命令行入口 |
| `docs` | 文档 |

### 1.5 正反例对照（本仓库真实案例）

**✅ 符合规范：**

```text
fix: programme_id="programme name" bug, change to programme_ref
refactor(state): split AgentState into Input/Output states; move Citation into agent/state
feat: phase 3 - unified QA prompt for mixed-evidence answers
```

**❌ 不规范及改法：**

| 原提交（本仓库真实） | 问题 | 应改为 |
|---|---|---|
| `checkpoint: switch to LLM semantic router` | `checkpoint` 不是合法 type | `feat(router): switch to LLM semantic router` |
| `add: multi turn conversation` | `add` 不是合法 type | `feat: add multi-turn conversation support` |
| `modify: citation logic` | `modify` 不表达改动类型 | `fix(citation): correct citation logic`（若是修复）或 `refactor(citation): ...`（若是重构） |
| `docs+fix: phase 4 - document v2 compound-query architecture` | 一个 commit 塞了两个 type | 拆成两个 commit：`docs: ...` 和 `fix: ...` |
| `align with chat-ui, citation schema` | 没有 type，语义模糊 | `feat: align citation schema with chat-ui` |

---

## 2. 分支规范

### 2.1 命名格式

```text
<type>/<简短描述>
```

| 规则 | 说明 |
|---|---|
| 前缀 | 与 commit type 对应：`feature/`、`fix/`、`refactor/`、`docs/`、`chore/`、`hotfix/`、`release/` |
| 描述 | 2~5 个英文单词，kebab-case（小写 + 连字符），名词短语或动宾短语 |
| issue 号（可选） | 有关联 issue 时加在描述前：`feature/12-compound-query` |

**✅ 正确示例：**

```text
feature/compound-query
fix/programme-ref-bug
refactor/llm-router
docs/git-guide
```

**❌ 错误示例（本仓库真实）及改法：**

| 原分支名 | 问题 | 应改为 |
|---|---|---|
| `ruleRouter2LLMRouter` | 无前缀、camelCase、含义模糊（本质是重构） | `refactor/llm-router` |
| `sub-query4multi-question` | 无前缀、数字代替单词 | `feature/compound-query` |

> ❌ 禁止：用日期命名、`final`/`final2`/`v2` 后缀、中文、驼峰、超长描述。

### 2.2 一个分支只做一个任务

分支 = 一个 PR 单元。一个分支里**不要**同时做"重构 + 新功能 + 文档"——合入 master 时无法单独 review 或回滚其中任何一部分。任务拆不开时，拆成多个分支依次合入。

### 2.3 分支生命周期

```text
master ────────────────► 合入并删除后 ◄── feature/xxx（已完成使命）
   ▲                                    ▲
   │ 永远可部署                          │
   └─────── 从最新 master 拉出 ──────────┘
```

1. 从**最新** master 拉出分支；
2. 分支保持短命（1~5 天），期间若 master 有更新，rebase 到最新 master；
3. 完成后推送到远端，开 PR（Pull Request）；
4. CI 通过 + review 通过 → 合入 master；
5. **立即删除本地和远端分支**。

### 2.4 分支套分支（stacked branches，慎用）

"从功能分支再拉子分支"（如你之前的 `ruleRouter2LLMRouter` → `sub-query4multi-question`）是合法但高风险的模式。生产环境纪律：

- **父分支先合入 master，子分支 rebase 到新 master** 再继续开发；
- 更稳妥的替代：**等父分支合入后，直接从 master 拉新分支**；
- 多人协作时，父分支一旦被 rebase / force-push，所有子分支都会继承重复 commit 甚至冲突。

---

## 3. 完整工作流（新手照着做）

### 3.1 首次克隆

```bash
git clone https://github.com/xinyi959/CityU-Agent.git
cd CityU-Agent
```

### 3.2 日常开发流程

```bash
# ① 同步最新 master
git checkout master
git pull --ff-only

# ② 创建功能分支
git checkout -b feature/my-task

# ③ 开发……然后按需提交（每完成一个小逻辑单元就提交一次）
git add <具体文件1> <具体文件2>      # 逐个 add，不要无脑 git add .
git commit -m "feat(xxx): concise description"

# ④ 推送到远端（首次加 -u 建立跟踪）
git push -u origin feature/my-task

# ⑤ 在 GitHub 上开 PR → 等待 CI + review → 合并（squash merge）

# ⑥ 收尾：切回 master、拉取、删除本地与远端分支
git checkout master
git pull --ff-only
git branch -d feature/my-task
git push origin --delete feature/my-task
```

### 3.3 合并策略（选一种并保持一致）

| 策略 | 效果 | 适用 |
|---|---|---|
| **Squash merge**（默认推荐） | master 每个 commit = 一个完整任务，历史最干净 | 常规功能 |
| **Merge commit（`--no-ff`）** | 保留分支拓扑，能回溯"该功能何时合入" | 需要按 PR 粒度回溯的大型改动 |
| ~~直接 push master~~ | ❌ 禁止 | 开发必须在分支上进行 |

### 3.4 合入后的同步

PR 合并后，本地 master 落后于远端时：

```bash
git checkout master
git pull --ff-only        # 快速前进，不产生多余 merge commit
```

---

## 4. 常见错误与速查表

### 4.1 新手常见错误

| 错误 | 后果 | 正确做法 |
|---|---|---|
| 直接在 master 上 commit | 污染可部署分支，无法 review | 开发一律在分支上 |
| 提交了 `.env` | 密钥泄露到远端，无法撤回 | 确认 `.gitignore` 包含 `.env`；已误提交用 `git rm --cached .env` |
| 一次 commit 塞太多改动 | 无法单独回滚 | 原子提交（§1.1） |
| 提交信息随意写 | 历史无法追溯 | 按 §1.2 格式 |
| `git add .` 无脑全加 | 带进垃圾文件和调试代码 | 逐个 `git add <文件>` |
| 分支合完不删 | 仓库堆积 stale 分支 | 合完即删（§2.3） |
| push 前不 pull | 产生不必要的冲突 | 先 `git pull --ff-only` |
| 提交半成品 | 队友拿到跑不起来的代码 | 自测通过再提交 |

### 4.2 命令速查表

| 场景 | 命令 |
|---|---|
| 查看状态 | `git status` |
| 暂存文件 | `git add <file>` |
| 提交 | `git commit -m "type(scope): subject"` |
| 拉取最新 | `git pull --ff-only` |
| 创建分支 | `git checkout -b feature/xxx` |
| 推送分支 | `git push -u origin feature/xxx` |
| 查看历史 | `git log --oneline --graph` |
| 查看某次提交 | `git show <commit>` |
| 撤销暂存 | `git restore --staged <file>` |
| 放弃未暂存改动 | `git restore <file>` |
| 删除本地分支 | `git branch -d feature/xxx` |
| 删除远端分支 | `git push origin --delete feature/xxx` |

---

## 5. 进阶（可选，熟悉后再用）

- **commitlint + husky**：在 commit 时强制校验提交信息格式，不符合直接拦截；
- **`.gitmessage` 模板**：`git commit` 时自动带出格式模板，减少手写错误；
- **CHANGELOG 自动化**：基于 Conventional Commits 用 `conventional-changelog` 自动生成变更日志；
- **分支保护**：在 GitHub Settings → Branches 中禁止直接 push master、强制 PR + CI 通过。

---

## 附：提交前 30 秒自查

```text
□ 这个 commit 只做了一件事？
□ git status 里没有 .env / 缓存 / 垃圾文件？
□ 代码跑过了、测试通过了？
□ 提交信息是 <type>(<scope>): <subject> 格式？
□ 没有密钥、绝对路径等敏感信息？
```

全部打勾 → 提交。否则 → 先处理再提交。
