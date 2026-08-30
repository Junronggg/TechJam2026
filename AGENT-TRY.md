# TechJam Agent 运行与自主性记录

> 本文件只记录 Research Agent 本身：规划、memory、controller、运行轨迹、时间预算、停止原因、人工介入、token 和 Agent ablation。
> 模型、loss、feature engineering、rolling 和 ensemble 分数见 [`TRY.md`](TRY.md)。

## 当前结论

- 端到端 autonomous loop 已实现：`observe → plan → execute → evaluate → reflect → update memory → repeat/stop`。
- Agent 可以从 baseline 自动运行到当前冠军，研究过程默认不读取 test。
- 首次完整 autonomous run 为 5 个实验、0 人工介入，并按 convergence rule 停止。
- 短程 memory ablation 中三种模式选择相同，因此不能声称 pattern 在短轨迹上有效。
- 旧版 broad family policy 在 replay 中避免过 2 个 temporal 实验；精确审计后确认 rolling 只否定“两项同时开启”，当前 scoped policy 只阻止这个已验证组合，不误伤尚未 rolling 的单特征。
- `family_policies` 现由 validation-only rolling/placebo/paired-seed artifacts 自动生成；每条策略带来源哈希、模型作用域、科学结论和比赛状态。
- 旧策略只在 task、model、feature schema 仍匹配时生效；artifact 改变后下次运行会重新归因，不再依赖人工同步 JSON 结论。

## Agent 闭环

```text
读取 validation-only evidence
→ 生成合法候选池
→ 计算 expected gain / evidence / novelty / cost / redundancy
→ 选择一个实验
→ 隔离执行训练与 validation
→ Critic 区分 observation 与 interpretation
→ 输出 KEEP / REJECT / STOP_DIRECTION / ENSEMBLE_ONLY / REINTERPRET
→ 必要时自动运行 placebo controls
→ 更新 experiment_history.jsonl 和 research_memory.json
→ 按收敛、时间、次数或搜索空间停止
```

关键审计输出：

| 文件/字段 | 用途 |
|---|---|
| `experiment_history.jsonl` | 每次实验的配置、指标、parent、判断和失败分支 |
| `research_memory.json` | validation-only hypotheses 与 family-level patterns |
| `candidate_selection` | 被选候选、top-5 候选和评分依据 |
| `counterfactual_choices` | 三种 memory mode 在同一时刻会选择什么 |
| `memory_changed_choice` | memory 是否真的改变本轮 action |
| `manual_interventions.jsonl` | 人工介入原因、动作及是否可避免 |
| `summary.json` | 时间、停止原因、实验数、token、人工介入和最终冠军 |

## Agent 运行记录

### Run A：Planner memory 真实消融

相同配置、数据、seed 和 `max_iterations=5`，不计算 test：

| Mode | Best Primary | Best iteration | 候选实验 | 无效候选 | 时间 | 人工介入 |
|---|---:|---:|---:|---:|---:|---:|
| `no_memory` | 0.604713 | 2 | 4 | 2 | 301.6 秒 | 0 |
| `raw_history` | 0.604713 | 2 | 4 | 2 | 287.8 秒 | 0 |
| `distilled_patterns` | 0.604713 | 2 | 4 | 2 | 281.5 秒 | 0 |

三种轨迹全部为：

```text
BPR → ensemble → like-only multitask → DCNv2
```

判断：**负结果。** Pattern 功能存在，但在这条短轨迹中没有改变 action；时间差只视为运行波动。

### Run B：首次完整 autonomous trajectory

命令：

```bash
python scripts/run_agent.py \
  --researcher deterministic \
  --memory-mode distilled_patterns \
  --max-iterations 50
```

运行目录：`logs/run_20260830T095900Z`

| Iteration | Agent action | Primary | 决策 |
|---:|---|---:|---|
| 0 | FM+BCE baseline | 0.601470 | REFERENCE |
| 1 | FM+BPR, lr=0.0003 | 0.603963 | KEEP_CANDIDATE |
| 2 | FM+BPR + DeepFM ensemble | **0.604713** | KEEP_CANDIDATE |
| 3 | Like-only Multi-task | 0.604400 | REJECT |
| 4 | DCNv2 | 0.604164 | REJECT |

| 审计项 | 结果 |
|---|---:|
| Stop reason | `converged` |
| Total / candidate experiments | 5 / 4 |
| Convergence streak | 3 |
| Wall clock | 280.9 秒 |
| Manual / avoidable interventions | 0 / 0 |
| LLM requests / tokens | 0 / 0 |
| Test metrics | `null` |
| Memory-influenced selections | 0 |

判断：端到端 autonomy 成立；这次运行本身没有证明 distilled memory 改善选择。

### Run C：跨 run memory 离线压力测试

这是引入自动 policy 生成前的历史结果；当时使用的是手写、未细分 config scope 的 broad family policy。

命令：

```bash
python scripts/replay_planner_memory.py --max-steps 12
```

数据来源：27 份历史 `experiment_history.jsonl`、116 条成功 validation 记录、34 个归一化配置。候选被选择前只检查是否有日志支持，不读取结果；test summary 不加载。

| Mode | Replay 实验 | 无效实验 | 最后选择 | 判断 |
|---|---:|---:|---|---|
| `no_memory` | 6 | 2 | temporal 0.605010 | 被单 split 小涨误导 |
| `raw_history` | 6 | 2 | temporal 0.605010 | 当前 run 原始历史不足以阻止重复 |
| `distilled_patterns` | 4 | 2 | ensemble 0.604713 | 跳过 2 个 rolling 已否定实验 |

Temporal 在单 split 上出现 `+0.000218`、`+0.000080`，但 rolling 只有 1/3 folds 提升、平均 `-0.000246`。Distilled memory 使用 `stop_direction` 保留稳健冠军。

判断：memory 第一次被证明会改变 trajectory，并减少两次无效实验。限制是 replay 重用历史结果，只验证 Planner 决策，不等于新的独立训练。

报告：`artifacts/planner_memory_replay.json`（生成文件，不提交）。

### Run D：Persistent evidence 接线后的 fresh run

命令：

```bash
python scripts/run_agent.py \
  --researcher deterministic \
  --memory-mode distilled_patterns \
  --max-iterations 5
```

运行目录：`logs/run_20260830T103503Z`

| Iteration | Agent action | Primary | 决策 |
|---:|---|---:|---|
| 0 | FM+BCE baseline | 0.601470 | REFERENCE |
| 1 | FM+BPR, lr=0.0003 | 0.603963 | KEEP_CANDIDATE |
| 2 | FM+BPR + DeepFM ensemble | **0.604713** | KEEP_CANDIDATE |
| 3 | Like-only Multi-task | 0.604400 | REJECT |
| 4 | DCNv2 | 0.604164 | REJECT |

| 审计项 | 结果 |
|---|---:|
| Stop reason | `max_iterations`（预先限制为 5） |
| Convergence streak | 3 |
| Wall clock | 309.1 秒 |
| Manual / avoidable interventions | 0 / 0 |
| LLM requests / tokens | 0 / 0 |
| Test metrics | `null` |
| Memory-influenced selections | 0 |

判断：所有分数精确复现，说明 persistent evidence 接线没有破坏训练和评估。短轨迹在停止前仍未进入 memory 会阻止的 family，因此 action 没有分叉；不能用 Run C 覆盖这个负结果。

### Run E：Artifact → policy 自动归因与 scoped replay

输入清单：`configs/evidence_manifest.json`。生成快照：`configs/generated_family_policies.json`。

```bash
python scripts/build_family_policies.py \
  --check-against configs/generated_family_policies.json
python scripts/replay_planner_memory.py --max-steps 12
```

自动生成 scoped policies；同一 family 在配置不同时可有多条互不误伤的策略：

| Family | 自动策略 | 科学结论 | 作用模型 | 主要依据 |
|---|---|---|---|---|
| heterogeneous ensemble | confirm/exploit | VALIDATED | ensemble | rolling 3/3，均值 +0.001123 |
| cross network | confirm/exploit | VALIDATED | DCNv2 | rolling 3/3，均值 +0.000248 |
| global context | gather evidence | UNCERTAIN | FM | rolling 3/3，但 paired-seed 区间跨 0 |
| global context | gather evidence | UNCERTAIN | ensemble | rolling 3/3，但 official split 为轻微负值 |
| multitask like-only | confirm/exploit | VALIDATED | multitask DeepFM + BCE | like-only rolling 3/3，均值 +0.000309 |
| pairwise multitask | stop | REJECTED | BPR + like 的两个精确配置 | 分别 -0.001079 / -0.000791 vs BCE |
| candidate history | stop | REJECTED | FM | 两个 placebo 失败，count/adjacency 也为 noise |
| sequence model | stop | REJECTED | sequence DeepFM | -0.000493 且成本高 |
| temporal counts | stop | REJECTED | ensemble + 两项 temporal 同开 | rolling 1/3，均值 -0.000246 |

最新 scoped replay 得到：`no_memory/raw_history` 跑 6 个实验并依次加入两项 temporal；`distilled_patterns` 跑 5 个实验，只允许第一项单特征，在准备形成已被 rolling 否定的双特征组合时停止。因此少执行 1 个已有直接反证的实验，但 best 仍被单 split 的 `0.604931` 小涨影响。全过程标记 `validation_only=true`、`test_metrics_loaded=false`。

实现判断：这一步把“人阅读 TRY.md 后手写 stop list”改成“Agent 从结构化实验产物更新 planning policy”。同时它揭示了证据粒度缺口：组合实验的失败不能自动推广为每个单特征都失败。下一步应让 slow-confirmation 自动对 `user_recent_3d_activity` 单项做 rolling，再决定是否停止整个 family。它是 evidence-driven policy update，不是 RL，也不代表自动发现了新的模型实现。

### Run F：扩展 action 后验证 pairwise multi-task

Agent action space 新增 `pairwise_multitask`：同一个 MultiTask DeepFM 中用 BPR 学 long-view 用户内排序，并继续用 like BCE 辅助共享表示。

| 配置 | Primary | 相对 like+BCE | 自动结论 |
|---|---:|---:|---|
| k16 / aux0.1 / lr0.001 | 0.603322 | -0.001079 | STOP_DIRECTION（精确 scope） |
| 外部报告 k32 / aux0.3 / lr0.001 | 0.603610 | -0.000791 | STOP_DIRECTION（精确 scope） |

这两条结果已进入 evidence manifest，分别绑定 model、objective、embedding、auxiliary signal/weight 和 learning rate，不会阻止未来采用不同 target 或 sampler 的新 pairwise mechanism。Agent 因此能执行该能力，也能根据自己的 validation 结果停止重复配置。

## Memory 机制

`research_memory.json` 包含两层：

```text
hypotheses:
  单次实验的 validation 证据、诊断与结论

research_patterns:
  按 family 蒸馏的可复用策略
```

当前策略：

| Policy | 自动行为 |
|---|---|
| `exploit_with_confirmation` | 有正向证据；要求 rolling 或 paired seeds 确认 |
| `ensemble_only` | 不替换冠军，只做预声明的互补性/融合检查 |
| `retest_with_control` | 先运行 constant、shuffled、same-cardinality controls |
| `gather_evidence` | 只允许一个便宜单变量实验 |
| `stop_direction` | 不再重复等价的失败方向 |

Persistent validation memory 位于 `configs/research_evidence.json`。`scripts/run_agent.py` 每次启动还会读取 `configs/evidence_manifest.json`，从明确列出的 validation artifacts 重新生成 scoped policies，并覆盖相同 family 的旧手写 policy。Deterministic 与 LLM Researcher 使用同一份合并证据；test 指标不能进入规划证据。

每条自动策略包含：

```text
policy_id
scientific_verdict / competition_status
applies_to: task + feature_schema + models
expires_if
created_from: artifact path + sha256 + extracted validation result
```

## 论文机制迁移

| 来源 | 借鉴内容 | 本项目实现 | 没有照搬 |
|---|---|---|---|
| RecMind | Self-Inspiring / 回顾完整探索路径 | 成功、失败、noise、placebo 和 side branch 都进入 memory | 让 LLM 直接进行大候选集推荐 |
| TAIRA | Manager、层级规划、Thought Pattern Distillation | Controller/Runner 分工；family-level patterns | 多个昂贵 LLM Agent |
| STARec | fast/slow deliberate reasoning | Runner 执行；Critic/Controller 归因后才写长期结论 | SFT、GRPO 或声称复现论文 memory validation |
| SAPIENT | exploration/exploitation 与规划 | 小型 tree frontier、budget-aware candidate scoring | 完整 MCTS 和昂贵真实训练 rollout |

我们的表述应为：系统受这些机制启发，不是四篇论文实现的拼接或严格复现。

## 下一步 Agent 工作

优先级：

1. 用 fresh autonomous run 验证自动生成的 policy 能减少真实训练次数；若官方 convergence 更早触发，应如实记录无法观察分叉。
2. 将 slow confirmation 从 planner 标签升级为可执行的 rolling/paired-seed job，而不是只做优先级加分。
3. 若要继续 pairwise multi-task，先取得或明确重建对方的 loss、auxiliary target 和 sampler，不能继续猜参数。
4. 再比较 cheap one-step lookahead 与当前 greedy planner。

暂不宣称：

- Agent 已做 RL 或参数级 self-learning。
- LLM 本身提高了推荐 Primary。
- 短程 memory ablation 已证明 pattern 有效。
- Offline replay 等价于 fresh independent trials。

## Agent 复现入口

```bash
# 标准 deterministic Agent
python scripts/run_agent.py --researcher deterministic

# 三种 memory mode
python scripts/run_agent.py --researcher deterministic --memory-mode no_memory
python scripts/run_agent.py --researcher deterministic --memory-mode raw_history
python scripts/run_agent.py --researcher deterministic --memory-mode distilled_patterns

# 一次执行三组真实消融
python scripts/run_memory_ablation.py --max-iterations 5

# 不重新训练的历史 replay
python scripts/replay_planner_memory.py --max-steps 12

# 验证自动策略快照与当前 artifacts 一致
python scripts/build_family_policies.py \
  --check-against configs/generated_family_policies.json

# LLM planner（需要 API key）
python scripts/run_agent.py --researcher llm --model gpt-4.1-mini
```

## 对应提交

- `e18e652 feat: add evidence-driven autonomous research loop`
- `622b985 feat: apply persistent evidence to experiment planning`
