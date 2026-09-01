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
- 官方 convergence 与内部研究探索已分开：默认仍按 `epsilon=0.002, rounds=3` 停止；显式研究模式会记录官方收敛点后继续探索，不能改写该收敛事实。
- LLM 现在只能原样选择 deterministic top-5 候选中的完整 `changes`，不能漏掉学习率或自行拼配置；prompt 同时获得 train-only dataset facts 与方法适用条件。
- Planner 新增了通用的 ensemble-calibration 候选（`fm_zscore_deepfm_rank`），以及
  label-free static-metadata capabilities（`video_music_type`、`video_tag_components`）。
  这不是把 action space 固定成 BPR/LambdaRank 等模型菜单，而是把可执行能力接入同一
  配置搜索和 evidence 流程。后来又修复了一个过早停止点：ensemble family 的旧 noisy
  变体不再压制尚未测试的 calibration weight；本轮又把 `0.63/0.64` 邻域和延后的反向
  calibration 接入 Planner，代码通过 `192 passed`。

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
| `submission_candidates.json` | 科学结论与提交资格分离；被否定或 control 配置不会因一次分数进入提交池 |

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

### Run F：修复过早停止后的自主校准轨迹

命令：

```bash
python scripts/run_agent.py \
  --researcher deterministic \
  --max-iterations 5 \
  --research-after-convergence
```

运行目录：`logs/run_20260831T124346Z`

| Iteration | Agent action | Primary | 决策 |
|---:|---|---:|---|
| 0 | FM+BCE baseline | 0.601470 | REFERENCE |
| 1 | FM+BPR, lr=0.0003 | 0.603963 | KEEP_CANDIDATE |
| 2 | FM+BPR + DeepFM ensemble, z-score, weight=0.4 | 0.604713 | KEEP_CANDIDATE |
| 3 | FM z-score + DeepFM rank calibration | 0.604746 | KEEP_CANDIDATE |
| 4 | **Agent 自动跟进 DeepFM weight=0.65** | **0.605291** | **KEEP_CANDIDATE** |

| 审计项 | 结果 |
|---|---:|
| Stop reason | `max_iterations` |
| Official convergence point | iteration 4，streak=3 |
| Total / candidate experiments | 5 / 4 |
| Manual interventions | 0 |
| Test metrics | `null` |
| Best Primary | **0.6052911282** |

这条轨迹证明两点：第一，Planner 能根据上一轮 calibration 结果自动提出下一步 weight follow-up；第二，之前没有达到 `0.605` 不完全是模型上限，也有 Planner 把未测变体过早挡掉的问题。该分数仍需 rolling 和 paired-seed confirmation，不能直接宣称泛化提升。

### Run G：Rank calibration paired-seed confirmation

为验证 Run F 的 `0.605291` 候选，固定参考配置为 `FM+BPR + DeepFM+BCE`、z-score、
DeepFM weight `0.4`；候选配置只改变为 `fm_zscore_deepfm_rank`、weight `0.65`。
两边在相同 seeds `0–3` 下重训，未读取 test label。

| Seed | Reference | Candidate | Delta |
|---:|---:|---:|---:|
| 0 | 0.604713 | 0.605291 | +0.000578 |
| 1 | 0.604061 | 0.604734 | +0.000673 |
| 2 | 0.604362 | 0.604723 | +0.000362 |
| 3 | 0.604435 | 0.604433 | -0.000002 |

汇总：`3/4` seeds 为正，paired mean delta **+0.000403**，paired std **0.000300**，
近似 95% 区间 **[-0.000074, +0.000880]**，共 8 次模型训练、约 494 秒。
因此自动证据状态为 `UNCERTAIN / ELIGIBLE`：它可以进入提交候选池，但不能写成
“已确认的统计提升”，稳健 fallback 仍是 `0.604713` 的 rolling 3/3 ensemble。

### Run H：局部 calibration 搜索空间扩展

发现原配置只允许 `0.3/0.4/0.5/0.6/0.65/0.7` 六个融合权重。保持 FM 和
DeepFM checkpoint 完全不变，增加 `0.63/0.64` 两个邻域值并评估三种校准方向。
官方 validation 上 `fm_zscore_deepfm_rank + weight=0.63` 达到 **0.605365**，
高于旧峰值 `0.605291`；rolling 的增量为 `+0.000843/-0.000484/+0.000531`
（2/3、均值 `+0.000297`）。复用 Run G 四个 seed 的组件 checkpoint 做 paired
weight check 时，`0.63` 为 4/4 正、均值 `+0.000430`，但近似 95% 区间
`[-0.000061, +0.000922]` 仍跨 0。因此它是更强的 validation 提交候选，不能替换
rolling 3/3 的 `0.604713`，也不能只凭局部扫描宣称统计确认。

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

加入新日志后重新 replay：archive 包含 30 份 history、159 行、128 条成功 validation 记录和 37 个归一化配置。`no_memory` 跑 7 个、`raw_history` 跑 6 个，二者都会形成两项 temporal 同开的 `0.605010`；`distilled_patterns` 跑 6 个，只允许第一项 temporal，随后选择已有日志支持的 DCNv2 分支，并阻止形成 rolling 已否定的双 temporal 配置。Distilled best 仍会被单 split 的 `0.604931` 小涨影响，因此这只证明 scoped memory 改变路径、避开一个已知坏组合，不证明它自动找到更高冠军。全过程标记 `validation_only=true`、`test_metrics_loaded=false`。

实现判断：这一步把“人阅读 TRY.md 后手写 stop list”改成“Agent 从结构化实验产物更新 planning policy”。同时它揭示了证据粒度缺口：组合实验的失败不能自动推广为每个单特征都失败。下一步应让 slow-confirmation 自动对 `user_recent_3d_activity` 单项做 rolling，再决定是否停止整个 family。它是 evidence-driven policy update，不是 RL，也不代表自动发现了新的模型实现。

### Run F：扩展 action 后验证 pairwise multi-task

Agent action space 新增 `pairwise_multitask`：同一个 MultiTask DeepFM 中用 BPR 学 long-view 用户内排序，并继续用 like BCE 辅助共享表示。

| 配置 | Primary | 相对 like+BCE | 自动结论 |
|---|---:|---:|---|
| k16 / aux0.1 / lr0.001 | 0.603322 | -0.001079 | STOP_DIRECTION（精确 scope） |
| alternative config k32 / aux0.3 / lr0.001 | 0.603610 | -0.000791 | STOP_DIRECTION（精确 scope） |

这两条结果已进入 evidence manifest，分别绑定 model、objective、embedding、auxiliary signal/weight 和 learning rate，不会阻止未来采用不同 target 或 sampler 的新 pairwise mechanism。Agent 因此能执行该能力，也能根据自己的 validation 结果停止重复配置。

### Run G：OpenRouter LLM Researcher 端到端接入

本地配置保存在 git-ignored `.env`；真实 key 不进入代码、`.env.example`、运行日志或提交。先运行无训练连接检查，再执行两轮 validation-only autonomous run：

```bash
python scripts/check_llm_connection.py
python scripts/run_agent.py --researcher llm --max-iterations 2
```

运行目录：`logs/run_20260830T142119Z`

| 审计项 | 结果 |
|---|---:|
| Provider / model | OpenRouter / `openai/gpt-4.1-mini` |
| LLM requests / failures | 1 / 0 |
| Deterministic fallbacks | 0 |
| Prompt / completion / total tokens | 8158 / 135 / 8293 |
| Manual interventions | 0 |
| Agent action | FM BCE → FM BPR, lr=0.0003 |
| Baseline / candidate Primary | 0.601470 / 0.603963 |
| Delta | +0.002493 |
| Stop reason | `max_iterations`（预先限制为 2） |

LLM 读取结构化 validation memory、合法 action space、候选评分和剩余预算后，选择了与 deterministic planner 相同的首个动作。这个结果证明 LLM 路径真实可用，并能产生合法、可执行且有证据支持的 hypothesis；它不证明“接 API 本身提高模型分数”，因为提升来自被选择的 BPR 训练方案。

首次受限网络运行 `logs/run_20260830T141747Z` 发生 1 次 LLM failure 后自动回退到 deterministic。随后增加了安全连接检查和 `llm_fallbacks` 审计字段，使网络/API 故障不再被误认为 LLM 自主选择。

### Run H：LLM 长程自主运行与双层收敛

命令：

```bash
python scripts/run_agent.py \
  --researcher llm \
  --max-iterations 8 \
  --research-after-convergence
```

运行目录：`logs/run_20260830T144609Z`。全程 validation-only，未计算 test，未记录人工介入。

| Iteration | 来源 | 动作 | Primary | 相对当前冠军的结果 |
|---:|---|---|---:|---|
| 0 | system | FM+BCE baseline | 0.601470 | reference |
| 1 | LLM | FM+BPR, lr=0.0003 | 0.603963 | 新冠军 |
| 2 | LLM | 0.6/0.4 ensemble | **0.604713** | 新冠军 |
| 3 | LLM | Like Multi-task，但沿用父节点 lr=0.0003 | 0.603831 | reject |
| 4 | deterministic fallback | Like Multi-task, lr=0.001 | 0.604400 | reject |
| 5 | LLM | BPR + user recent activity | 0.604419 | 低于冠军 |
| 6 | LLM | temporal parent + DCNv2 | 0.604332 | stop exact direction |
| 7 | LLM | temporal parent + ensemble weight 0.5 | 0.604694 | 低于冠军 0.000019 |

| 审计项 | 结果 |
|---|---:|
| Official competition convergence | iteration 4，冠军仍为 iteration 2 |
| Research after convergence | 继续执行 iteration 5–7 |
| Stop reason | `max_iterations` |
| LLM requests / failures / fallback | 11 / 1 / 1 |
| Prompt / completion / total tokens | 122139 / 1433 / 123572 |
| Wall clock | 809.7 秒 |
| Manual interventions | 0 |
| Test metrics | `null` |

这次运行证明 Agent 能在官方收敛点后，按预设研究预算继续选择和执行分支；额外三轮没有超过冠军，所以不能把“继续研究”写成涨分。它也暴露出旧 LLM contract 的缺陷：iteration 3 选择了 Like 机制，却漏掉证据配置中的 `learning_rate=0.001`，随后三次修复失败并触发 fallback。

Run H 之后做了两项代码修正：

1. LLM 必须原样返回 ranked candidate 的完整 `changes`；任何增删字段都会自动重试。
2. `submission_candidates.json` 默认只允许 reference/KEEP/ENSEMBLE_ONLY；只有带明确 `competition_status=ELIGIBLE` 的 scoped evidence 才能让科学上未确认的配置进入候选池。Run H 旧文件中宽松的 candidate count 不作为最终提交策略证据。

因此，Run H 是真实长程轨迹和缺陷发现记录，不能倒推声称旧轨迹当时已具有这些保护。修正后另做 live smoke run：`logs/run_20260831T010826Z`。LLM 用 1 次请求原样选择 `training_objective=bpr + learning_rate=0.0003`，无失败、无 fallback、0 人工介入，Primary 复现 `0.603963`，test 为 `null`；prompt/completion/total tokens 为 `10497 / 94 / 10591`。这验证了新 contract 的真实 API 路径。

### Run I：自动 Evidence Escalator

这次升级解决的是 Agent 工作流缺口，不是新增模型或 feature，因此不记入 `TRY.md` 的排行榜。过去 planner 只能把候选标成“需要 rolling / paired seeds”；现在 Controller 可以把确认要求转成真实可执行 action：

```text
single-split discovery
→ matched placebo（若该 feature 需要归因控制）
→ 3-fold expanding-window rolling
→ paired seeds 0/1/2/3
→ VALIDATED / UNCERTAIN / REJECTED
```

启动方式：

```bash
python scripts/run_agent.py \
  --researcher llm \
  --research-after-convergence \
  --auto-confirm
```

固定规则：

- 普通小增益低于 `0.0002` 时不浪费 confirmation 预算；高 novelty 或已有 diversity advantage 可进入确认。
- rolling 至少 3 folds、其中至少 2 folds 上涨且平均 delta 为正，才进入 paired seeds。
- paired-seed 均值或多数 seed 不为正则 `REJECTED`；均值为正但区间跨 0 为 `UNCERTAIN + ELIGIBLE`；区间完全为正才为 `VALIDATED + ELIGIBLE`。
- confirmation 不更新冠军 checkpoint，也不重置官方 convergence streak；它单独记录 action 数、底层 training-run 数和用时。
- 所有确认必须显式返回 `test_labels_used=false`，生产路径仍校验官方 evaluator hash。

真实数据 smoke check 使用通用 confirmation executor 对 `FM+BPR` 与 `FM+BPR+global_context` 做了 rolling 对照：

| 项目 | 结果 |
|---|---:|
| Rolling wins | 3 / 3 |
| Mean delta | **+0.000810** |
| Training runs | 6 |
| Runtime | 232.1 秒 |
| Test labels | 未使用 |

它与已有专用脚本结果一致，证明通用执行器能正确执行任意合法 config 的 reference/candidate 对照；这不是新的 `0.604713` 以上结果。自动化测试覆盖 discovery 筛选、rolling 失败停止、rolling 后 paired seeds、区间跨 0 的双状态、已有 artifact 去重、预算计数和提交池状态。

### Run J：Prompt / Skill / Controller 分层

这次是架构重构，不是 feature engineering，因此同样不改 `TRY.md` 的模型排行榜。

```text
Prompt
= 研究原则与判断方式（软约束）

Skill Registry
= 当前真正可执行的实验能力

Controller
= test、预算、超时、evaluator、能力绑定等硬约束
```

Prompt 独立定义以下原则：只使用 train/validation、优先新 information source、尊重 `STOP_DIRECTION`、小涨需要 confirmation、新 categorical history feature 需要 placebo、ensemble 前检查 diversity，并区分 observation / interpretation。

第一版 Registry 只注册 10 个复用能力：

| 类别 | Skills |
|---|---|
| Discovery | `profile_candidate`, `build_feature` |
| Training | `train_model`, `train_with_auxiliary_loss` |
| Evidence | `run_placebo`, `run_rolling`, `run_paired_seeds`, `analyze_prediction_diversity` |
| Memory | `read_research_memory`, `update_research_memory` |

每个 skill 都有稳定 `skill_id`、owner、真实 handler、状态和 `test_labels_allowed=false`。每个 ranked candidate 现在额外包含：

```text
skill_id
required_confirmation
risk
```

Planner 选择后会生成固定的 audited decision record：

```text
hypothesis
mechanism_basis
family
proposed_action
expected_gain
novelty
risk
required_confirmation
```

其中 expected gain、family、skill 和 confirmation 由可信 registry/ranking 填入，不要求 LLM 自报，避免 LLM 修改或编造这些字段。Controller 执行前重新解析 config 对应的 skill；未注册 skill 或绑定不一致会返回 `missing_capability` / `invalid_skill_binding`，不会开始训练。Run metadata 和 summary 也会记录 registry version 与 available skills。

当前 `capability_builder_enabled=false`，并明确记录 `train_graph`、`build_new_model_family` 是能力缺口。也就是说系统已经能区分“不会做”和“实验失败”，但还不能让 LLM 自动写新模型代码。这是下一阶段受限 Capability Builder 的输入接口，而不是伪装成已经完成。

验证结果：全套 **191 tests passed**，原 deterministic 首选仍是 BPR，已有模型结果与冠军 `0.604713` 没有改变。

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

`configs/research_context.json` 另外提供 train-only 数据事实和 method conditions。例如 BPR 只在同用户存在正负样本且按用户排序时优先；censored watch-time 明确区分 exact observation 与 completed-play lower bound。这是有来源范围的研究上下文，不是把人工答案或 test 结果硬编码进 prompt。

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

1. 当已注册动作不足时，让 planner 输出结构化 `capability_need`，明确缺少的信息源、接口和验证标准，而不是幻觉式选择不存在的 skill。
2. 实现受限 Capability Builder：只能填充预定义 model/schema/runner/test 模板，通过 smoke test 后才允许注册；仍不允许 LLM 任意修改仓库。
3. 现有 leakage-safe lightweight sequence 已进入 `train_model` 能力；下一个真正不同的缺口优先考虑 LightGCN/graph family。
4. 做 planner replay 的 pre/post contract 对照，并比较 budgeted one-step lookahead 与当前 greedy planner。

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

# LLM planner（先在 git-ignored .env 配置 API key）
python scripts/check_llm_connection.py
python scripts/run_agent.py --researcher llm

# 记录官方收敛点后继续使用剩余研究预算；最终比赛演示需预先声明该模式
python scripts/run_agent.py --researcher llm --research-after-convergence

# 同时让 promising discovery 自动执行 rolling → paired-seed confirmation
python scripts/run_agent.py --researcher llm --research-after-convergence --auto-confirm
```

## 对应提交

- `e18e652 feat: add evidence-driven autonomous research loop`
- `622b985 feat: apply persistent evidence to experiment planning`
