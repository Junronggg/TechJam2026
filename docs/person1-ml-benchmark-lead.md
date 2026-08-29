# Person 1 — ML / Benchmark Lead Responsibilities and Delivery Status

## English Version

> Role objective: build a trustworthy, reproducible, leakage-safe ML research environment. Person 1 defines what the agent can safely research; Person 1 does not manually choose every experiment for the agent.

For the detailed model, parameter, feature, and optimization inventory, see [`models-features-optimization-guide.md`](models-features-optimization-guide.md). For the focused BPR experiment, see [`bpr-experiment.md`](bpr-experiment.md).

### 1. Overall Progress

| Task | Status | Current evidence | Main gap |
|---|---|---|---|
| P1.1 Benchmark ownership | ✅ Mostly complete | Official FM reproduced; metrics and splits documented | Clean-environment reproduction record |
| P1.2 Stable training interface | 🟡 Partial | `src/techjam_agent/runner.py` runs FM and LightGBM | `recommender/train.py` is not connected; two frameworks remain |
| P1.3 Feature registry | 🟡 Partial | Two encoders support multiple safe features | Missing cache keys, unified fit/transform contract, and temporal features |
| P1.4 Item historical features | ✅ Minimum accepted | Popularity and item rate/count | No gain yet; recency and author priors remain |
| P1.5 Personalization | ✅ Minimum accepted | User activity/rate and user×tab/tag features | Real user×tag experiment remains |
| P1.6 Objectives | ✅ Complete | BCE/BPR are configurable with a controlled comparison | Multi-seed BPR and sampling parameters |
| P1.7 Model registry | 🟡 Partial | FM and LightGBM run | Registry status mismatch; DeepFM not implemented |
| P1.8 Legal parameters | ✅ Mostly complete | Config allowlists and validator | Seed/BPR/smoothing spaces need expansion |
| P1.9 Leakage audit | 🟡 Partial | LOO, test isolation, evaluator hash | Temporal audit and multi-seed report |
| P1.10 Sanity experiments | ✅ Complete | Seven controlled experiments | Package as final submission evidence |

### 2. P1.1 — Benchmark Note

#### Task

Each row represents one video impression for one user. The model emits a real-valued score used to rank already logged candidates within that user. This is not full-catalog retrieval.

- Target: binary `long_view`
- GAUC: whether positive items rank above negative items within each user, weighted by positive count
- nDCG@5: whether positive items appear near the top five positions
- Primary: `(GAUC + nDCG@5) / 2`

#### Data and fixed split

```text
data/KuaiRand-Pure/data/
├── video_features_basic_pure.csv
├── log_standard_4_08_to_4_21_pure.csv
└── log_standard_4_22_to_5_08_pure.csv
```

| Split | Dates | Current rows | Purpose |
|---|---|---:|---|
| Train | Apr 8–21 | 1,141,112 | Training and feature fitting |
| Validation | Apr 22–28 | 124,909 | Early stopping and KEEP/REJECT |
| Test | Apr 29–May 8 | 170,588 | Evaluate validation-best once after research |

#### Baseline pipeline

```text
data.load()
→ train-only vocabularies and duration buckets
→ user_id, video_id, author_id, tab, dur_bucket
→ NumPy FM + Adam + BCE
→ validation scores
→ official evaluate.py
→ GAUC / nDCG@5 / Primary
```

Official evaluator SHA-256:

```text
ecfde28392eb14fec4f488083251df50624e1af2b86278b962daecfb42d195de
```

#### Reproduction

```bash
python scripts/verify_setup.py
cd kuairand-starter-kit
python baseline.py --model fm --data_dir ../data/KuaiRand-Pure/data
```

Current seed-0 result:

| GAUC | nDCG@5 | Primary | Best epoch | Training time |
|---:|---:|---:|---:|---:|
| 0.667133 | 0.535806 | 0.601470 | 7 | About 15–20 seconds, excluding loading |

This is close to the official `0.6674 / 0.5357 / 0.6016`. Small differences can come from platform, NumPy version, floating-point order, and display precision.

**Acceptance:** benchmark, target, splits, metrics, and baseline are understood and reproducible. A complete clean-virtual-environment recording/log is still needed.

### 3. P1.2 — Stable Training Interface

The current real interface is in `src/techjam_agent/runner.py`:

```python
ExperimentRunner.run(config, checkpoint) -> validation_metrics
ExperimentRunner.finalize(config, checkpoint, submission_path) -> test_metrics
```

The config includes model, training objective, hyperparameters, features, and LightGBM-specific parameters. Training returns validation metrics only; `finalize()` is the only path allowed to access test. `isolated.py` executes every experiment in a child process with a 900-second timeout.

The second architecture expects `recommender/train.py::train_model()`, but it still raises `NotImplementedError`. The next engineering task is to connect FM, BPR FM, and LightGBM through adapters and keep one canonical API.

**Acceptance:** the same config and seed can be invoked repeatedly without exposing FM internals. The current pipeline satisfies this; the unified `recommender/train.py` does not yet.

### 4. P1.3 — Feature Registry

Every final feature definition should include:

```text
name, required_columns, supported_models, fit_split,
fit/transform logic, unseen fallback, parameters,
cache_key, leakage_rule, implemented
```

| Feature | Type | Status | Safety rule / fallback |
|---|---|---|---|
| `user_id` | Categorical | Available | Validation unseen → UNK |
| `video_id` | Categorical | Available | Validation unseen → UNK |
| `author_id` | Categorical | Available | Missing metadata → UNK |
| `tab` | Categorical | Available | Unseen → UNK |
| `dur_bucket` | Categorical | Available | Bucket edges fit on train only |
| `item_popularity` | Target-free history | Encoder implemented | Unseen count=0 |
| `user_activity` | Target-free history | Encoder implemented | Unseen count=0 |
| `item_long_view_rate` | Target-derived | Executed | Train LOO; unseen → global prior |
| `user_long_view_rate` | Target-derived | Executed | Train LOO; unseen → global prior |
| `user_tag_affinity` | Target-derived | Encoder implemented | Pair → global prior; real run pending |
| `user_tab_long_view_rate` | Target-derived | Executed | Train LOO; unseen → global prior |
| `continuous_history_stats` | Continuous bundle | Executed | Train LOO; LightGBM only |
| `temporal_recency` | Temporal | Not implemented | Must use strict past windows |

Expensive statistics are currently recomputed in each child process. Recommended cache key:

```text
dataset_digest + train_range + feature_name + feature_version
+ smoothing + window_days + buckets + leave_one_out
```

No cache may contain validation or test labels.

**Acceptance:** more than 4–6 configurable features exist. The two registries/encoders, caching, and temporal features still need unification.

### 5. P1.4 — Item-side Features

Item popularity is `log1p(train item exposure count)`. It is label-free, unseen items fall back to zero, and it can be bucketed for FM or kept continuous for trees/neural models.

Smoothed item long-view rate is:

```text
(item_positive + α × global_rate) / (item_count + α)
```

Current `α=20`; train uses leave-one-out, validation uses full train statistics, and unseen items fall back to the global rate.

Known results: FM bucketed item rate scored `0.591682`; the LightGBM continuous global bundle scored `0.590084`. Both were rejected. More promising follow-ups are recent 1/3/7-day exposure/rates and author/tag priors.

**Acceptance:** count/rate, smoothing, unseen fallback, and leakage protection exist. Feature distribution, missingness, and unseen-rate reports remain.

### 6. P1.5 — Personalization Features

Implemented features include `user_activity`, `user_long_view_rate`, `user_tab_long_view_rate`, and the `user_tag_affinity` encoder.

Recommended sparse-pair fallback:

```text
user-category smoothed estimate when sufficiently supported
→ category prior
→ user prior
→ global prior
```

Expose `min_pair_count`, `smoothing_strength`, and `fallback_strategy`.

LightGBM + user×tab scored `0.597528`, below pure LightGBM at `0.599817`. However, `tab` is impression context, while video `tag` is closer to content category; user×tag remains worth a coverage audit and isolated experiment.

**Acceptance:** at least one personalization feature runs safely and has an isolated ablation. The real user×tag experiment remains.

### 7. P1.6 — Objective / Loss

Pointwise BCE:

```text
-y log σ(s) - (1-y) log(1-σ(s))
```

Pairwise BPR:

```text
-log σ(score_positive - score_negative)
```

Current BPR pairs examples within the same user, samples one negative per positive with replacement, resamples each epoch, and skips all-positive/all-negative users. Model, features, split, and evaluator remain identical to BCE.

Result: Primary improved from `0.601470` to `0.603396`, so the agent selected KEEP.

Next parameters: `pairs_per_positive` 1/2/4, BPR-specific learning rate, max pairs per user, uniform-user weighting, hard negatives, margin, and temperature.

**Acceptance:** objective switch, mathematical definition, controlled comparison, and reproducible execution are complete. Multi-seed validation remains.

### 8. P1.7 — Model Registry

| Model | Actually runnable | Objective | Best Primary | Note |
|---|---|---|---:|---|
| FM | ✅ | BCE/BPR | **0.603396** | Current best |
| LightGBM | ✅ | Binary BCE | 0.599817 | Stable but below FM |
| DeepFM | ❌ | Planned BCE/BPR | — | Registry placeholder only |

`recommender/models/__init__.py` still marks all models as `implemented=False`, which does not match the real `src/techjam_agent` capability. It should be corrected when the unified Trainer is connected.

Recommended adapter contract:

```python
fit(train_x, train_y, valid_x, valid_y, config)
predict(x) -> one finite score per row
save_checkpoint(path)
load_checkpoint(path)
```

**Acceptance:** FM and a second model, LightGBM, run end to end. Registry unification and DeepFM remain.

### 9. P1.8 — Legal Parameters and Compatibility

FM allowlist:

```text
embedding_dim: 8,16,24,32,48,64
learning_rate: 0.0003,0.0005,0.001,0.002,0.005
epochs: 10,20,30,40
l2: 0,1e-6,1e-5,1e-4
batch_size: 4096,8192,16384
patience: 3,4,5
seed: currently only 0
```

LightGBM defaults:

```text
learning_rate=0.05, num_leaves=31, n_estimators=500,
min_child_samples=100, subsample=0.9, colsample_bytree=0.9,
reg_lambda=1e-4, early_stopping_rounds=30
```

Current rules: BPR only supports FM; LightGBM only supports BCE; continuous statistics and user×tab continuous features only support LightGBM; unknown, missing, or out-of-range parameters are rejected before training; evaluator and baseline metadata are protected.

Still needed: seeds 0–4, BPR pair count, smoothing/bucket/min-count, temporal windows, DeepFM compatibility, and multi-task loss constraints.

**Acceptance:** nonsense configurations are mechanically rejected. Feature-specific parameters still need full schemas.

### 10. P1.9 — Leakage and Reproducibility Audit

| Risk | Current protection |
|---|---|
| Validation target leakage | Aggregates fit on train only |
| Train self-label leakage | Target encoding uses LOO |
| Test feedback leakage | Iterations contain validation only; test runs once at the end |
| Modified evaluator | SHA-256 integrity check |
| Randomness | Seed stored in config and logs |
| Hanging experiment | Child-process isolation and 900-second timeout |
| Lost failure evidence | JSON, JSONL, and tree snapshot |

Remaining audits: repeated same-config/seed report, BCE/BPR seeds 0–4, temporal “future does not affect past” tests, feature distribution audit, and clean-environment reproduction on macOS and Windows.

**Acceptance:** validation/test feedback leakage is blocked. Multi-seed and cross-platform reports remain.

### 11. P1.10 — Manual Sanity Experiments

| # | Parent | Main isolated change | Primary | Delta vs FM BCE | Decision |
|---:|---|---|---:|---:|---|
| 0 | — | FM+BCE baseline | 0.601470 | — | KEEP |
| 1 | FM BCE | BCE → BPR | **0.603396** | **+0.001927** | KEEP |
| 2 | FM BCE | + user rate bucket | 0.600448 | -0.001022 | REJECT |
| 3 | FM BCE | + item rate bucket | 0.591682 | -0.009788 | REJECT |
| 4 | FM BCE | FM → LightGBM | 0.599817 | -0.001653 | REJECT |
| 5 | LightGBM | + continuous global stats | 0.590084 | -0.011386 | REJECT |
| 6 | LightGBM | + user×tab rate/count | 0.597528 | -0.003941 | REJECT |

All operators finish training, produce finite aligned predictions, use the official evaluator, and log their config, metrics, runtime, and decision without test feedback during research.

**Acceptance:** baseline, item feature, personalization, objective, and model operators have all been executed.

### 12. Next Work Order

#### P0 — BPR multi-seed

1. Expand seed allowlist to 0–4.
2. Run BCE and BPR for every seed.
3. Report paired-delta mean, standard deviation, and confidence interval.
4. Decide whether BPR is a stable improvement.

#### P1 — Unified training interface

1. Define a canonical `ModelAdapter`.
2. Connect FM/BPR FM/LightGBM to `recommender/train.py`.
3. Make both entry points share the same backend.
4. Correct model-registry status.

#### P1 — Registry and cache

1. Merge registry metadata with the real encoder.
2. Put smoothing/window/bucket parameters into schemas.
3. Add dataset/version-aware caching.
4. Produce distribution/unseen/NaN audits.

#### P2 — New research capability

1. `pairs_per_positive` 1/2/4.
2. Strictly historical recent exposure/activity.
3. LightGBM LambdaRank.
4. Minimal DeepFM BCE.
5. DeepFM+BPR.
6. Click/like multi-task learning.

### 13. Person 1 Deliverables

```text
docs/person1-ml-benchmark-lead.md
docs/models-features-optimization-guide.md
docs/bpr-experiment.md
configs/experiment.json
configs/project.json
recommender/features.py
recommender/train.py                 # currently pending connection
tests/test_agent.py
tests/test_architecture.py
```

Person 1 is complete not when the repository contains the largest possible number of models, but when:

> The Planner, Executor, and demo team can supply a legal configuration, run a stable leakage-safe experiment, and know exactly what changed, whether the result is trustworthy, and what can safely be researched next.

---

## 中文版 ================================================================================

> 角色目标：搭建可信、可复现、无数据泄漏的机器学习研究环境。Person 1 负责定义 Agent 能安全研究什么，不负责替 Agent 选择每一轮实验。

详细模型、参数、特征与优化路线见 [`models-features-optimization-guide.md`](models-features-optimization-guide.md)，BPR 单项实验见 [`bpr-experiment.md`](bpr-experiment.md)。

### 1. 总体进度

| 任务 | 状态 | 当前证据 | 主要缺口 |
|---|---|---|---|
| P1.1 Benchmark | ✅ 基本完成 | 官方 FM 已复现，指标/split 已文档化 | clean-environment 复现记录 |
| P1.2 统一训练接口 | 🟡 部分完成 | `src/techjam_agent/runner.py` 可运行 FM/LightGBM | `recommender/train.py` 尚未接通，双框架待统一 |
| P1.3 Feature registry | 🟡 部分完成 | 两套 encoder 支持多种安全 feature | 缺 cache key、统一 fit/transform、temporal feature |
| P1.4 Item 历史特征 | ✅ 完成最小验收 | popularity、item rate/count | 未提分；recency/author prior 未做 |
| P1.5 Personalization | ✅ 完成最小验收 | user activity/rate、user×tab/tag | user×tag 尚需真实实验 |
| P1.6 Objective | ✅ 完成 | BCE/BPR 可配置且有公平对照 | BPR 多 seed、负采样参数 |
| P1.7 Model registry | 🟡 部分完成 | FM 与 LightGBM 可运行 | registry 状态不一致；DeepFM 未实现 |
| P1.8 参数兼容规则 | ✅ 基本完成 | config 白名单与 validator | seed/BPR/smoothing 参数待扩展 |
| P1.9 Leakage audit | 🟡 部分完成 | LOO、test 隔离、evaluator hash | temporal audit、多 seed 报告 |
| P1.10 Sanity experiments | ✅ 完成 | 7 个受控实验 | 整理成最终提交证据 |

### 2. P1.1 — Benchmark Note

### 2.1 任务

每行表示一次用户看到视频的曝光。模型输出实数分数，用于在同一用户的已曝光候选中排序；这不是全库召回。

- Target：`long_view`，二元 `0/1`
- GAUC：同用户正样本是否排在负样本前，按正例数加权
- nDCG@5：正样本是否出现在用户前五名靠前位置
- Primary：`(GAUC + nDCG@5) / 2`

### 2.2 数据与切分

```text
data/KuaiRand-Pure/data/
├── video_features_basic_pure.csv
├── log_standard_4_08_to_4_21_pure.csv
└── log_standard_4_22_to_5_08_pure.csv
```

| Split | 日期 | 当前行数 | 用途 |
|---|---|---:|---|
| Train | 04-08 至 04-21 | 1,141,112 | 训练与 feature fitting |
| Validation | 04-22 至 04-28 | 124,909 | early stopping、KEEP/REJECT |
| Test | 04-29 至 05-08 | 170,588 | 研究结束后只评 validation-best |

### 2.3 Baseline pipeline

```text
data.load()
→ train-only vocabulary/duration buckets
→ user_id, video_id, author_id, tab, dur_bucket
→ NumPy FM + Adam + BCE
→ validation scores
→ official evaluate.py
→ GAUC / nDCG@5 / Primary
```

官方 evaluator SHA-256：

```text
ecfde28392eb14fec4f488083251df50624e1af2b86278b962daecfb42d195de
```

### 2.4 复现

```bash
python scripts/verify_setup.py
cd kuairand-starter-kit
python baseline.py --model fm --data_dir ../data/KuaiRand-Pure/data
```

当前 seed 0：

| GAUC | nDCG@5 | Primary | Best epoch | 训练时间 |
|---:|---:|---:|---:|---:|
| 0.667133 | 0.535806 | 0.601470 | 7 | 约 15–20 秒，不含加载 |

与官方约 `0.6674 / 0.5357 / 0.6016` 接近。微小差异可能来自平台、NumPy 版本、浮点运算顺序和显示精度。

**验收：** benchmark、target、split、指标与 baseline 均可解释和复现；仍需补一次全新虚拟环境录屏/日志。

### 3. P1.2 — 稳定训练接口

当前真实接口位于 `src/techjam_agent/runner.py`：

```python
ExperimentRunner.run(config, checkpoint) -> validation_metrics
ExperimentRunner.finalize(config, checkpoint, submission_path) -> test_metrics
```

Config 包含：

```text
model
training_objective
hyperparameters
features
lightgbm_hyperparameters
```

训练阶段只返回 validation；`finalize()` 才可访问 test。`isolated.py` 将每个实验放入独立子进程并施加 900 秒 timeout。

另一套架构期望 `recommender/train.py::train_model()`，但当前仍为 `NotImplementedError`。下一项工程工作是用 adapter 将 FM/BPR FM/LightGBM 接到统一 Trainer contract，最终只保留一个 canonical API。

**验收：** 同 config+seed 可重复调用，外部不需要理解 FM 内部；当前 pipeline 已满足，统一 `recommender/train.py` 尚未满足。

### 4. P1.3 — Feature Registry

每个 feature 最终应登记：

```text
name, required_columns, supported_models, fit_split,
fit/transform logic, unseen fallback, parameters,
cache_key, leakage_rule, implemented
```

当前 feature：

| Feature | 类型 | 状态 | 安全规则 / fallback |
|---|---|---|---|
| `user_id` | categorical | 可用 | validation unseen → UNK |
| `video_id` | categorical | 可用 | validation unseen → UNK |
| `author_id` | categorical | 可用 | metadata missing → UNK |
| `tab` | categorical | 可用 | unseen → UNK |
| `dur_bucket` | categorical | 可用 | 桶边界只 fit train |
| `item_popularity` | target-free history | encoder 已实现 | unseen count=0 |
| `user_activity` | target-free history | encoder 已实现 | unseen count=0 |
| `item_long_view_rate` | target-derived | 已运行 | train LOO；unseen → global prior |
| `user_long_view_rate` | target-derived | 已运行 | train LOO；unseen → global prior |
| `user_tag_affinity` | target-derived | encoder 已实现 | pair → global prior；待真实实验 |
| `user_tab_long_view_rate` | target-derived | 已运行 | train LOO；unseen → global prior |
| `continuous_history_stats` | continuous bundle | 已运行 | train LOO；仅 LightGBM |
| `temporal_recency` | temporal | 未实现 | 必须严格过去窗口 |

昂贵统计当前会在子进程重复计算。建议 cache key：

```text
dataset_digest + train_range + feature_name + feature_version
+ smoothing + window_days + buckets + leave_one_out
```

缓存不得包含 validation/test labels。

**验收：** 已有 4–6 个以上 feature 可配置；尚需合并两套 registry/encoder、增加 cache 与 temporal feature。

### 5. P1.4 — Item-side Features

### Item popularity

```text
log1p(train item exposure count)
```

- 不使用 label
- unseen item 回退 0
- 可分桶进入 FM，或连续进入 LightGBM/神经网络

### Smoothed item long-view rate

```text
(item_positive + α × global_rate) / (item_count + α)
```

- 当前 `α=20`
- train 使用 LOO
- validation 只使用完整 train
- unseen item 回退 global rate

已知结果：FM 分桶 item rate `0.591682`；LightGBM global continuous bundle `0.590084`，均失败。

下一步更值得研究 recent 1/3/7-day exposure/rate、author/tag prior，而不是继续复制全局 rate。

**验收：** count/rate、平滑、unseen fallback、防泄漏已实现；待补 feature 分布、缺失率和 unseen-rate 报告。

### 6. P1.5 — Personalization Features

已实现：

```text
user_activity
user_long_view_rate
user_tab_long_view_rate
user_tag_affinity encoder
```

稀疏 pair 推荐 fallback：

```text
user-category smoothed estimate（样本足够）
→ category prior
→ user prior
→ global prior
```

应暴露 `min_pair_count`、`smoothing_strength`、`fallback_strategy`。

LightGBM + user×tab 的 Primary 为 `0.597528`，低于纯 LightGBM `0.599817`。但 `tab` 是曝光上下文，video `tag` 更接近内容类别；user×tag 仍值得先做覆盖率检查再独立实验。

**验收：** personalization 已能配置和独立 ablation；user×tag 真实实验未完成。

### 7. P1.6 — Objective / Loss

### Pointwise BCE

```text
-y log σ(s) - (1-y) log(1-σ(s))
```

### Pairwise BPR

```text
-log σ(score_positive - score_negative)
```

当前 BPR：

- 同一用户内配对
- 每个正样本随机匹配 1 个负样本
- 有放回采样，每 epoch 重采样
- 全正/全负用户跳过
- 与 BCE 使用相同模型、features、split 和 evaluator

结果：Primary `0.601470 → 0.603396`，Agent KEEP。

下一步参数：`pairs_per_positive` 1/2/4、BPR 专用 lr、max pairs/user、uniform-user weighting、hard negative、margin/temperature。

**验收：** objective config switch、数学定义、公平对照与可复现运行已完成；多 seed 未完成。

### 8. P1.7 — Model Registry

| Model | 真实可运行 | Objective | 当前最好 Primary | 说明 |
|---|---|---|---:|---|
| FM | ✅ | BCE/BPR | **0.603396** | 当前最佳 |
| LightGBM | ✅ | binary BCE | 0.599817 | 稳定但不如 FM |
| DeepFM | ❌ | 规划 BCE/BPR | — | registry 占位，未实现 |

注意：`recommender/models/__init__.py` 目前仍把所有 model 标为 `implemented=False`，与 `src/techjam_agent` 的真实能力不一致，需要修正并接入统一 Trainer。

推荐统一 adapter：

```python
fit(train_x, train_y, valid_x, valid_y, config)
predict(x) -> one finite score per row
save_checkpoint(path)
load_checkpoint(path)
```

**验收：** FM 与第二模型 LightGBM 可用；registry 和通用接口未完全统一；DeepFM 需等 pipeline 稳定再做。

### 9. P1.8 — Legal Parameters / Compatibility

### FM 白名单

```text
embedding_dim: 8,16,24,32,48,64
learning_rate: 0.0003,0.0005,0.001,0.002,0.005
epochs: 10,20,30,40
l2: 0,1e-6,1e-5,1e-4
batch_size: 4096,8192,16384
patience: 3,4,5
seed: 当前仅 0
```

### LightGBM 默认值

```text
learning_rate=0.05, num_leaves=31, n_estimators=500,
min_child_samples=100, subsample=0.9, colsample_bytree=0.9,
reg_lambda=1e-4, early_stopping_rounds=30
```

### 当前兼容规则

- BPR 只允许 FM
- LightGBM 当前只允许 BCE
- continuous stats 与 user×tab 连续特征只允许 LightGBM
- 未知/缺失/越界参数在训练前拒绝
- evaluator 和 baseline metadata 受保护

待增加：seed 0–4、BPR pair count、smoothing/bucket/min-count、temporal window、DeepFM 与 multi-task 规则。

**验收：** nonsense config 可提前拒绝；feature-specific 参数仍需全部 schema 化。

### 10. P1.9 — Leakage / Reproducibility Audit

| 风险 | 当前措施 |
|---|---|
| validation target leakage | aggregate 只 fit train |
| train self-label leakage | target encoding 使用 LOO |
| test feedback leakage | iteration 只含 validation；test 最终只跑一次 |
| evaluator 被修改 | SHA-256 校验 |
| 随机性 | seed 写入 config/log |
| 实验卡死 | 子进程隔离 + 900 秒 timeout |
| 失败证据丢失 | JSON、JSONL、tree snapshot |

仍需完成：同 config+seed 重复性报告、BCE/BPR seed 0–4、temporal“未来不影响过去”测试、feature distribution audit、macOS/Windows clean-environment 复现。

**验收：** 当前无 validation/test feedback leakage；多 seed 与跨平台报告待完成。

### 11. P1.10 — Manual Sanity Experiments

| # | Parent | 主要唯一变化 | Primary | Delta vs FM BCE | 决策 |
|---:|---|---|---:|---:|---|
| 0 | — | FM+BCE baseline | 0.601470 | — | KEEP |
| 1 | FM BCE | BCE → BPR | **0.603396** | **+0.001927** | KEEP |
| 2 | FM BCE | + user rate bucket | 0.600448 | -0.001022 | REJECT |
| 3 | FM BCE | + item rate bucket | 0.591682 | -0.009788 | REJECT |
| 4 | FM BCE | FM → LightGBM | 0.599817 | -0.001653 | REJECT |
| 5 | LightGBM | + continuous global stats | 0.590084 | -0.011386 | REJECT |
| 6 | LightGBM | + user×tab rate/count | 0.597528 | -0.003941 | REJECT |

这些 operator 均已完成训练、有限值/对齐检查、官方评估和日志记录，且研究阶段不使用 test feedback。

**验收：** baseline、item feature、personalization、objective、model 共 5 类 operator 已实际执行。

### 12. 下一步工作顺序

### P0：BPR multi-seed

1. seed 白名单扩展到 0–4
2. 每个 seed 分别运行 BCE/BPR
3. 输出 paired delta mean/std/confidence interval
4. 决定 BPR 是否为稳定提升

### P1：统一训练接口

1. 定义 canonical `ModelAdapter`
2. FM/BPR FM/LightGBM 接入 `recommender/train.py`
3. 两个入口共享同一 backend
4. 修正 model registry 状态

### P1：Feature registry/cache

1. 合并 registry metadata 与真实 encoder
2. smoothing/window/bucket 参数全部进入 schema
3. 加入 dataset/version-aware cache
4. 输出 distribution/unseen/NaN audit

### P2：新增研究能力

1. `pairs_per_positive` 1/2/4
2. 严格时间滚动的 recent exposure/activity
3. LightGBM LambdaRank
4. 最小 DeepFM BCE
5. DeepFM+BPR
6. click/like multi-task

### 13. Person 1 交付文件

```text
docs/person1-ml-benchmark-lead.md
docs/models-features-optimization-guide.md
docs/bpr-experiment.md
configs/experiment.json
configs/project.json
recommender/features.py
recommender/train.py                 # 当前待接通
tests/test_agent.py
tests/test_architecture.py
```

Person 1 的完成标准不是“实现尽可能多的模型”，而是：

> Planner、Executor 和演示人员都能通过合法配置，稳定运行无泄漏实验，并准确知道实验改变了什么、结果是否可信、下一步还能安全研究什么。
