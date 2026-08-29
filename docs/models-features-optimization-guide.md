# Models, Parameters, Features, and Optimization

## English

### Models

| Model | Objective | Status | Validation Primary |
|---|---|---|---:|
| Random baseline | Random score | Organizer reference only | Not used for optimization |
| Item popularity | Smoothed train item rate | Organizer reference only | Below FM |
| FM | BCE | Ready | 0.601470 |
| FM | BPR | **Current best** | **0.603396** |
| LightGBM | Binary BCE | Ready; rejected | 0.599817 |
| DeepFM | Planned BCE/BPR | Not implemented | — |

### FM Parameters

| Parameter | Default | Legal values |
|---|---:|---|
| `embedding_dim` | 16 | 8, 16, 24, 32, 48, 64 |
| `learning_rate` | 0.001 | 0.0003, 0.0005, 0.001, 0.002, 0.005 |
| `epochs` | 40 | 10, 20, 30, 40 |
| `l2` | 1e-6 | 0, 1e-6, 1e-5, 1e-4 |
| `batch_size` | 8192 | 4096, 8192, 16384 |
| `patience` | 4 | 3, 4, 5 |
| `seed` | 0 | Currently only 0; next 0–4 |

### BPR Parameters

Current: configurable 1/2/4 same-user negatives per positive, replacement sampling, resampled each epoch.

Current and planned parameters:

```text
pairs_per_positive: 1, 2, 4
negative_sampling: random, hard
max_pairs_per_user
user_pair_weighting: uniform-user, uniform-pair
bpr_learning_rate
```

The main architecture already accepts these FM/BPR feature parameters:

| Parameter | Default | Validator range | Current integration |
|---|---:|---:|---|
| `pairs_per_positive` | 1 | 1, 2, 4 | Current runnable Agent and main FM backend |
| `feature_smoothing` | 20 | 1–1000 | Main FeatureEncoder only |
| `feature_buckets` | 20 | 2–100 | Main FeatureEncoder only |

The main validator also accepts continuous ranges `k: 4–128`, `lr: 1e-5–0.1`, `epochs: 1–100`, and `l2: 0–1`. The current runnable `src/techjam_agent` uses the narrower discrete FM allowlist above.

### LightGBM Parameters

| Parameter | Default |
|---|---:|
| `learning_rate` | 0.05 |
| `num_leaves` | 31 |
| `n_estimators` | 500 |
| `min_child_samples` | 100 |
| `subsample` | 0.9 |
| `colsample_bytree` | 0.9 |
| `reg_lambda` | 1e-4 |
| `early_stopping_rounds` | 30 |

### Features

Two feature sets currently coexist:

- Current `scripts/run_agent.py` switches: `user_long_view_rate`, `item_long_view_rate`, `continuous_history_stats`, `user_tab_long_view_rate`.
- Main `recommender/feature_encoding.py`: the five base fields plus `item_popularity`, `user_activity`, `item_long_view_rate`, and `user_tag_affinity`.

| Feature | Type | Models | Status/result |
|---|---|---|---|
| `user_id` | Categorical | FM/LGBM | Base |
| `video_id` | Categorical | FM/LGBM | Base |
| `author_id` | Categorical | FM/LGBM | Base |
| `tab` | Categorical | FM/LGBM | Base |
| `dur_bucket` | Train-fitted bucket | FM/LGBM | Base |
| `item_popularity` | Log count | FM | Implemented encoder |
| `user_activity` | Log count | FM | Implemented encoder |
| `user_long_view_rate` | Smoothed LOO bucket | FM | 0.600448; rejected |
| `item_long_view_rate` | Smoothed LOO bucket | FM | 0.591682; rejected |
| `continuous_history_stats` | Rates + log counts | LGBM | 0.590084; rejected |
| `user_tab_long_view_rate` | Pair rate + count | LGBM | 0.597528; rejected |
| `user_tag_affinity` | Smoothed LOO bucket | FM | Encoder ready; run pending |
| `temporal_recency` | Rolling history | Future | Not implemented |

### Optimizable Functions

#### Data and feature functions

| Function | Current responsibility | Optimization options |
|---|---|---|
| `data.load()` | Load and date-split rows | Streaming/chunked CSV, cached parsed data, lower memory |
| `data.encode()` | Five-field categorical encoding | Reusable vocabularies, cached matrices, numeric feature values |
| `FeatureEncoder.fit_transform()` | Train-fit safe feature encoding | Separate `fit/transform`, persistent cache, sparse output |
| `fit_historical_statistics()` | Build item/user/pair aggregates | Vectorization, daily aggregates, incremental updates |
| `_smoothed_rate()` | Smoothed target rate | Tune smoothing, min count, hierarchical priors |
| `_quantile_edges()` | Numeric-to-category buckets | Tune bucket count, monotonic/fixed bins, robust quantiles |
| `aggregate()` / `aggregate_pair()` | Current pipeline aggregates | Cache, vectorize, temporal cutoffs, fallback hierarchy |
| `smoothed_rate_bucket()` | LOO rate bucket | Tune prior/buckets; compare continuous representation |

#### Model and objective functions

| Function | Current responsibility | Optimization options |
|---|---|---|
| `FM.__init__()` | Initialize W/V and Adam state | Initialization scale, embedding dim, optimizer configuration |
| `FM.logits()` | FM first/second-order score | Numeric-valued FM, field-aware interactions, interaction masking |
| `FM.step()` | BCE gradient update | Class/user weighting, focal loss, optimizer, gradient clipping |
| `FM.predict()` | Batched scoring | Batch size, memory mapping, parallel scoring |
| `build_pair_indices()` | Same-user BPR pairs | Multiple negatives, user-balanced pairs, hard negatives |
| `bpr_step()` | Pairwise BPR gradient | BPR lr/L2, margin, temperature, weighting, clipping |
| `_run_lightgbm()` | Binary LightGBM training | LambdaRank, group-by-user, custom ranking metric, parameter search |
| `_lightgbm_matrices()` | Categorical + continuous matrix | Feature selection, categorical encoding, cached matrices |
| `run_validation_fm()` | Validation-only FM adapter | Multi-seed, pairs/positive, feature smoothing/buckets |

#### Evaluation and agent functions

| Function | Current responsibility | Optimization options |
|---|---|---|
| `evaluate()` | Official GAUC/nDCG@5 | Do not change; optimize only runtime with equivalence tests |
| `Controller._execute()` | Run/log/KEEP/REJECT | Retry policy, resource metrics, statistical decision rules |
| `Controller._converged()` | Stop after weak improvements | Correct post-best rounds, confidence-aware stopping |
| `DeterministicResearcher.propose()` | Fixed experiment order | Evidence-driven ordering, branch coverage, failed-action memory |
| `OpenAICompatibleResearcher.propose()` | LLM structured proposal | Better schema, prompt evidence, token limits, provider abstraction |
| `GroundedCritic.review()` / `review()` | Interpret measured results | Multi-seed confidence, cost-aware critique, follow-up generation |
| `TreeSearchPolicy.select()` | Choose experiment parent | Exploration/novelty/runtime weights, branch pruning |
| `ExperimentValidator.validate_config()` | Reject illegal configs | Model-specific schemas, conditional parameters, resource limits |
| `SubprocessExecutor.run()` / `IsolatedExperimentRunner.run()` | Isolate experiment | Streaming logs, memory limits, retry/recovery, checkpoint resume |

Official functions that must not be behaviorally modified: `evaluate()`, `auc()`, and `ndcg_at_k()`. Any performance rewrite must pass exact-equivalence tests.

### Compatibility Rules

- BPR: FM only.
- Continuous history and user×tab continuous features: LightGBM only.
- DeepFM: unavailable.
- Unknown/out-of-range parameters: reject before training.
- Historical target features: train LOO; validation/test use train only.
- Test metrics: never returned to the research loop.

### Verified Results

| Experiment | Primary | Decision |
|---|---:|---|
| FM+BCE | 0.601470 | Baseline |
| FM+BPR, lr=0.001 | 0.603396 | KEEP |
| FM+BPR, lr=0.0005 | 0.603696 | KEEP |
| FM+BPR, lr=0.0003 | **0.603963** | KEEP; current best |
| FM+BPR, 2 negatives/positive | 0.603379 | REJECT; no meaningful change |
| FM+BPR, 4 negatives/positive | 0.602794 | REJECT |
| FM + user rate | 0.600448 | REJECT |
| FM + item rate | 0.591682 | REJECT |
| LightGBM | 0.599817 | REJECT |
| LightGBM + global stats | 0.590084 | REJECT |
| LightGBM + user×tab | 0.597528 | REJECT |

### Optimization Order

1. BCE/BPR seeds 0–4 using BPR lr=0.0003.
2. BPR hard-negative sampling and user weighting.
3. Strict-past 1/3/7-day item/user features.
4. LightGBM LambdaRank grouped by user.
5. Minimal DeepFM+BCE.
6. DeepFM+BPR.
7. Click/like multi-task learning.
8. Watch-time auxiliary loss.
9. Ensemble only after complementary models exist.

### Code Extension Points

#### Add a model

1. `src/techjam_agent/config.py`: model and legal parameters.
2. `configs/experiment.json`: defaults.
3. New model module: fit/predict/save/load.
4. `src/techjam_agent/runner.py`: run/finalize dispatch.
5. `src/techjam_agent/proposals.py`: legal Agent actions.
6. `requirements.txt`: pinned dependency.
7. Tests: config, dispatch, checkpoint, test isolation.

#### Add a feature

1. Registry metadata and required columns.
2. Train-only `fit()` and deterministic `transform()`.
3. Unseen/missing fallback.
4. Leakage rule and parameters.
5. Cache key.
6. Model compatibility.
7. Leakage and alignment tests.

#### Unify frameworks

1. Implement `recommender/train.py`.
2. Connect FM/BPR/LightGBM adapters.
3. Correct `recommender/models` status.
4. Use one runner/backend for both entry points.

---

## 中文

### 模型

| 模型 | Objective | 状态 | Validation Primary |
|---|---|---|---:|
| Random baseline | 随机分数 | 官方参考，不参与优化 | 不使用 |
| Item popularity | 平滑 train item rate | 官方参考 | 低于 FM |
| FM | BCE | 可用 | 0.601470 |
| FM | BPR | **当前最佳** | **0.603396** |
| LightGBM | Binary BCE | 可用；已拒绝 | 0.599817 |
| DeepFM | 计划 BCE/BPR | 未实现 | — |

### FM 参数

| 参数 | 默认值 | 合法值 |
|---|---:|---|
| `embedding_dim` | 16 | 8, 16, 24, 32, 48, 64 |
| `learning_rate` | 0.001 | 0.0003, 0.0005, 0.001, 0.002, 0.005 |
| `epochs` | 40 | 10, 20, 30, 40 |
| `l2` | 1e-6 | 0, 1e-6, 1e-5, 1e-4 |
| `batch_size` | 8192 | 4096, 8192, 16384 |
| `patience` | 4 | 3, 4, 5 |
| `seed` | 0 | 当前仅 0；下一步 0–4 |

### BPR 参数

当前：每个正样本可配 1/2/4 个同用户负样本，有放回采样，每个 epoch 重采样。

当前及计划参数：

```text
pairs_per_positive: 1, 2, 4
negative_sampling: random, hard
max_pairs_per_user
user_pair_weighting: uniform-user, uniform-pair
bpr_learning_rate
```

Main 架构已经支持的额外参数：

| 参数 | 默认值 | Validator 范围 | 当前接入位置 |
|---|---:|---:|---|
| `pairs_per_positive` | 1 | 1、2、4 | 当前可运行 Agent 和 main FM backend |
| `feature_smoothing` | 20 | 1–1000 | 仅 main FeatureEncoder |
| `feature_buckets` | 20 | 2–100 | 仅 main FeatureEncoder |

Main validator 还允许连续范围：`k: 4–128`、`lr: 1e-5–0.1`、`epochs: 1–100`、`l2: 0–1`。当前实际运行的 `src/techjam_agent` 使用上方更窄的离散白名单。

### LightGBM 参数

| 参数 | 默认值 |
|---|---:|
| `learning_rate` | 0.05 |
| `num_leaves` | 31 |
| `n_estimators` | 500 |
| `min_child_samples` | 100 |
| `subsample` | 0.9 |
| `colsample_bytree` | 0.9 |
| `reg_lambda` | 1e-4 |
| `early_stopping_rounds` | 30 |

### 特征

仓库当前存在两组 feature 配置：

- 当前 `scripts/run_agent.py` 开关：`user_long_view_rate`、`item_long_view_rate`、`continuous_history_stats`、`user_tab_long_view_rate`。
- Main `recommender/feature_encoding.py`：5 个基础字段，加 `item_popularity`、`user_activity`、`item_long_view_rate`、`user_tag_affinity`。

| Feature | 类型 | 模型 | 状态/结果 |
|---|---|---|---|
| `user_id` | Categorical | FM/LGBM | 基础 |
| `video_id` | Categorical | FM/LGBM | 基础 |
| `author_id` | Categorical | FM/LGBM | 基础 |
| `tab` | Categorical | FM/LGBM | 基础 |
| `dur_bucket` | Train 拟合桶 | FM/LGBM | 基础 |
| `item_popularity` | Log count | FM | Encoder 已实现 |
| `user_activity` | Log count | FM | Encoder 已实现 |
| `user_long_view_rate` | 平滑 LOO 分桶 | FM | 0.600448；拒绝 |
| `item_long_view_rate` | 平滑 LOO 分桶 | FM | 0.591682；拒绝 |
| `continuous_history_stats` | Rate + log count | LGBM | 0.590084；拒绝 |
| `user_tab_long_view_rate` | Pair rate + count | LGBM | 0.597528；拒绝 |
| `user_tag_affinity` | 平滑 LOO 分桶 | FM | Encoder 可用；待运行 |
| `temporal_recency` | 滚动历史 | 未来 | 未实现 |

### 可优化的 Function

#### 数据与特征函数

| Function | 当前职责 | 可优化方向 |
|---|---|---|
| `data.load()` | 加载并按日期切分 | CSV 流式/分块读取、解析缓存、降低内存 |
| `data.encode()` | 五字段类别编码 | 复用 vocabulary、缓存矩阵、支持数值 feature value |
| `FeatureEncoder.fit_transform()` | 安全拟合和编码 | 拆分 `fit/transform`、持久 cache、稀疏输出 |
| `fit_historical_statistics()` | user/item/pair 聚合 | 向量化、按天预聚合、增量更新 |
| `_smoothed_rate()` | 平滑 target rate | smoothing、min count、分层 prior |
| `_quantile_edges()` | 连续值分桶 | bucket 数、固定/单调桶、robust quantile |
| `aggregate()` / `aggregate_pair()` | 当前 pipeline 聚合 | Cache、向量化、时间 cutoff、fallback hierarchy |
| `smoothed_rate_bucket()` | LOO rate 分桶 | prior/bucket 调参；对比连续表示 |

#### 模型与 Objective 函数

| Function | 当前职责 | 可优化方向 |
|---|---|---|
| `FM.__init__()` | 初始化 W/V 和 Adam | 初始化尺度、embedding dim、optimizer 配置 |
| `FM.logits()` | FM 一阶/二阶打分 | 数值 FM、field-aware interaction、interaction mask |
| `FM.step()` | BCE 梯度更新 | 类别/用户权重、focal loss、optimizer、gradient clipping |
| `FM.predict()` | 分批预测 | Batch size、memory map、并行打分 |
| `build_pair_indices()` | 同用户 BPR 配对 | 多负样本、user-balanced、hard negative |
| `bpr_step()` | BPR 梯度 | BPR lr/L2、margin、temperature、weight、clipping |
| `_run_lightgbm()` | Binary LightGBM | LambdaRank、user group、自定义 ranking metric、参数搜索 |
| `_lightgbm_matrices()` | 类别+连续矩阵 | Feature selection、类别编码、矩阵 cache |
| `run_validation_fm()` | Validation-only FM adapter | Multi-seed、pairs/positive、smoothing/buckets |

#### 评估与 Agent 函数

| Function | 当前职责 | 可优化方向 |
|---|---|---|
| `evaluate()` | 官方 GAUC/nDCG@5 | 不改行为；仅允许等价测试后的性能优化 |
| `Controller._execute()` | 执行/日志/KEEP/REJECT | Retry、资源指标、统计决策规则 |
| `Controller._converged()` | 连续弱提升后停止 | 修正 best 后轮数、confidence-aware stopping |
| `DeterministicResearcher.propose()` | 固定实验顺序 | 证据驱动排序、分支覆盖、失败经验 |
| `OpenAICompatibleResearcher.propose()` | LLM 结构化提案 | 更严格 schema、prompt evidence、token limit、provider adapter |
| `GroundedCritic.review()` / `review()` | 解释实验 | 多 seed 置信度、cost-aware critique、follow-up |
| `TreeSearchPolicy.select()` | 选择父节点 | exploration/novelty/runtime 权重、branch pruning |
| `ExperimentValidator.validate_config()` | 拒绝非法配置 | 模型专属 schema、条件参数、资源上限 |
| `SubprocessExecutor.run()` / `IsolatedExperimentRunner.run()` | 隔离实验 | 实时日志、内存限制、retry/recovery、checkpoint resume |

禁止改变官方 `evaluate()`、`auc()`、`ndcg_at_k()` 的行为；任何性能重写都必须通过精确等价测试。

### 兼容规则

- BPR 只支持 FM。
- 连续历史统计和 user×tab 连续特征只支持 LightGBM。
- DeepFM 当前不可用。
- 未知/越界参数训练前拒绝。
- Target 历史特征：train LOO；validation/test 只用 train。
- Test 指标不反馈给研究循环。

### 已验证结果

| 实验 | Primary | 决策 |
|---|---:|---|
| FM+BCE | 0.601470 | Baseline |
| FM+BPR，lr=0.001 | 0.603396 | KEEP |
| FM+BPR，lr=0.0005 | 0.603696 | KEEP |
| FM+BPR，lr=0.0003 | **0.603963** | KEEP；当前最佳 |
| FM+BPR，每个正样本 2 个负样本 | 0.603379 | REJECT；无有效变化 |
| FM+BPR，每个正样本 4 个负样本 | 0.602794 | REJECT |
| FM + user rate | 0.600448 | REJECT |
| FM + item rate | 0.591682 | REJECT |
| LightGBM | 0.599817 | REJECT |
| LightGBM + global stats | 0.590084 | REJECT |
| LightGBM + user×tab | 0.597528 | REJECT |

### 优化顺序

1. 使用 BPR lr=0.0003 做 BCE/BPR seed 0–4 配对验证。
2. BPR hard negative 和 user weighting。
3. 严格过去窗口的 1/3/7 天 item/user feature。
4. 按用户分组的 LightGBM LambdaRank。
5. 最小 DeepFM+BCE。
6. DeepFM+BPR。
7. Click/like multi-task。
8. Watch-time auxiliary loss。
9. 只有模型互补时再 ensemble。

### 代码扩展位置

#### 新增模型

1. `src/techjam_agent/config.py`：模型和合法参数。
2. `configs/experiment.json`：默认值。
3. 新模型模块：fit/predict/save/load。
4. `src/techjam_agent/runner.py`：run/finalize dispatch。
5. `src/techjam_agent/proposals.py`：Agent 合法动作。
6. `requirements.txt`：固定依赖。
7. Tests：config、dispatch、checkpoint、test isolation。

#### 新增特征

1. Registry metadata 和原始列。
2. Train-only `fit()` 和确定性 `transform()`。
3. Unseen/missing fallback。
4. Leakage rule 和参数。
5. Cache key。
6. 模型兼容规则。
7. Leakage 和行对齐测试。

#### 统一框架

1. 实现 `recommender/train.py`。
2. 接入 FM/BPR/LightGBM adapter。
3. 修正 `recommender/models` 状态。
4. 两个入口共用一个 runner/backend。
