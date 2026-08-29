# Models, Parameters, Features, and Optimization

## English

### Models

| Model | Objective | Status | Validation Primary |
|---|---|---|---:|
| Random baseline | Random score | Organizer reference only | Not used for optimization |
| Item popularity | Smoothed train item rate | Organizer reference only | Below FM |
| FM | BCE | Ready | 0.601470 |
| FM | BPR / hybrid | Ready | BPR 0.603963 (seed 0) |
| LightGBM | Binary BCE | Ready; rejected | 0.599817 |
| DeepFM | BCE/BPR | Implemented in NumPy | BCE best 0.603862; BPR 0.603530 |
| Multi-task DeepFM | long_view + click/like auxiliary BCE | PROMISING; 2/3 rolling folds improved | 0.604259 |
| FM+DeepFM ensemble | BPR FM + BCE DeepFM | KEEP; 3/3 rolling folds improved | **0.604713** |
| Ensemble + temporal | Same ensemble + strict-past counts | REJECT; 1/3 rolling folds improved | 0.605010 single split |

### FM Parameters

| Parameter | Default | Legal values |
|---|---:|---|
| `embedding_dim` | 16 | 8, 16, 24, 32, 48, 64 |
| `learning_rate` | 0.001 | 0.0003, 0.0005, 0.001, 0.002, 0.005 |
| `epochs` | 40 | 10, 20, 30, 40 |
| `l2` | 1e-6 | 0, 1e-6, 1e-5, 1e-4 |
| `batch_size` | 8192 | 4096, 8192, 16384 |
| `patience` | 4 | 3, 4, 5 |
| `seed` | 0 | 0, 1, 2, 3, 4 |
| `deepfm_hidden_dim` | 32 | 16, 32, 64 |
| `auxiliary_loss_weight` | 0.1 | 0.05, 0.1, 0.2, 0.5 |

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

- Current `scripts/run_agent.py` switches include historical rates, explicit user crosses, and strict-past 3-day user/item counts.
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
| `user_recent_3d_activity` | Strict-past 3-day count | FM/DeepFM/ensemble | Implemented; rolling REJECT |
| `item_recent_3d_exposure` | Strict-past 3-day count | FM/DeepFM/ensemble | Implemented; rolling REJECT |

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

- BPR and hybrid: FM and DeepFM.
- Continuous history and user×tab continuous features: LightGBM only.
- DeepFM: BCE and BPR available.
- Multi-task DeepFM: BCE only; click/like are training-only auxiliary labels, never inference features.
- Unknown/out-of-range parameters: reject before training.
- Historical target features: train LOO; validation/test use train only.
- Test metrics: never returned to the research loop.

### Verified Results

| Experiment | Primary | Decision |
|---|---:|---|
| FM+BCE | 0.601470 | Baseline |
| FM+BCE, seed=1 | 0.601761 | Paired seed check |
| FM+BCE, seed=2 | 0.601090 | Paired seed check |
| FM+BCE, seed=3 | 0.601503 | Paired seed check |
| FM+BPR, lr=0.001 | 0.603396 | KEEP |
| FM+BPR, lr=0.0005 | 0.603696 | KEEP |
| FM+BPR, lr=0.0003 | **0.603963** | KEEP; current best |
| FM+BPR, lr=0.0003, seed=1 | 0.603352 | Multi-seed check |
| FM+BPR, lr=0.0003, seed=2 | 0.603757 | Multi-seed check |
| FM+BPR, lr=0.0003, seed=3 | **0.604128** | Highest single run |
| FM+BPR, 2 negatives/positive | 0.603379 | REJECT; no meaningful change |
| FM+BPR, 4 negatives/positive | 0.602794 | REJECT |
| FM+BPR, semi-hard pool=2 | 0.601855 | REJECT |
| FM+BPR, semi-hard pool=4 | 0.587747 | REJECT |
| FM+BPR + user×tab cross | 0.602869 | REJECT |
| FM+BPR + user×author cross | 0.602180 | REJECT |
| FM+BPR + both crosses | 0.601198 | REJECT |
| DeepFM+BCE, lr=0.0005 | 0.603457 | REJECT |
| DeepFM+BCE, lr=0.001 | 0.603862 | REJECT; close to best |
| DeepFM+BCE, lr=0.002 | 0.603637 | REJECT |
| DeepFM+BPR, lr=0.0003 | 0.603530 | REJECT |
| Multi-task DeepFM, click+like weight=0.1 | 0.604259 | PROMISING; rolling check required |
| FM hybrid, BPR weight=0.75 | 0.603962 | REJECT; tied with pure BPR |
| FM hybrid, BPR weight=0.50 | 0.603912 | REJECT |
| FM hybrid, BPR weight=0.25 | 0.603507 | REJECT |
| Ensemble, DeepFM weight=0.3 | 0.604562 | REJECT |
| Ensemble, DeepFM weight=0.4 | **0.604713** | KEEP; current best |
| Ensemble, DeepFM weight=0.5 | 0.604203 | REJECT |
| Ensemble + user recent 3d activity | 0.604931 | KEEP on validation |
| Ensemble + user/item recent 3d exposure | **0.605010** | Validation best; test regressed |
| FM + user rate | 0.600448 | REJECT |
| FM + item rate | 0.591682 | REJECT |
| LightGBM | 0.599817 | REJECT |
| LightGBM + global stats | 0.590084 | REJECT |
| LightGBM + user×tab | 0.597528 | REJECT |

### Rolling Validation

| Fold | FM+BPR | Ensemble | Ensemble + temporal | Ensemble vs FM | Temporal vs ensemble |
|---|---:|---:|---:|---:|---:|
| 04/15–04/17 | 0.610742 | 0.611392 | 0.611736 | +0.000650 | +0.000345 |
| 04/18–04/20 | 0.580129 | 0.581429 | 0.580824 | +0.001301 | -0.000606 |
| 04/21–04/23 | 0.586890 | 0.588308 | 0.587830 | +0.001418 | -0.000477 |
| Mean | 0.592587 | **0.593710** | 0.593463 | **+0.001123** | -0.000246 |

Decision: keep the 0.4-weight FM+DeepFM ensemble because it improved 3/3 folds. Reject the temporal addition because it improved only 1/3 folds and regressed on the official test.

Multi-task DeepFM versus DeepFM on the same rolling folds: `+0.000204`, `+0.000440`, `-0.000050`; mean `+0.000198`, wins 2/3. Keep the implementation as promising but do not promote it to the best model or sweep its weight.

### Optimization Order

1. Feed the rolling evidence and failed-direction memory to the LLM Researcher.
2. Implement one stronger interaction model: DCNv2.
3. Test DCNv2 complementarity before adding it to the ensemble.
4. Revisit multi-task only if DCNv2 provides a stronger shared representation.
5. Validate the final ensemble across seeds without choosing the best seed.

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
| FM | BPR / hybrid | 可用 | BPR 0.603963（seed 0） |
| LightGBM | Binary BCE | 可用；已拒绝 | 0.599817 |
| DeepFM | BCE/BPR | NumPy 已实现 | BCE 最佳 0.603862；BPR 0.603530 |
| Multi-task DeepFM | long_view + click/like 辅助 BCE | PROMISING；rolling 2/3 提升 | 0.604259 |
| FM+DeepFM ensemble | BPR FM + BCE DeepFM | KEEP；rolling 3/3 提升 | **0.604713** |
| Ensemble + temporal | 同一 ensemble + 严格过去计数 | REJECT；rolling 仅 1/3 提升 | 单切分 0.605010 |

### FM 参数

| 参数 | 默认值 | 合法值 |
|---|---:|---|
| `embedding_dim` | 16 | 8, 16, 24, 32, 48, 64 |
| `learning_rate` | 0.001 | 0.0003, 0.0005, 0.001, 0.002, 0.005 |
| `epochs` | 40 | 10, 20, 30, 40 |
| `l2` | 1e-6 | 0, 1e-6, 1e-5, 1e-4 |
| `batch_size` | 8192 | 4096, 8192, 16384 |
| `patience` | 4 | 3, 4, 5 |
| `seed` | 0 | 0、1、2、3、4 |
| `deepfm_hidden_dim` | 32 | 16、32、64 |
| `auxiliary_loss_weight` | 0.1 | 0.05、0.1、0.2、0.5 |

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

- 当前 `scripts/run_agent.py` 开关包括历史比例、显式 user 交叉，以及严格过去 3 天的 user/item 计数。
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
| `user_recent_3d_activity` | 严格过去 3 天计数 | FM/DeepFM/ensemble | 已实现；rolling REJECT |
| `item_recent_3d_exposure` | 严格过去 3 天计数 | FM/DeepFM/ensemble | 已实现；rolling REJECT |

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

- BPR 和 hybrid 支持 FM、DeepFM。
- 连续历史统计和 user×tab 连续特征只支持 LightGBM。
- DeepFM 已支持 BCE 和 BPR。
- Multi-task DeepFM 仅支持 BCE；click/like 只作为训练辅助标签，不是推理特征。
- 未知/越界参数训练前拒绝。
- Target 历史特征：train LOO；validation/test 只用 train。
- Test 指标不反馈给研究循环。

### 已验证结果

| 实验 | Primary | 决策 |
|---|---:|---|
| FM+BCE | 0.601470 | Baseline |
| FM+BCE，seed=1 | 0.601761 | 配对 seed 验证 |
| FM+BCE，seed=2 | 0.601090 | 配对 seed 验证 |
| FM+BCE，seed=3 | 0.601503 | 配对 seed 验证 |
| FM+BPR，lr=0.001 | 0.603396 | KEEP |
| FM+BPR，lr=0.0005 | 0.603696 | KEEP |
| FM+BPR，lr=0.0003 | **0.603963** | KEEP；当前最佳 |
| FM+BPR，lr=0.0003，seed=1 | 0.603352 | 多 seed 验证 |
| FM+BPR，lr=0.0003，seed=2 | 0.603757 | 多 seed 验证 |
| FM+BPR，lr=0.0003，seed=3 | **0.604128** | 单次最高 |
| FM+BPR，每个正样本 2 个负样本 | 0.603379 | REJECT；无有效变化 |
| FM+BPR，每个正样本 4 个负样本 | 0.602794 | REJECT |
| FM+BPR，semi-hard 候选池=2 | 0.601855 | REJECT |
| FM+BPR，semi-hard 候选池=4 | 0.587747 | REJECT |
| FM+BPR + user×tab 交叉 | 0.602869 | REJECT |
| FM+BPR + user×author 交叉 | 0.602180 | REJECT |
| FM+BPR + 两种交叉 | 0.601198 | REJECT |
| DeepFM+BCE，lr=0.0005 | 0.603457 | REJECT |
| DeepFM+BCE，lr=0.001 | 0.603862 | REJECT；接近 best |
| DeepFM+BCE，lr=0.002 | 0.603637 | REJECT |
| DeepFM+BPR，lr=0.0003 | 0.603530 | REJECT |
| Multi-task DeepFM，click+like 权重=0.1 | 0.604259 | PROMISING；需 rolling 验证 |
| FM hybrid，BPR 权重=0.75 | 0.603962 | REJECT；与纯 BPR 持平 |
| FM hybrid，BPR 权重=0.50 | 0.603912 | REJECT |
| FM hybrid，BPR 权重=0.25 | 0.603507 | REJECT |
| Ensemble，DeepFM 权重=0.3 | 0.604562 | REJECT |
| Ensemble，DeepFM 权重=0.4 | **0.604713** | KEEP；当前最佳 |
| Ensemble，DeepFM 权重=0.5 | 0.604203 | REJECT |
| Ensemble + user 最近 3 天活跃度 | 0.604931 | Validation KEEP |
| Ensemble + user/item 最近 3 天曝光 | **0.605010** | Validation 最高；Test 回落 |
| FM + user rate | 0.600448 | REJECT |
| FM + item rate | 0.591682 | REJECT |
| LightGBM | 0.599817 | REJECT |
| LightGBM + global stats | 0.590084 | REJECT |
| LightGBM + user×tab | 0.597528 | REJECT |

### Rolling Validation

| Fold | FM+BPR | Ensemble | Ensemble + temporal | Ensemble 相对 FM | Temporal 相对 ensemble |
|---|---:|---:|---:|---:|---:|
| 04/15–04/17 | 0.610742 | 0.611392 | 0.611736 | +0.000650 | +0.000345 |
| 04/18–04/20 | 0.580129 | 0.581429 | 0.580824 | +0.001301 | -0.000606 |
| 04/21–04/23 | 0.586890 | 0.588308 | 0.587830 | +0.001418 | -0.000477 |
| 平均 | 0.592587 | **0.593710** | 0.593463 | **+0.001123** | -0.000246 |

决策：保留 DeepFM 权重 0.4 的 FM+DeepFM ensemble，因为它在 3/3 folds 都提升。拒绝 temporal 增量，因为它只在 1/3 folds 提升，且官方 test 也回落。

Multi-task DeepFM 相对普通 DeepFM 的 rolling 增量为：`+0.000204`、`+0.000440`、`-0.000050`；平均 `+0.000198`，2/3 folds 提升。保留实现并标记为 promising，但不升级为当前最佳，也不继续扫权重。

### 优化顺序

1. 把 rolling 证据和失败方向记忆输入 LLM Researcher。
2. 只实现一个更强交互模型：DCNv2。
3. 先验证 DCNv2 的预测互补性，再决定是否加入 ensemble。
4. 只有 DCNv2 提供更强共享表示时，再回头研究 multi-task。
5. 不挑最高 seed，验证最终 ensemble 的多 seed 稳定性。

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
