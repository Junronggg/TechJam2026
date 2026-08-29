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

Current: one same-user negative per positive, replacement sampling, resampled each epoch.

Next legal parameters:

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
| `pairs_per_positive` | 1 | 1–5 | Main FM backend only |
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
| FM+BPR | **0.603396** | KEEP |
| FM + user rate | 0.600448 | REJECT |
| FM + item rate | 0.591682 | REJECT |
| LightGBM | 0.599817 | REJECT |
| LightGBM + global stats | 0.590084 | REJECT |
| LightGBM + user×tab | 0.597528 | REJECT |

### Optimization Order

1. BCE/BPR seeds 0–4.
2. BPR negatives per positive: 1/2/4.
3. BPR hard-negative sampling and user weighting.
4. Strict-past 1/3/7-day item/user features.
5. LightGBM LambdaRank grouped by user.
6. Minimal DeepFM+BCE.
7. DeepFM+BPR.
8. Click/like multi-task learning.
9. Watch-time auxiliary loss.
10. Ensemble only after complementary models exist.

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

当前：每个正样本配一个同用户负样本，有放回采样，每个 epoch 重采样。

下一步合法参数：

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
| `pairs_per_positive` | 1 | 1–5 | 仅 main FM backend |
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
| FM+BPR | **0.603396** | KEEP |
| FM + user rate | 0.600448 | REJECT |
| FM + item rate | 0.591682 | REJECT |
| LightGBM | 0.599817 | REJECT |
| LightGBM + global stats | 0.590084 | REJECT |
| LightGBM + user×tab | 0.597528 | REJECT |

### 优化顺序

1. BCE/BPR seed 0–4。
2. BPR 每个正样本配 1/2/4 个负样本。
3. BPR hard negative 和 user weighting。
4. 严格过去窗口的 1/3/7 天 item/user feature。
5. 按用户分组的 LightGBM LambdaRank。
6. 最小 DeepFM+BCE。
7. DeepFM+BPR。
8. Click/like multi-task。
9. Watch-time auxiliary loss。
10. 只有模型互补时再 ensemble。

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
