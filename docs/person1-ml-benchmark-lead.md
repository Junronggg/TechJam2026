# Person 1 — ML / Benchmark Lead

## English

### Scope

- Own benchmark correctness, models, features, objectives, legal parameters, leakage safety, and reproducibility.
- Provide safe experiment operators to the Planner/Executor.
- Do not manually choose every experiment in the autonomous run.

### P1.1 Benchmark

| Item | Value |
|---|---|
| Task | Within-user ranking over logged impressions |
| Target | `long_view` |
| Train | 2022-04-08 to 2022-04-21; 1,141,112 rows |
| Validation | 2022-04-22 to 2022-04-28; 124,909 rows |
| Test | 2022-04-29 to 2022-05-08; 170,588 rows |
| Metrics | GAUC, nDCG@5 |
| Primary | `(GAUC + nDCG@5) / 2` |
| Baseline inputs | `user_id`, `video_id`, `author_id`, `tab`, `dur_bucket` |
| Baseline seed | 0 |
| Baseline result | GAUC 0.667133; nDCG@5 0.535806; Primary 0.601470 |
| Evaluator | Protected `kuairand-starter-kit/evaluate.py` |

Status: complete. Remaining: clean-environment reproduction log.

### P1.2 Stable Training Interface

- Current: `ExperimentRunner.run(config, checkpoint) -> validation_metrics`
- Final only: `ExperimentRunner.finalize(...) -> test_metrics + submission`
- Isolated worker timeout: 900 seconds
- Gap: connect FM/BPR/LightGBM to canonical `recommender/train.py`; remove duplicate training interfaces.

### P1.3 Feature Registry

Required fields per feature:

```text
name, required_columns, supported_models, fit_split,
fit/transform, fallback, parameters, cache_key,
leakage_rule, implemented
```

Implemented: base IDs/context, item popularity, user activity, user/item rates, user×tab, user×tag encoder, continuous history bundle.

Gaps: unified registry, cache, temporal features.

### P1.4 Item Features

- `item_popularity = log1p(train exposure count)`
- Smoothed `item_long_view_rate`
- Train target encoding uses leave-one-out.
- Validation unseen item falls back to global prior.
- Gap: recent 1/3/7-day item features and distribution audit.

### P1.5 Personalization

- `user_activity`
- `user_long_view_rate`
- `user_tab_long_view_rate`
- `user_tag_affinity` encoder
- Gap: real user×tag ablation; hierarchical fallback and `min_pair_count`.

### P1.6 Objectives

| Objective | Definition | Status |
|---|---|---|
| BCE | Pointwise long-view classification | Ready |
| BPR | Same-user positive score > negative score | Ready; current best |

Ready: `pairs_per_positive = 1/2/4`. Gap: seeds 0–4, hard negatives, user weighting.

### P1.7 Models

| Model | Status | Validation Primary |
|---|---|---:|
| FM+BCE | Ready | 0.601470 |
| FM+BPR | Ready; best | **0.603396** |
| LightGBM | Ready; rejected | 0.599817 |
| DeepFM | Not implemented | — |

Gap: correct registry status and unified adapters.

### P1.8 Legal Configuration

- Parameter allowlists exist.
- Invalid model/objective/feature combinations are rejected before training.
- BPR supports FM only.
- Continuous history features support LightGBM only.
- Protected organizer files cannot be modified.
- Gap: seed 0–4 and feature/BPR-specific schemas.

### P1.9 Leakage and Reproducibility

- Aggregates fit on train only.
- Train target encoding uses leave-one-out.
- Research history contains validation metrics only.
- Test runs once for validation-best.
- Evaluator SHA-256 is verified.
- Config, seed, runtime, checkpoint, metrics, decisions, and failures are logged.
- Gap: multi-seed and cross-platform report; temporal leakage tests.

### P1.10 Sanity Experiments

| Experiment | Primary | Decision |
|---|---:|---|
| FM+BCE | 0.601470 | Baseline |
| FM+BPR, lr=0.001 | 0.603396 | KEEP |
| FM+BPR, lr=0.0005 | 0.603696 | KEEP |
| FM+BPR, lr=0.0003 | **0.603963** | KEEP; current best |
| FM+BPR, 2 negatives/positive | 0.603379 | REJECT |
| FM+BPR, 4 negatives/positive | 0.602794 | REJECT |
| FM + user rate | 0.600448 | REJECT |
| FM + item rate | 0.591682 | REJECT |
| LightGBM | 0.599817 | REJECT |
| LightGBM + global stats | 0.590084 | REJECT |
| LightGBM + user×tab | 0.597528 | REJECT |

### Next Tasks

1. BCE/BPR paired runs for seeds 0–4 using BPR lr=0.0003.
2. Test hard-negative sampling and user weighting.
3. Unify training interface and registries.
4. Add leakage-safe recent features.
5. Test LightGBM LambdaRank.
6. Implement DeepFM after pipeline unification.

---

## 中文

### 职责范围

- 负责 benchmark、模型、特征、loss、合法参数、防泄漏和可复现性。
- 向 Planner/Executor 提供安全实验操作。
- 不在自动运行中人工选择每轮实验。

### P1.1 Benchmark

| 项目 | 内容 |
|---|---|
| 任务 | 用户内已曝光视频排序 |
| Target | `long_view` |
| Train | 2022-04-08 至 04-21；1,141,112 行 |
| Validation | 2022-04-22 至 04-28；124,909 行 |
| Test | 2022-04-29 至 05-08；170,588 行 |
| 指标 | GAUC、nDCG@5 |
| Primary | `(GAUC + nDCG@5) / 2` |
| Baseline 输入 | `user_id`、`video_id`、`author_id`、`tab`、`dur_bucket` |
| Seed | 0 |
| Baseline 结果 | GAUC 0.667133；nDCG@5 0.535806；Primary 0.601470 |
| Evaluator | 受保护的 `kuairand-starter-kit/evaluate.py` |

状态：完成。缺口：clean-environment 复现日志。

### P1.2 稳定训练接口

- 当前：`ExperimentRunner.run(config, checkpoint) -> validation_metrics`
- 最终：`ExperimentRunner.finalize(...) -> test_metrics + submission`
- 子进程 timeout：900 秒
- 缺口：接通 `recommender/train.py`，删除重复训练接口。

### P1.3 Feature Registry

每个 feature 必须包含：

```text
名称、原始列、支持模型、fit split、fit/transform、
fallback、参数、cache key、leakage rule、实现状态
```

已实现：基础 ID/上下文、item popularity、user activity、user/item rate、user×tab、user×tag encoder、连续历史统计。

缺口：统一 registry、cache、时间特征。

### P1.4 Item 特征

- `item_popularity = log1p(train exposure count)`
- 平滑 `item_long_view_rate`
- Train target encoding 使用 LOO。
- Validation unseen item 回退 global prior。
- 缺口：最近 1/3/7 天 item 特征和分布审计。

### P1.5 Personalization

- `user_activity`
- `user_long_view_rate`
- `user_tab_long_view_rate`
- `user_tag_affinity` encoder
- 缺口：真实 user×tag 消融、分层 fallback、`min_pair_count`。

### P1.6 Objective

| Objective | 定义 | 状态 |
|---|---|---|
| BCE | Pointwise long-view 分类 | 可用 |
| BPR | 同用户正样本分数 > 负样本 | 可用；当前最佳 |

已支持：`pairs_per_positive = 1/2/4`。缺口：seed 0–4、hard negative、用户权重。

### P1.7 模型

| 模型 | 状态 | Validation Primary |
|---|---|---:|
| FM+BCE | 可用 | 0.601470 |
| FM+BPR | 可用；最佳 | **0.603396** |
| LightGBM | 可用；已拒绝 | 0.599817 |
| DeepFM | 未实现 | — |

缺口：修正 registry 状态并统一 adapter。

### P1.8 合法配置

- 已有参数白名单。
- 非法 model/objective/feature 组合会在训练前拒绝。
- BPR 当前只支持 FM。
- 连续历史特征当前只支持 LightGBM。
- 禁止修改官方受保护文件。
- 缺口：seed 0–4 和 feature/BPR 专属 schema。

### P1.9 Leakage 与复现

- 聚合只 fit train。
- Train target encoding 使用 LOO。
- Agent history 只包含 validation。
- Test 只对 validation-best 最终运行一次。
- Evaluator 使用 SHA-256 校验。
- Config、seed、runtime、checkpoint、metrics、decision、failure 均记录。
- 缺口：多 seed、跨平台报告、时间泄漏测试。

### P1.10 Sanity Experiments

| 实验 | Primary | 决策 |
|---|---:|---|
| FM+BCE | 0.601470 | Baseline |
| FM+BPR，lr=0.001 | 0.603396 | KEEP |
| FM+BPR，lr=0.0005 | 0.603696 | KEEP |
| FM+BPR，lr=0.0003 | **0.603963** | KEEP；当前最佳 |
| FM+BPR，每个正样本 2 个负样本 | 0.603379 | REJECT |
| FM+BPR，每个正样本 4 个负样本 | 0.602794 | REJECT |
| FM + user rate | 0.600448 | REJECT |
| FM + item rate | 0.591682 | REJECT |
| LightGBM | 0.599817 | REJECT |
| LightGBM + global stats | 0.590084 | REJECT |
| LightGBM + user×tab | 0.597528 | REJECT |

### 下一步

1. 使用 BPR lr=0.0003 做 seed 0–4 的 BCE/BPR 配对实验。
2. 测试 hard negative 和用户权重。
3. 统一训练接口和 registry。
4. 增加无泄漏的 recent feature。
5. 测试 LightGBM LambdaRank。
6. Pipeline 统一后实现 DeepFM。
