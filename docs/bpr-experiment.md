# BPR Pairwise FM 实验说明 / Experiment Notes

## 中文版

### 这一步做了什么

本次改动为现有 FM 模型增加了 BPR（Bayesian Personalized Ranking）训练方式。

原始 FM 使用 BCE loss，把每条曝光独立看成二分类问题：

```text
这个用户会不会 long_view 这个视频？
```

BPR 则在同一个用户内部生成正负样本对，并直接学习：

```text
long_view 的视频分数 > 没有 long_view 的视频分数
```

比赛使用 GAUC 和 nDCG@5 评价用户内排序，因此 BPR 与最终评价目标更加一致。

### 如何保证实验公平

BCE 和 BPR 实验保持以下内容完全相同：

- 相同的 FM 模型结构
- 相同的 5 个输入字段
- 相同的数据划分
- 相同的 embedding dimension、learning rate 和 early stopping
- 相同的官方 `evaluate.py`

唯一变化是：

```text
training_objective: bce → bpr
```

BPR 的每个正负样本对都来自同一个用户。每个有效正样本在当前 epoch 随机匹配一个该用户的负样本，全正或全负用户不参与 pairwise 训练。

### 实验结果

Seed 0 的首次结果：

| 实验 | GAUC | nDCG@5 | Primary |
|---|---:|---:|---:|
| FM + BCE | 0.667133 | 0.535806 | 0.601470 |
| FM + BPR | **0.669711** | **0.537082** | **0.603396** |
| 提升 | +0.002578 | +0.001276 | **+0.001927** |

Test split 的 Primary 同样从 `0.595330` 上升到 `0.597506`。

Agent 因此自动将 BPR 实验标记为 `KEEP`，并保存为当前 best model。

完整运行日志位于：

```text
logs/run_20260829T074736Z/
```

日志目录默认被 Git 忽略，不会随 commit 上传。

### 主要代码位置

- `src/techjam_agent/bpr.py`：同用户正负样本配对和 BPR 梯度更新
- `src/techjam_agent/runner.py`：根据配置选择 BCE 或 BPR 训练
- `src/techjam_agent/config.py`：验证 `training_objective`
- `src/techjam_agent/proposals.py`：让 Research Agent 提出 BPR 实验
- `configs/experiment.json`：默认实验配置
- `tests/test_agent.py`：配对约束和 Agent 控制流程测试

### 如何复现

安装依赖并准备好 KuaiRand-Pure 数据后运行：

```bash
python scripts/run_agent.py --researcher deterministic --max-iterations 1
```

程序将依次运行：

```text
Iteration 0: FM + BCE baseline
Iteration 1: FM + BPR
```

随后自动比较 Primary，执行 `KEEP/REJECT`，并写出实验日志、最佳模型和 submission。

运行测试：

```bash
python -m unittest discover -s tests -v
```

### 当前结论与下一步

BPR 是目前第一个同时提升 GAUC 和 nDCG@5 的方向，但 Primary 增量 `0.001927` 略低于项目设定的显著阈值 `0.002`。

因此目前可以确认它是一个有希望的方向，但还不能仅凭单个 seed 宣称稳定提升。下一步应使用多个随机种子重复 BCE/BPR 对照；如果提升稳定，再研究负样本数量、负样本选择策略和 BPR learning rate。

---

## English Version

### What changed

This change adds BPR (Bayesian Personalized Ranking) training to the existing Factorization Machine.

The original FM uses binary cross-entropy and treats every impression as an independent classification problem:

```text
Will this user long-view this video?
```

BPR creates positive-negative pairs for the same user and directly learns:

```text
score(long-view video) > score(non-long-view video)
```

Because the benchmark evaluates within-user ranking using GAUC and nDCG@5, BPR is more closely aligned with the final evaluation objective.

### Experimental control

The BCE and BPR experiments use exactly the same:

- FM architecture
- Five input fields
- Dataset splits
- Embedding dimension, learning rate, and early stopping settings
- Official `evaluate.py`

The only experimental change is:

```text
training_objective: bce → bpr
```

Every BPR pair belongs to one user. During each epoch, every eligible positive impression is matched with a randomly sampled negative impression from that user. Users with only positive or only negative impressions cannot form ranking pairs and are excluded from pairwise training.

### Results

The first seed-0 experiment produced:

| Experiment | GAUC | nDCG@5 | Primary |
|---|---:|---:|---:|
| FM + BCE | 0.667133 | 0.535806 | 0.601470 |
| FM + BPR | **0.669711** | **0.537082** | **0.603396** |
| Improvement | +0.002578 | +0.001276 | **+0.001927** |

Primary on the test split also increased from `0.595330` to `0.597506`.

The agent therefore marked the BPR experiment as `KEEP` and saved it as the current best model.

The complete run log is located at:

```text
logs/run_20260829T074736Z/
```

The log directory is ignored by Git and is not included in commits by default.

### Main files

- `src/techjam_agent/bpr.py`: same-user pair sampling and BPR gradient updates
- `src/techjam_agent/runner.py`: selects BCE or BPR training from the configuration
- `src/techjam_agent/config.py`: validates `training_objective`
- `src/techjam_agent/proposals.py`: lets the Research Agent propose the BPR experiment
- `configs/experiment.json`: default experiment configuration
- `tests/test_agent.py`: pair constraints and controller tests

### Reproduction

After installing the dependencies and preparing KuaiRand-Pure, run:

```bash
python scripts/run_agent.py --researcher deterministic --max-iterations 1
```

The program runs:

```text
Iteration 0: FM + BCE baseline
Iteration 1: FM + BPR
```

It then compares Primary, makes the automatic `KEEP/REJECT` decision, and writes the experiment logs, best model, and submission.

Run the tests with:

```bash
python -m unittest discover -s tests -v
```

### Conclusion and next step

BPR is the first direction tested so far that improves both GAUC and nDCG@5. However, its Primary improvement of `0.001927` is slightly below the project significance threshold of `0.002`.

It is therefore a promising result, but one seed is not enough to claim a stable improvement. The next step is to repeat the BCE/BPR comparison across multiple random seeds. If the gain remains consistent, later experiments can investigate the number of negatives per positive, negative-sampling strategies, and the BPR learning rate.
