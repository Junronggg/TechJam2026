# Project Resume / 项目交接摘要

## English

### Objective

Build an autonomous ML research agent for the KuaiRand-Pure ranking task. The agent
must improve a recommender through evidence-driven iterations while minimizing human
intervention and preventing validation/test leakage.

### Current implementation

- Controller loop: observe → propose → isolated train/evaluate → critique → memory →
  keep/reject/reinterpret/stop.
- Models and objectives: FM with BCE/BPR, DeepFM, DCNv2, field-aware FM, grouped
  ranking, sequential and graph candidates, multi-task and ensemble operators.
- Safety: train-only statistics, strict past-only sequence context, official evaluator
  digest check, duplicate protection, subprocess timeouts, convergence/budget limits,
  one-shot finalization guard, and manual-intervention logging.
- Evidence: structured experiment history, research memory, parent/child lineage,
  rolling/seed/placebo records, prediction diversity and submission-candidate status.

### Current peer result

The robust validation fallback is FM+BPR plus DeepFM (Primary `0.6047128`, rolling
3/3). The selected final local test result is GAUC `0.666354`, nDCG@5 `0.532377`,
and Primary `0.599365` over `170,588` rows. It is a local test-split result, not a
hidden-test score.

### Files to read first

`README.md` → `SUBMISSION_SUMMARY.md` → `TRY.md` → `AGENT-TRY.md` → `RUN_LOG.md` →
`docs/architecture.md`.

## 中文

### 项目目标

为 KuaiRand-Pure 推荐排序任务构建一个能自主迭代的 ML Research Agent。它根据
验证集证据选择下一项实验，尽量减少人工干预，并严格避免验证集/测试集泄漏。

### 当前状态

- 已实现完整闭环：观察、提出假设、隔离训练、验证、反思、更新记忆和停止。
- 已覆盖 FM/BCE、FM/BPR、DeepFM、DCNv2、排序、序列、图、多任务和 ensemble
  等候选能力。
- 已加入超时、预算、重复实验保护、官方 evaluator 校验、strict-time 特征和
  final 一次性评估保护。
- peer 的稳健验证结果是 FM+BPR + DeepFM，Primary `0.6047128`，rolling `3/3`；
  最终本地 test 结果为 GAUC `0.666354`、nDCG@5 `0.532377`、Primary `0.599365`。
- 该结果来自本地 test split，不是 hidden-test 分数。

### 交接顺序

先看 `README.md`、`SUBMISSION_SUMMARY.md`、`TRY.md`、`AGENT-TRY.md`、
`RUN_LOG.md` 和 `docs/architecture.md`。
