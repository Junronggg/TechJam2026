# Validation Performance Research

All results below use the untouched organizer evaluator and validation split only.
The new candidates have not been evaluated on test.

## Current ranking

| Candidate | Validation Primary | Evidence |
|---|---:|---|
| DCNv2/FM/LambdaRank rank blend (weights 0.25/0.50/0.25) | **0.604788** | Single validation fit; `+0.000248` vs prior DCNv2 incumbent; replicate |
| Categorical DCNv2 incumbent | **0.604540** | Prior agent validation winner |
| FM+BPR + raw tag, seed 0 | **0.604234** | Single seed; `+0.000538` vs matched tuned seed 0; replicate |
| FM+BPR ensemble, seeds 3+4, `lr=0.0005` | **0.604216** | Best of 26 validation seed subsets |
| FM+BPR ensemble, seeds 0-3 | 0.603991 | Four-member cumulative ensemble |
| FM+BPR + 50-bin duration, seed 0 | 0.603722 | Single seed; replication required |
| FM+BPR, seed 0, `lr=0.0005` | 0.603696 | Best single seed |
| FM+BPR, seeds 0-4 mean, `lr=0.0005` | 0.603521 | Improved 5/5 paired seeds |
| FM+BCE, seeds 0-4 mean | 0.601572 | Paired control |

The paired BPR-minus-BCE mean delta is `+0.001949` with population standard
deviation `0.000318`. BPR has lower across-seed Primary deviation (`0.000111`) than
BCE (`0.000316`).

## Rejected or unconfirmed branches

| Candidate | Primary | Interpretation |
|---|---:|---|
| Dense DCNv2 + hour/upload-age/freshness | 0.602543 | Beats official baseline, trails incumbent |
| Three-way blend using the weak temporal dense branch | 0.603076 | Diversity helps, but anchor is too weak |
| Field-aware FM+BPR, k=8, seed 0 | 0.602902 | Executable alternative; below incumbent |
| Linear+BPR, seed 0 | 0.602078 | Useful no-interaction control; below incumbent |
| Two random negatives per positive | 0.603380 | No gain over one negative |
| Same-author BPR negatives | 0.602531 | Regression |
| Same-tab BPR negatives | 0.590392 | Strong regression; tab is useful signal |
| User-author long-view rate | 0.602001 | Regression |
| Past-only user-tag long-view rate, seed 0 | 0.603238 | `-0.000458` vs matched tuned seed 0; within epsilon |
| Author long-view rate | 0.601635 | Regression |
| LightGBM binary | 0.599817 | Regression |
| LightGBM LambdaRank, capped at 100 trees | 0.593569 | Fast 15-second regression/control |
| FM+BPR + recent-positive transition, seed 0 | 0.602148 | Sequential hybrid; below matched FM+BPR |
| FPMC+BPR, k=8, seed 0 | 0.584411 | Fast sequence control; strong regression |
| Candidate-aware DIN+BCE, 10-event metadata history, k=8 | 0.603876 | Competitive (+0.002276 vs official), but below incumbent |
| LightGCN+BPR, k=32, 2 layers, 5 epochs | 0.491423 | New graph path works, but pure graph signal is strongly insufficient |
| LightGCN/FM hybrid, 2 layers, 5 epochs | 0.602685 | Blend selected graph weight 0.0; safely fell back to FM |

The raw-tag result is `+0.000538` over its matched tuned seed-0 control and only
`+0.000019` over the selected two-seed incumbent. It must not be promoted without
multi-seed replication. The duration result is only `+0.000026` over the best single
seed and also needs replication. The seeds 3+4 ensemble was selected after
testing multiple validation subsets, so its `0.604216` is model-selection evidence,
not an unbiased estimate of test performance.

The heterogeneous blend gain is also below the configured `0.002` epsilon. It is a
strict incumbent update for final model selection, but receives zero search reward and
is not an expansion parent until matched replication clears the confirmation rule.

The sequential rows use strictly earlier positive interactions. Rows tied at the same
timestamp are encoded before their outcomes update history, and validation/test context
is frozen from training history. The weak FPMC result shows that a first-order transition
alone discards too much author/tab/duration signal. Retaining those fields in the sequential
FM hybrid recovers most of the gap, but its `0.602148` still trails matched tuned FM+BPR
`0.603696`; neither sequential configuration should be promoted.

The first full-data LightGCN scout took about 18 seconds of model runtime and improved
monotonically across five epochs, but remained far below the `0.604788` incumbent. It is
retained as failure evidence and as an optional diversity component; the hybrid searches
blend weights including zero, so a weak graph component cannot force a regression.

The first candidate-aware DIN scout peaked at epoch 3 with GAUC `0.670614`, nDCG@5
`0.537138`, and Primary `0.603876`. Candidate-conditioned item/author/tag/duration
history therefore recovers meaningful signal and beats the official baseline, but this
small configuration does not beat the heterogeneous incumbent. Ranking-aligned and
hard-negative variants remain separate validation hypotheses.

Validation error-slice analysis reports 22,377 users, including 6,785 users with no
positive validation item. Consequently the dataset-level nDCG ceiling is approximately
`0.696787`. This does not prevent improvement, but it makes a literal 10–20% relative
Primary gain an aggressive research target, not an implementation guarantee.

## Reproduction commands

```powershell
.\.venv\Scripts\python.exe -X utf8 .\scripts\compare_objectives.py `
  --seeds 0 1 2 3 4 --bpr-learning-rate 0.0005

.\.venv\Scripts\python.exe -X utf8 .\scripts\run_validation.py `
  --model fm_ensemble --objective bpr --ensemble-size 2 `
  --ensemble-seed-set 3,4 --learning-rate 0.0005

.\.venv\Scripts\python.exe -X utf8 .\scripts\run_validation.py `
  --model din --objective group_softmax --sequence-length 20 `
  --negatives-per-positive 4 --hard-negative-pool-size 16
```

Keep research validation-only. Run `scripts/final_evaluate.py` once, only after the
team freezes the final validation-selected configuration.
