# Submission Summary

## Final result

The final prediction file is [`submissions/final.csv`](submissions/final.csv).
It contains `170,588` rows in the official order.

| Metric | Value |
|---|---:|
| GAUC | `0.666354` |
| nDCG@5 | `0.532377` |
| Primary | **`0.599365`** |

## Selected configuration

- FM+BPR ranking signal blended with DeepFM+BCE.
- FM branch learning rate: `0.0003`.
- DeepFM branch weight: `0.6`.
- Ensemble calibration: `fm_zscore_deepfm_rank`.
- Seed: `0`.

## Agent run

The five-step autonomous run is archived under `logs/run_final/`. It contains the
per-iteration hypotheses, configuration changes, validation metrics, decisions,
and the derived research trajectory. No manual intervention was recorded.

## Uploaded generated files

```text
logs/run_final/summary.json
logs/run_final/research_trajectory.json
logs/run_final/experiment_history.jsonl
logs/run_final/iteration_*.json
artifacts/best_config.json
artifacts/best_metrics.json
artifacts/best_manifest.json
submissions/final.csv
```

Other run directories, checkpoints, caches, datasets, and local research notes are
excluded by `.gitignore`.
