# TechJam 2026 — Autonomous ML Research Agent

An autonomous validation-time research pipeline for the TikTok TechJam
KuaiRand-Pure recommendation benchmark. The repository contains the peer
implementation, its final local evaluation package, and the code needed to
reproduce the research loop.

## Repository layout

```text
src/techjam_agent/       planner, controller, models, skills and evaluator glue
scripts/                 setup, verification and experiment entry points
kuairand-starter-kit/    organizer-provided baseline and official evaluator
configs/                 project, experiment and evidence configuration
logs/run_final/           final run evidence required for submission
artifacts/                selected configuration, metrics and manifest
submissions/final.csv    final prediction file
RUN_LOG.md               consolidated per-iteration agent log
SUBMISSION_SUMMARY.md    final result and upload checklist
```

The detailed exploratory notes (`TRY.md`, `AGENT-TRY.md` and older research
documents) remain under `docs/` for local reference and are intentionally
ignored by Git. Historical runs, checkpoints, caches, datasets and secrets
are also ignored.

## Final submission package

The files intended for upload are exactly:

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

See [`SUBMISSION_SUMMARY.md`](SUBMISSION_SUMMARY.md) for the selected
configuration and metrics. The final local evaluator result is:

| Metric | Value |
|---|---:|
| GAUC | `0.666354` |
| nDCG@5 | `0.532377` |
| Primary | **`0.599365`** |

`submissions/final.csv` contains 170,588 predictions in the official
`row_id,user_id,video_id,score` format.

## Reproduce the environment

Use Python 3.9 or newer. Download and extract KuaiRand-Pure into:

```text
data/KuaiRand-Pure/data/
```

The required files are listed in `configs/project.json`. The dataset is local
and is not committed.

On macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python3 scripts/verify_setup.py
```

On Windows:

```powershell
.\scripts\setup.cmd
.\.venv\Scripts\Activate.ps1
python -X utf8 .\scripts\verify_setup.py
```

## Verification and experiments

Check the final prediction file without retraining:

```bash
python3 kuairand-starter-kit/submit.py \
  --check --split test \
  --data_dir data/KuaiRand-Pure/data \
  submissions/final.csv
```

Run the autonomous validation-only loop during development:

```bash
python3 scripts/run_agent.py --researcher deterministic
```

The controller selects experiments from the executable skill registry,
evaluates validation GAUC/nDCG@5, records the hypothesis and decision, updates
structured research memory, and enforces the experiment/time budget. Test
labels are not used to choose an experiment. The final one-shot evaluation is
represented by the already packaged `logs/run_final/` and `submissions/final.csv`.

Run lightweight code checks:

```bash
python3 -m compileall -q src scripts tests
PYTHONPATH=src python3 -m unittest discover -s tests -q
```

## Research result

The strongest validated direction in the recorded trajectory was a ranking
aware FM objective (`BPR`). A complementary DeepFM model improved the blend,
while redundant DCNv2/DeepFM combinations and several sparse history features
did not provide reliable additional gain. The agent log records the hypotheses,
configuration changes, validation metrics, controls, and stopping decision in
chronological order.

## Safety and reproducibility

- The organizer evaluator is treated as a fixed reference.
- Generated outputs are allow-listed in `.gitignore`; other runs stay local.
- `RUN_LOG.md` contains the consolidated agent trajectory and intervention
  count.
- No manual intervention was recorded for the packaged run.
- API credentials belong in the ignored `.env` file and are never committed.
