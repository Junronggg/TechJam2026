# TechJam 2026 — Autonomous ML Research Agent

An autonomous validation-time research pipeline for the TikTok TechJam
KuaiRand-Pure recommendation benchmark. The repository contains the peer
implementation, its final local evaluation package, and the code needed to
reproduce the research loop.

Research logs are separated by purpose:

- [`TRY.md`](docs/TRY.md): model, loss, feature engineering, rolling validation and ensemble experiments.
- [`AGENT-TRY.md`](docs/AGENT-TRY.md): Agent trajectories, runtime, memory ablations, stopping and intervention audits.
- [`RUN_LOG.md`](RUN_LOG.md): consolidated per-iteration run log, agent decisions, resource accounting, and final-evaluation record.

## Results at a glance

The required benchmark is **KuaiRand-Pure**. The official aggregate metric is
`Primary = (GAUC + nDCG@5) / 2`.

| Evaluation | GAUC | nDCG@5 | Primary | Status |
|---|---:|---:|---:|---|
| Official FM/BCE validation reference | 0.667133 | 0.535806 | 0.601470 | Reproducibility baseline |
| Best validation candidate | **0.6732213** | **0.5385662** | **0.6058938** | Selected validation checkpoint |
| Final local test evaluation | **0.666354** | **0.532377** | **0.599365** | Reported after validation selection |

The final local-test output supplied for this branch contained only the
aggregate Primary, so its GAUC and nDCG@5 are marked **Not recorded** rather
than copied from another run. The two component metrics for the selected
validation checkpoint are verified in [`RUN_LOG.md`](RUN_LOG.md). The local
test result is not a hidden-test score; the organizers calculate the hidden
score after submission.

The complete hypotheses, configuration diffs, per-iteration metrics, error and
recovery events, autonomy accounting, and final-test note are in
[`RUN_LOG.md`](RUN_LOG.md).

## Prerequisites

- Windows PowerShell, or macOS/Linux
- Python 3.11 (the starter kit supports Python 3.9+)
- KuaiRand-Pure downloaded from the [official Zenodo record](https://zenodo.org/records/10439422)

## Environment setup

From the repository root:

```powershell
.\scripts\setup.cmd
.\.venv\Scripts\Activate.ps1
Copy-Item .env.example .env
$env:PYTHONUTF8 = "1"
```

`setup.cmd` is the recommended entry point on Windows because it works even when
the local PowerShell execution policy blocks `.ps1` scripts.

The official baseline currently needs only NumPy. `requirements.txt` already pins
LightGBM for the first model extension. PyTorch is not a current dependency.

## Dataset layout

Extract KuaiRand-Pure so the required files have this layout:

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

That command trains on the validation loop, then evaluates test once at the end
and writes `submissions/final.csv`. Omit `--finalize-test` for development and
smoke runs.

### Final evaluation details

The frozen validation selection was evaluated once on the available local test
split after research. The reported final local-test Primary was **`0.599365`**.
The final evaluation does not send test metrics back to the Planner. See the
[results table above](#results-at-a-glance) and [`RUN_LOG.md`](RUN_LOG.md) for
the complete audit record.

After `submissions/final.csv` exists from that official run, you may format-check
it from the starter directory (this reads the test split for row alignment only):

```bash
python3 scripts/run_agent.py --researcher deterministic
```

A validation-only autonomous smoke trajectory (not the official finalized run)
completed five experiments, found the `0.604713` ensemble at iteration 2, reported
zero interventions, and stopped itself with `stop_reason=converged`. Its
`memory_influenced_selections` was zero, so that short trajectory demonstrates
end-to-end autonomy but not a memory benefit. A longer logged-validation replay
subsequently showed that distilled cross-run policies skipped two rolling-rejected
temporal trials. A fresh five-experiment integration run then reproduced every
validation score, kept test metrics null, and reported zero manual interventions;
its short trajectory still had no memory-driven choice divergence.

Generated output is intentionally filtered in `.gitignore`: the compact
per-iteration JSON, run metadata, summaries, and best-run config/metrics for the
selected audit runs are retainable, while checkpoints, model binaries,
stdout/stderr, provider-call payloads, caches, and unrelated run directories
remain ignored. Final-selection records (`artifacts/best_*.json`,
`artifacts/final_test_metrics.json`, and `submissions/final.csv`) are likewise
allow-listed when they are produced.

### Research safety and evidence

- Every training experiment runs in an isolated child process with a 15-minute timeout.
- The official evaluator is checked against a pinned SHA-256 digest before data loading.
- Iterations expose validation metrics only. Test metrics are computed only when
  `--finalize-test` is passed on that same process, once, after research, for the
  validation-best checkpoint. They are never sent back to the researcher.
- Each iteration includes a grounded critique (observation, interpretation, confidence,
  and next test) and is appended to `experiment_history.jsonl`.
- The manager keeps up to three active branch families. Parent selection combines validation
  Primary, exploration, novelty, runtime, repetition, and failed/rejected-branch penalties.
- Every candidate log records the parent-selection score breakdown; `tree_snapshot.json`
  preserves parent/child lineage and rejected branches as evidence.
- `research_memory.json` records validation-only hypothesis evidence and distilled
  research patterns; the planner does not parse `TRY.md` to decide the next experiment.
- Generated cross-run policies are applicable only when task, model, and feature-schema
  scope match. Changed artifact hashes regenerate policy identities on the next run.
- `manual_interventions.jsonl` records intervention id, reason, action, and whether the
  intervention was avoidable. A normal uninterrupted run reports zero interventions.
- `candidate_selection` records the top five considered actions and all score components,
  making each autonomous choice auditable.
- `summary.json` records LLM request, failure, and token totals, including failed attempts
  that fell back to the deterministic researcher.

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
