# TechJam 2026 — Autonomous ML Research Agent

This repository contains our validation-safe autonomous research system for the
TikTok TechJam KuaiRand-Pure recommender benchmark. The `peer` research branch is
the source of the curated experiment history; the merged `main` code supplies the
expanded runner, model implementations, safety checks, and test suite.

## What is included

- `src/techjam_agent/`: controller, researcher/planner, memory, feature builders,
  FM/BPR/DeepFM/DCNv2, ranking, sequence, graph, and ensemble operators.
- `scripts/`: setup, data profiling, validation, autonomous runs, and final-evaluation
  utilities.
- `configs/`: project, experiment, evidence, and research-state configuration.
- `TRY.md`: model, loss, feature, rolling-validation, and ensemble experiments.
- `AGENT-TRY.md`: agent trajectories, memory/policy tests, fallback and autonomy audits.
- `RUN_LOG.md`: the submission-oriented run and iteration record.
- `docs/`: architecture and benchmark notes.

Generated data, checkpoints, API secrets, and exploratory outputs are intentionally
ignored. Only a selected sanitized run should be added to a submission package.

## Reproduce the benchmark

Download KuaiRand-Pure from the [official Zenodo record](https://zenodo.org/records/10439422)
and extract it as:

```text
data/KuaiRand-Pure/data/
  video_features_basic_pure.csv
  user_features_pure.csv
  log_standard_4_08_to_4_21_pure.csv
  log_standard_4_22_to_5_08_pure.csv
```

The dataset directory is ignored by Git. Set `TECHJAM_DATA_DIR` in the local `.env`
when storing it elsewhere. Never commit `.env` or an API key.

On macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python3 scripts/verify_setup.py
```

On Windows PowerShell:

```powershell
.\scripts\setup.cmd
.\.venv\Scripts\Activate.ps1
Copy-Item .env.example .env
.\.venv\Scripts\python.exe -X utf8 .\scripts\verify_setup.py
```

Reproduce the organizer baseline from the starter directory:

```bash
cd kuairand-starter-kit
python3 baseline.py --model fm --data_dir ../data/KuaiRand-Pure/data
```

The expected validation Primary is approximately `0.6015` (GAUC approximately
`0.6674`, nDCG@5 approximately `0.5357`). All ranking metrics are computed by the
untouched organizer evaluator.

## Autonomous research loop

The one authoritative entry point is `scripts/run_agent.py`:

```bash
python3 scripts/run_agent.py --researcher deterministic --autonomous --max-iterations 10
```

With an OpenAI-compatible provider, copy `.env.example` to `.env`, set
`OPENAI_API_KEY`, `OPENAI_BASE_URL`, and `OPENAI_MODEL`, then run:

```bash
python3 scripts/run_agent.py --researcher llm --open-ended --autonomous --max-iterations 10
```

If an LLM proposal is invalid, duplicated, unavailable, or fails authentication,
the controller records the failure and uses a deterministic fallback. The fallback
is visible in the run summary and does not masquerade as an LLM decision.

Each iteration follows:

```text
observe validation evidence
  → propose one legal experiment
  → run in an isolated child process
  → evaluate GAUC / nDCG@5 / Primary
  → critique and update research memory
  → keep, reject, or reinterpret
```

The controller enforces the hard limits (at most 50 experiments and 6 hours),
prevents test metrics from entering planning, rejects duplicate configurations,
and records parent/child lineage, errors, recoveries, interventions, and LLM token
usage. `--research-after-convergence` is available for a pre-declared research run;
it does not change the official convergence rule.

The selected final local test result is GAUC `0.666354`, nDCG@5 `0.532377`, and
Primary `0.599365` over `170,588` rows. This is the only final score retained in the
submission summary; it is a local test-split result, not a hidden-test score.

## Our current validation evidence

The peer branch's strongest robust result is the FM+BPR plus DeepFM ensemble:

- Stable ensemble candidate: Primary `0.604713`; rolling evidence `3/3` wins.
- Best single-split candidate: Primary `0.6052911282`; paired-seed evidence is
  `3/4` positive, so it remains an uncertain research candidate rather than a
  confirmed improvement.
- BPR improves the official FM baseline; LightGBM with the tested BCE/statistics
  variants, global target-rate fields, explicit sparse crosses, and several sequence
  controls were rejected or reinterpreted.

These are validation results from the peer research archive, not hidden-test claims.
See [TRY.md](TRY.md), [RUN_LOG.md](RUN_LOG.md), and [AGENT-TRY.md](AGENT-TRY.md) for
the exact hypotheses, configuration changes, metrics, controls, and decisions.

## Submission package

Keep the uploaded package small and auditable:

- source code under `src/`, `scripts/`, `configs/`, `tests/`, and
  `kuairand-starter-kit/`;
- `README.md`, `SUBMISSION_SUMMARY.md`, `PROJECT_RESUME.md`, `TRY.md`,
  `AGENT-TRY.md`, and `RUN_LOG.md`;
- one sanitized directory `logs/run_final/` containing
  `summary.json`, `research_trajectory.json`, `experiment_history.jsonl`, and
  `iteration_*.json`;
- the selected final local result recorded in `SUBMISSION_SUMMARY.md`.

Do not upload `.env`, API keys, the dataset/archive, `.venv/`, Python caches,
exploratory logs, checkpoints, or stale final artifacts from another branch. Only
`logs/run_final/` is retained as the run archive.

## Quick checks

```bash
python3 -m compileall -q src scripts tests
python3 -m unittest discover -s tests -v
```

See [docs/architecture.md](docs/architecture.md) for the controller/evidence
contract and [SUBMISSION_SUMMARY.md](SUBMISSION_SUMMARY.md) for the concise handoff.
