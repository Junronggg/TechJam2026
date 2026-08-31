# TechJam 2026 — Autonomous ML Research Agent

Preparation workspace for TikTok TechJam Task 2 using the organizer-provided
KuaiRand-Pure starter kit. This repository currently contains environment and
benchmark configuration and a safe autonomous recommender research system.

## Prerequisites

- Windows PowerShell
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

The environment pins the dependencies used by the active FM and LightGBM branches.

## Dataset layout

Extract KuaiRand-Pure so the required files have this layout:

```text
data/
  KuaiRand-Pure/
    data/
      video_features_basic_pure.csv
      user_features_pure.csv
      log_standard_4_08_to_4_21_pure.csv
      log_standard_4_22_to_5_08_pure.csv
```

The data directory is ignored by Git. If you store it elsewhere, set
`TECHJAM_DATA_DIR` in `.env` and pass the same path to starter-kit commands.
The downloaded archive is verified against MD5
`0820331067a3784d9691136f772b35a7` before use.

## Preparation checks

```powershell
.\.venv\Scripts\python.exe -X utf8 .\scripts\verify_setup.py
```

Once the dataset check passes, reproduce the organizer baseline from the starter
directory:

```powershell
Set-Location .\kuairand-starter-kit
..\.venv\Scripts\python.exe -X utf8 baseline.py --model fm --data_dir ..\data\KuaiRand-Pure\data
```

Expected validation scores are approximately GAUC `0.6674`, nDCG@5 `0.5357`,
and primary `0.6016`. Do not start agent/model development until these reproduce.

Then generate and validate a sample submission:

```powershell
..\.venv\Scripts\python.exe -X utf8 submit.py --make --split test --data_dir ..\data\KuaiRand-Pure\data ..\submissions\baseline.csv
..\.venv\Scripts\python.exe -X utf8 submit.py --check --split test --data_dir ..\data\KuaiRand-Pure\data ..\submissions\baseline.csv
```

## Protected organizer files

Treat `kuairand-starter-kit/evaluate.py` and the published baseline metadata as
fixed references. Keep generated data, logs, artifacts, submissions, secrets,
and the virtual environment out of version control.

## Autonomous research MVP

Before model search, build the validation-safe EDA/data profile:

```powershell
.\.venv\Scripts\python.exe -X utf8 .\scripts\profile_dataset.py
```

This produces a readable report plus structured Planner evidence under
`artifacts/data-profile/`. LLM research runs automatically load the generated
profile, while test-period labels remain unread. See
[the data-profiling guide](docs/data-profiling.md).

The built-in experiment space now includes linear, FM, field-aware FM (FFM),
FM-ensemble, LightGBM/LambdaRank, DeepFM, categorical and dense DCNv2, a two-tower
candidate model, sequential FM, FPMC, SASRec, multitask learning, and a heterogeneous
DCNv2/FM/tree blend. In addition, the LLM researcher has an open-ended generated-code
branch for new architectures and feature transforms. The researcher proposes a validated experiment as JSON, while deterministic code
trains, evaluates, keeps or rejects, logs, and checks convergence. In autonomous mode,
the Controller also generates bounded, compatible feature/optimizer bundles that were not
hand-written as recipes, so the Planner can discover useful interactions without a human
selecting the next experiment. The registered model and setup path remains the execution
fallback and safety boundary. In open-ended mode,
the LLM can also propose a compact model/feature module implementing the validated
``fit_validate``/``finalize`` contract. The Controller hashes, statically checks, and
isolates that module before execution; it never receives test labels or promotion authority.

The LightGBM branch first tests the original five categorical fields, then adds
continuous train-only user/item long-view rates and log interaction counts. Training
rows use leave-one-out target statistics; validation and test use train statistics only.

The FM branch also supports `training_objective: "bpr"`. BPR samples positive/negative
pairs within the same user and optimizes the positive item to rank above the negative
item, while leaving the model structure, features, split, and official evaluator
unchanged. With learning rate `0.0005`, it improved all five paired seeds: mean
validation Primary rose from `0.601572` (BCE) to `0.603521` (BPR). This is a historical
replication result; the current durable winner is the hybrid-blend configuration
reported below.

The operator registry also exposes request hour/weekday, upload age/freshness, user
activity, video type, time-decayed item/author/tag popularity, recent-history candidate
similarity, a train-vocabulary raw tag field, strictly-past
user-tag impression/rate features with tag/global backoff, leakage-safe
author/user-author history features, a train-fitted 50-bin duration feature,
controlled 1/2/4 BPR negatives, context-matched negative sampling, sampled group-softmax,
and grouped LightGBM LambdaRank. Encoded data, feature columns, and sequential
representations are fingerprinted and cached under `artifacts/cache/`. Negative results
remain logged.
See [the performance research log](docs/performance-research.md).

Research runs start from the durable validation incumbent and load
validation-only evidence from earlier `logs/run_*` records and archived validation
candidates. This prevents the LLM from paying to rediscover known configurations.
The linear+BPR control (`0.602078`) and FFM+BPR (`0.602902`, k=8) are fully executable
but currently trail the FM+BPR ensemble.

The first sequence screen is also complete. A strict past-only FPMC control scored
`0.584411`; the stronger FM plus recent-positive transition hybrid scored `0.602148`.
Both remain executable negative controls. Their results do not justify adding a large
Transformer blindly; a future SASRec candidate must introduce richer sequence evidence
while preserving candidate metadata/context.

The current durable validation winner is a `hybrid_blend` with BPR, two sampled
negatives, same-author negative sampling, train-fitted duration buckets, and raw tags.
It scored GAUC `0.6732213`, nDCG@5 `0.5385662`, and Primary `0.6058938` on the
validation split. Relative to the official baseline Primary `0.6016`, this is an
absolute improvement of approximately `+0.0043`.

After the validation winner was frozen, the explicit final-evaluation command produced
the following local test-split result:

| Split/result | GAUC | nDCG@5 | Primary |
| --- | ---: | ---: | ---: |
| Official baseline (validation) | 0.6674 | 0.5357 | 0.6016 |
| Validation-best hybrid blend | 0.6732213 | 0.5385662 | 0.6058938 |
| Final local test evaluation | 0.6665823 | 0.5314327 | 0.5990075 |
| Validation delta over official baseline | +0.0058213 | +0.0028662 | +0.0042938 |

The final local test metrics are:

```text
GAUC:    0.6665823
nDCG@5:  0.5314327
Primary: 0.5990075
Users:   23,875
Rows:    170,588
```

These are local test metrics only. They were not supplied to the Planner and are not a
hidden-test result; the competition organizers evaluate the submitted `final.csv` on
the hidden test set.

### Reproducible run evidence

The two representative runs used for the submission record are:

| Run | Researcher | Iterations | Candidate experiments | Manual interventions | LLM requests/failures | Best validation Primary |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `run_20260831T171347Z` | Deterministic, autonomous | 4 | 3 | 0 | 0 / 0 | 0.6058938 |
| `run_20260831T131357Z` | OpenAI-compatible LLM with fallback | 6 | 5 | 0 | 5 / 5 authentication failures; deterministic fallback | 0.6053735 |

Each run directory contains `run_meta.json`, `iteration_*.json`, `summary.json`,
`research_trajectory.json`, `tree_snapshot.json`, and `experiment_history.jsonl`.
The LLM run also contains the sanitized/redacted `llm_calls.jsonl` audit. Review it
before upload and remove any provider-specific or secret-bearing fields. The per-iteration
records include the hypothesis, candidate/configuration diff, validation metrics,
errors or recovery actions, and promotion decision required by the Starter Kit.

For top-five experiments, set the legal `validation_metric` hyperparameter to
`"nDCG@5"`. This changes only epoch stopping and blend-weight selection; the
Controller still promotes the official Primary score, and the untouched evaluator
still reports both GAUC and nDCG@5. The run summary also records the best nDCG
checkpoint even when its Primary score is not the incumbent.

Run the autonomous researcher (it uses the LLM when `OPENAI_API_KEY` is set, otherwise the
offline researcher automatically):

```powershell
.\.venv\Scripts\python.exe -X utf8 .\scripts\run_agent.py --researcher auto --autonomous --max-iterations 10
```

To make the autonomous search explicitly investigate top-five quality while
preserving the official Primary promotion rule, add `--focus-metric "nDCG@5"`.

On macOS or Linux, use:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python3 scripts/run_agent.py --researcher auto --autonomous
```

To let an OpenAI-compatible model choose experiments, set `OPENAI_API_KEY` and run:

```bash
python3 scripts/run_agent.py --researcher llm --open-ended --autonomous --model gpt-4.1 --max-iterations 10
```

For OpenRouter on Windows PowerShell, the client still uses the OpenAI-compatible
environment-variable names:

```powershell
$env:OPENAI_API_KEY = "your-openrouter-key"
$env:OPENAI_BASE_URL = "https://openrouter.ai/api/v1"
$env:OPENAI_MODEL = "openai/gpt-4.1-mini"
.\.venv\Scripts\python.exe -X utf8 .\scripts\run_agent.py --researcher llm --open-ended --autonomous --max-iterations 10
```

Never put the real key in `.env.example`, source files, logs, or the Git history.

To run a validation-only top-five trial directly:

```powershell
.\.venv\Scripts\python.exe -X utf8 .\scripts\run_validation.py `
  --model hybrid_blend --objective bpr --validation-metric "nDCG@5" `
  --blend-mode zscore --negatives-per-positive 2
```

Open-ended code-branch proposals are enabled by default for the LLM researcher;
pass `--no-open-ended` to run only the registered configuration catalog.

Use `--start-from baseline` only when deliberately reproducing the original FM
baseline. Use `--fresh-memory` only for a controlled ablation of persistent memory.
At least five candidate experiments are attempted before convergence can stop a run.

The LLM path records an explicit `deterministic_fallback` if a proposal is invalid,
duplicated, or unavailable. Sanitized per-attempt evidence is written to
`logs/run_*/llm_calls.jsonl` and `logs/run_*/llm_calls/`; the proposal/metric timeline is
written to `research_trajectory.json`. Every run keeps its own winner under
`logs/run_*/best/`. Shared `artifacts/best_config.json` and `best_model.npz` are updated
only when a run meets or beats the durable incumbent, so a weaker fresh run cannot
erase the project best. All ranking metrics still come from the untouched organizer
`kuairand-starter-kit/evaluate.py`. Use `--final-eval` only after selecting the final
validation winner, or run `scripts/final_evaluate.py` explicitly.

To evaluate the already-selected validation winner and create the submission file:

```powershell
.\.venv\Scripts\python.exe -X utf8 .\scripts\final_evaluate.py `
  --data-dir .\data\KuaiRand-Pure\data `
  --output .\submissions\final.csv
```

The command reads `artifacts/best_config.json` and `artifacts/best_model.npz`, then
writes `artifacts/final_test_metrics.json` and `submissions/final.csv`.

### GitHub/submission package

Commit the reproducible implementation and a small, curated evidence set:

- `README.md`, `docs/`, `requirements.txt`, and `.env.example`
- `src/techjam_agent/` (controller, planner/researcher, memory, models, features, and evaluator glue)
- `scripts/` (setup, profiling, validation, agent execution, and final evaluation)
- `configs/` (project, experiment, feature-leakage, evidence, and research settings)
- `tests/`
- `kuairand-starter-kit/` including the organizer evaluator and reference baseline
- The selected run-log directories listed above, or an equivalent sanitized archive
- `submissions/final.csv`, `artifacts/best_config.json`, `artifacts/best_manifest.json`,
  `artifacts/best_metrics.json`, `artifacts/best_model.npz`, and
  `artifacts/final_test_metrics.json`

Do not commit `.env` or any API key, `.venv/`, the KuaiRand data/archive, feature/data
caches, Python bytecode, or every exploratory artifact from `artifacts/`, `logs/`, and
`submissions/`. If logs or final artifacts are kept under the repository's ignored
paths, add only the selected files explicitly (for example with `git add -f`).

### Research safety and evidence

- Every training experiment runs in an isolated child process with a 15-minute default
  timeout; the three-component blend has a 20-minute model-specific limit.
- The official evaluator is checked against a pinned SHA-256 digest before data loading.
- Iterations expose validation metrics only. Test metrics are never sent back to the
  researcher and are computed only through the explicit final-evaluation path.
- Each iteration includes a grounded critique (observation, interpretation, confidence,
  next test, and selective structured reflection) and is appended to
  `experiment_history.jsonl`.
- The manager keeps up to three active branch families. Parent selection combines validation
  Primary, exploration, novelty, runtime, repetition, and failed/rejected-branch penalties.
- Only strict global validation winners (plus the baseline root) can be expansion parents.
  Every weaker branch remains zero-reward scientific evidence and is never retrieved as a
  success.
- Every candidate log records the parent-selection score breakdown; `tree_snapshot.json`
  preserves parent/child lineage and rejected branches as evidence.
- `summary.json` records LLM request, failure, and token totals, including failed attempts
that fell back to the deterministic researcher.

For ordinary configuration proposals, the Controller supplies an exact catalog of
compatible, non-duplicate `candidate_id` values and an explicit catalog of valid
evidence IDs; local validation requires the returned changes to match the selected
candidate. Open-ended proposals still select the Controller-issued `code_branch`
candidate and must pass the source contract and static gate. Deterministic, fallback,
and LLM researchers are normalized through this same contract.

Quick checks that do not require the dataset:

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall -q src scripts tests
```

The one authoritative implementation is `scripts/run_agent.py` backed by
`src/techjam_agent/`. The earlier root-level prototype has been removed so planner,
runner, critic, memory, and tests cannot silently diverge between two systems.
See [the architecture document](docs/architecture.md) for the progressive research
loop and its evidence contract.

Run the unit tests:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -v
```
