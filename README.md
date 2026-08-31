# TechJam 2026 — Autonomous ML Research Agent

Preparation workspace for TikTok TechJam Task 2 using the organizer-provided
KuaiRand-Pure starter kit. This repository currently contains environment and
benchmark configuration and a safe autonomous FM research MVP.

Research logs are separated by purpose:

- [`TRY.md`](TRY.md): model, loss, feature engineering, rolling validation and ensemble experiments.
- [`AGENT-TRY.md`](AGENT-TRY.md): Agent trajectories, runtime, memory ablations, stopping and intervention audits.

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

The official baseline currently needs only NumPy. Larger ML dependencies such as
PyTorch and LightGBM will be selected and pinned when model development begins,
avoiding a premature heavyweight environment.

## Dataset layout

Extract KuaiRand-Pure so the required files have this layout:

```text
data/
  KuaiRand-Pure/
    data/
      video_features_basic_pure.csv
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

The active system is a constrained autonomous research loop. It generates legal
candidate experiments, scores them by expected gain, evidence strength, novelty,
compute cost, and redundancy, selects one, trains it, evaluates validation ranking,
reflects on the result, updates structured memory, and repeats. It never gives an LLM
permission to edit the repository.

The memory has two levels: per-experiment hypotheses and family-level distilled
research patterns. Patterns can request confirmation, ensemble-only evaluation,
matched controls, one cheap evidence-gathering run, or stopping a repeatedly weak
direction. This updates the planning policy from experiment history; it is not RL or
parameter-level self-learning.

With `--auto-confirm`, a positive discovery is no longer merely tagged for later
human review. A deterministic evidence escalator can schedule expanding-window rolling
validation and, only after rolling passes, paired-seed confirmation. These are separate
confirmation actions: they cannot replace the champion, do not update the official
convergence streak, never read test labels, and consume both the action and wall-clock
budgets. Small redundant gains below `0.0002` stop before this expensive stage unless
the candidate has high novelty or a measured diversity advantage. Use
`--research-after-convergence` as well when the predeclared research run should finish
queued confirmation after the official convergence point has been recorded.

For categorical history features, a small apparent gain automatically schedules
constant, shuffled, and same-cardinality random controls. Control runs cannot become
the champion. Each model also writes validation-only prediction artifacts used for
fixed cold/warm, history, popularity, duration, and time slices plus conditional error
complementarity against the current champion.

The LightGBM branch first tests the original five categorical fields, then adds
continuous train-only user/item long-view rates and log interaction counts. Training
rows use leave-one-out target statistics; validation and test use train statistics only.

The FM branch also supports `training_objective: "bpr"`. BPR samples positive/negative
pairs within the same user and optimizes the positive item to rank above the negative
item, while leaving the model structure, features, split, and official evaluator
unchanged. The selected seed-0 configuration (`lr=0.0003`) improved validation Primary
from `0.601470` to `0.603963`; paired seeds 0–3 improved by `+0.002344` on average.

The NumPy model suite also includes DeepFM, `multitask_deepfm`, and low-rank DCNv2.
The multi-task model can isolate click, like, censored completion, capped log-watch,
and one-sided censored-watch targets; the current evidence-backed default is like-only.
The one-sided objective treats incomplete watch time as exact and a completed play as
a duration lower bound, so it is not capped MSE. Auxiliary
outcomes are training-only targets and are never passed as prediction-time features.
Multi-task DeepFM can also combine a within-user BPR long-view objective with the
pointwise auxiliary head. In the controlled like-only experiment this scored
`0.603322` versus `0.604400` for BCE; the externally reported `k=32, aux=0.3`
configuration scored `0.603610`. Both exact configurations remain reproducible but
are rejected by scoped evidence rather than advertised as improvements.
The FM branch also supports a learned constant `global_context` field; it improved
all three rolling folds, but won only 3/4 paired seeds and is therefore still a
candidate. Candidate-history fields failed their matched placebo check. Run the
reproducible checks with:

```bash
python scripts/run_rolling_validation.py
python scripts/run_multitask_rolling.py
python scripts/run_sequence_rolling.py
python scripts/run_dcnv2_rolling.py
python scripts/run_global_context_ablation.py
python scripts/run_constant_context_rolling.py
python scripts/run_global_context_multiseed.py
python scripts/analyze_conditional_complementarity.py
python scripts/evaluate_history_gated_ensemble.py
python scripts/run_lightweight_sequence_ablation.py
python scripts/evaluate_random_exposure_robustness.py
python scripts/run_pairwise_multitask_ablation.py
python scripts/run_censored_watchtime_ablation.py
```

Run the offline deterministic researcher first:

```powershell
.\.venv\Scripts\python.exe -X utf8 .\scripts\run_agent.py --researcher deterministic
```

On macOS or Linux, use:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python3 run_agent.py --researcher deterministic
```

Run the deterministic planner memory ablation with the same experiment budget:

```bash
python3 run_agent.py --researcher deterministic --memory-mode no_memory
python3 run_agent.py --researcher deterministic --memory-mode raw_history
python3 run_agent.py --researcher deterministic --memory-mode distilled_patterns
```

Each run records `planner_memory_mode` in `run_meta.json` and `summary.json`.
For a controlled comparison, `no_memory` disables dynamic outcome feedback but
keeps the same legal candidate catalog, static priors, and duplicate protection.
Each real selection also records counterfactual top choices for all three modes;
`memory_influenced_selections` counts how often memory actually changed the action
without launching extra training runs.
Run all three modes with one command and a shared experiment cap:

```bash
python3 scripts/run_memory_ablation.py --max-iterations 5
```

The validation-only comparison is written to `artifacts/memory_ablation.json`.

Stress-test cross-run planning against previously logged validation outcomes:

```bash
python3 scripts/replay_planner_memory.py --max-steps 12
```

This offline replay does not retrain models or load test metrics. In the current log
archive, `no_memory` and `raw_history` both formed the two-feature temporal combination
already rejected by rolling validation. The scoped `distilled_patterns` policy allows
the not-yet-rolled single feature but blocks the second action that would recreate the
rejected combination, saving one logged experiment. This demonstrates a changed planning
trajectory and deliberately does not generalize combination evidence to every individual
feature; it is not a new independent model-score result.

To let an OpenAI-compatible model choose experiments, copy `.env.example` to the
git-ignored `.env`, set the key locally, and run:

```bash
python3 scripts/check_llm_connection.py
python3 scripts/run_agent.py --researcher llm
```

`scripts/run_agent.py` loads `.env` automatically while preserving values already
exported by the shell. The example defaults to OpenRouter's OpenAI-compatible endpoint
and `openai/gpt-4.1-mini`; use `OPENAI_BASE_URL` and `OPENAI_MODEL` to select another
compatible provider or model. The connection checker reports only status, provider,
model, and token usage—it never prints the key.

The LLM is a constrained selector, not a code editor: it must return one complete
`changes` object exactly as listed in the deterministic top-five candidate ranking.
Invalid, incomplete, modified, or duplicate choices are retried and then audited before
deterministic fallback. `configs/research_context.json` supplies training-only dataset
facts and conditional method guidance; it contains no validation/test answer sheet.

Both the deterministic planner and the LLM receive validation-only persistent evidence from
`configs/research_evidence.json`. Before every run, `configs/evidence_manifest.json`
routes selected rolling, placebo, and paired-seed artifacts through the deterministic
policy builder. Generated policies carry artifact hashes, task/model/schema scope,
scientific verdicts, and competition status; they override old manual policies only for
the same family and matching scope. Override the files with `--evidence-file PATH` and
`--evidence-manifest PATH`. Test-split metric fields are not part of planning evidence
and are removed before an LLM prompt is sent.

Official convergence and optional research exploration are recorded separately. The
default command still stops after the organizer rule (`epsilon=0.002` for three rounds).
For a predeclared research run that records that point but continues within the same
iteration/time limits, use:

```bash
python3 scripts/run_agent.py --researcher llm --research-after-convergence
```

To enable the automatic evidence ladder for promising discoveries, run:

```bash
python3 scripts/run_agent.py \
  --researcher llm \
  --research-after-convergence \
  --auto-confirm
```

The ladder is `single validation discovery → 3-fold rolling → paired seeds 0–3`.
`summary.json` separates discovery, placebo-control, and confirmation actions and also
reports the underlying confirmation training-run count. Detailed records are stored in
`confirmation_history.jsonl` and `confirmations/`. A real executor smoke check reproduced
the existing `global_context` rolling result (3/3 wins, mean delta `+0.000810`, six
training runs); this validates orchestration and is not a new model-score claim.

Every run writes `submission_candidates.json`. A configuration's scientific verdict
(`VALIDATED`, `UNCERTAIN`, `REJECTED`, and so on) remains distinct from its competition
status (`ELIGIBLE`, `RESEARCH_ONLY`, or `NOT_ELIGIBLE`). This allows a promising but
not statistically confirmed candidate to remain comparable without presenting it as
a confirmed research result.

Audit whether the committed policy snapshot still matches the current artifacts:

```bash
python3 scripts/build_family_policies.py \
  --check-against configs/generated_family_policies.json
```

The LLM path automatically falls back to the deterministic policy if a proposal is
invalid, duplicated, or temporarily unavailable. `summary.json` records each fallback
and a sanitized provider error such as `HTTP 401`, so a deterministic proposal cannot
be mistaken for an LLM decision. Outputs are written to a timestamped
`logs/run_*` directory, `artifacts/best_config.json`, `artifacts/best_model.npz`, and
`submissions/final.csv`. All ranking metrics still come from the untouched organizer
`kuairand-starter-kit/evaluate.py`.

Development runs do not evaluate test by default. After the autonomous trajectory is
finished and frozen, run the one intended final test evaluation with:

```bash
python3 run_agent.py --researcher llm --finalize-test
```

A validation-only deterministic reference trajectory completed five experiments,
found the `0.604713` ensemble at iteration 2, reported zero interventions, and stopped
itself with `stop_reason=converged`. Its `memory_influenced_selections` was zero, so that
short trajectory demonstrates end-to-end autonomy but not a memory benefit. A longer
logged-validation replay subsequently showed that distilled cross-run policies skipped
two rolling-rejected temporal trials. A fresh five-experiment integration run then
reproduced every validation score, kept test metrics null, and reported zero manual
interventions; its short trajectory still had no memory-driven choice divergence.

### Research safety and evidence

- Every training experiment runs in an isolated child process with a 15-minute timeout.
- The official evaluator is checked against a pinned SHA-256 digest before data loading.
- Iterations expose validation metrics only; test metrics are computed once, after research,
  for the validation-best checkpoint. They are never sent back to the researcher.
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

Quick checks that do not require the dataset:

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall -q src scripts tests
```

The authoritative runnable agent is `src/techjam_agent/` through `python run_agent.py`.
The root command delegates to `scripts/run_agent.py`; its old architecture prototype is
still available only through explicit `--dry-run` or `--real-run` compatibility flags.

## Research-agent architecture

The active `src/techjam_agent` Planner/Critic loop and lightweight experiment-tree
manager include safety checks, persistent evidence memory, budget logic, evidence
logging, and allow-listed LLM proposals. The earlier root-level implementation remains
available for architecture smoke tests and comparison.
See [the architecture document](docs/architecture.md).

Run the architecture smoke test without training a model:

```powershell
.\.venv\Scripts\python.exe -X utf8 run_agent.py --dry-run
```

Run the real validation benchmark with the deterministic Planner and isolated FM
backend (baseline plus up to three experiments):

```powershell
.\.venv\Scripts\python.exe -X utf8 run_agent.py --real-run --iterations 3
```

Real runs use only the KuaiRand-Pure train/validation periods. Post-validation
rows are discarded before their relevance label is read. Evidence is stored in
`artifacts/real-runs/`, including checkpoints, validation predictions, stdout,
metrics, experiment lineage, and the final resource summary.

Run the unit tests:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -v
```
