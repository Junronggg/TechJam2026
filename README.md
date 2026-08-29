# TechJam 2026 — Autonomous ML Research Agent

Preparation workspace for TikTok TechJam Task 2 using the organizer-provided
KuaiRand-Pure starter kit. This repository currently contains environment and
benchmark configuration and a safe autonomous FM research MVP.

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

The first version uses a constrained experiment space: the researcher proposes a
validated FM or LightGBM experiment as JSON, while deterministic code trains,
evaluates, keeps or rejects, logs, checks convergence, and writes the best test
submission. It never gives an LLM permission to edit the repository.

The LightGBM branch first tests the original five categorical fields, then adds
continuous train-only user/item long-view rates and log interaction counts. Training
rows use leave-one-out target statistics; validation and test use train statistics only.

The FM branch also supports `training_objective: "bpr"`. BPR samples positive/negative
pairs within the same user and optimizes the positive item to rank above the negative
item, while leaving the model structure, features, split, and official evaluator
unchanged. The first seed-0 comparison improved validation primary from `0.601470` to
`0.603396`; this should be repeated across seeds before claiming a stable gain.

Run the offline deterministic researcher first:

```powershell
.\.venv\Scripts\python.exe -X utf8 .\scripts\run_agent.py --researcher deterministic
```

On macOS or Linux, use:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python3 scripts/run_agent.py --researcher deterministic
```

To let an OpenAI-compatible model choose experiments, set `OPENAI_API_KEY` and run:

```bash
python3 scripts/run_agent.py --researcher llm --model gpt-4.1-mini
```

The LLM path automatically falls back to the deterministic policy if a proposal is
invalid, duplicated, or temporarily unavailable. Outputs are written to a timestamped
`logs/run_*` directory, `artifacts/best_config.json`, `artifacts/best_model.npz`, and
`submissions/final.csv`. All ranking metrics still come from the untouched organizer
`kuairand-starter-kit/evaluate.py`.

Quick checks that do not require the dataset:

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall -q src scripts tests
## Research-agent architecture

The Planner/Critic research loop and lightweight experiment-tree manager are now
scaffolded. The interfaces, safety checks, memory, budget logic, and evidence
logging are runnable, but no LLM or new recommender implementation is connected.
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
