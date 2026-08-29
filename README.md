# TechJam 2026 — Autonomous ML Research Agent

Preparation workspace for TikTok TechJam Task 2 using the organizer-provided
KuaiRand-Pure starter kit. This repository currently contains environment and
benchmark configuration only; the autonomous research agent has not been built.

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
