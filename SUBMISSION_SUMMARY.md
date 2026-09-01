# Submission Summary

## Project

**TechJam 2026 — Autonomous ML Research Agent for KuaiRand-Pure.** The system runs
validation-safe recommender experiments, evaluates them with the official GAUC and
nDCG@5 evaluator, records evidence, and chooses the next experiment without a
human selecting every action.

## What the agent does

1. Loads the baseline and validation-only research history.
2. Proposes one legal experiment and records its hypothesis and configuration diff.
3. Runs training in an isolated process with the 50-experiment / 6-hour limits.
4. Computes validation GAUC, nDCG@5, and Primary, then records errors and recovery.
5. Updates structured memory and separates `KEEP`, `REJECT`, `REINTERPRET`, and
   `ENSEMBLE_ONLY` conclusions.
6. Stops at the declared convergence rule or when the search budget is exhausted.

LLM proposals are optional. Invalid or unavailable LLM calls are recorded and use
the deterministic fallback. Test metrics are excluded from planning and are not
present in the selected peer run.

## Final test result

| Result | GAUC | nDCG@5 | Primary | Status |
|---|---:|---:|---:|---|
| Final local test evaluation | 0.666354 | 0.532377 | **0.599365** | **FINAL** |

This is the selected local test-split result over `170,588` rows. It is the only
final score retained in this submission summary. It is not a hidden-test score and
was not used by the agent to choose experiments.

The research trajectory and all validation experiments remain auditable in `TRY.md`,
`AGENT-TRY.md`, and `RUN_LOG.md`; they are not additional final scores.

## Required evidence files

- `logs/run_final/summary.json`
- `logs/run_final/research_trajectory.json`
- `logs/run_final/experiment_history.jsonl`
- `logs/run_final/iteration_*.json`
- `TRY.md` (ML/feature chronology)
- `AGENT-TRY.md` (agent/autonomy chronology)
- `RUN_LOG.md` (combined handoff log)

## Human intervention

The selected peer run reports **0 manual interventions**. The run was deterministic,
validation-only, and required no human action, restart, or manual experiment choice.

## Submission hygiene

Do not commit `.env`, API keys, the dataset, virtual environments, caches, raw
checkpoints, or exploratory run directories. Only `logs/run_final/` is retained as
the run archive for this package.
