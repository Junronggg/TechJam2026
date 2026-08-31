# Planner/Critic Research Architecture

This document covers both the legacy root `agent/` + `experiment/` prototype and
the active control architecture for the autonomous ML researcher. The prototype
is exercised by `python run_agent.py --dry-run` or `--real-run`; the authoritative
competition loop is `src/techjam_agent/`, launched with
`python3 scripts/run_agent.py`. The deterministic and LLM selectors share the
same executable skills, safety rules, memory, and evidence formats.

```mermaid
flowchart TD
    M[Research Manager] --> S[Select experiment-tree parent]
    S --> P[Research Prompt + Planner]
    P --> K[Skill Registry]
    K --> V[Controller validation and safe config builder]
    V --> R[Isolated experiment runner]
    R --> E[Official evaluator wrapper]
    E --> C[Grounded Critic]
    C --> X[Experiment memory and JSONL evidence]
    C --> Q[Evidence Escalator]
    Q --> R
    X --> M
```

## Prompt, Skills, and Controller

```text
Prompt     = how the researcher reasons (soft policy)
Skills     = what the current system can execute (registered capabilities)
Controller = what must never be violated (hard policy)
```

`src/techjam_agent/research_prompt.py` contains research principles, not model
answers. `src/techjam_agent/skills.py` registers ten reusable discovery,
training, evidence, and memory capabilities. Ranked candidates carry a primary
`skill_id`, risk, and required confirmation skills. The Controller resolves that
binding again before execution and rejects missing or mismatched capabilities.

The LLM cannot register a skill or edit code. An unavailable graph/model builder
is reported as a capability gap. A future Capability Builder must pass a bounded
schema, runner, and smoke-test contract before registration.

## Component boundaries

- `src/techjam_agent/controller.py` owns the loop, budgets, convergence, duplicate prevention,
  and final best-node designation.
- `src/techjam_agent/experiment_planner.py` generates and ranks executable candidates;
  `proposals.py` provides deterministic and OpenAI-compatible selectors.
- `src/techjam_agent/critic.py` separates measured observations from interpretations,
- `agent/planner.py` defines the structured Planner input/output contract. In this
  prototype the deterministic planner is an architecture fixture.
- `agent/critic.py` separates measured observations from interpretations,
  confidence, decisions, and follow-up tests.
- `src/techjam_agent/tree.py` keeps the best node from up to three research branches and
  scores candidates using exploitation, exploration, novelty, and runtime cost.
- `src/techjam_agent/memory.py` stores configurations, results, lessons,
  visit counts, and the best-node marker.
- `src/techjam_agent/config.py` restricts models, features, parameter ranges, and
  legal immutable configuration changes.
- `src/techjam_agent/isolated.py` and `runner.py` train real models in subprocesses,
  persist validation predictions, and use the hash-pinned organizer evaluator.
- `src/techjam_agent/evidence_escalator.py` turns promising discoveries into
  rolling and paired-seed confirmation jobs.
- `experiment/runner.py` is the experiment boundary. The dry-run backend returns
  clearly labelled simulated metrics; `--real-run` trains the official FM
  backend on validation only.
- `experiment/validator.py` restricts operations, models, features, parameter
  ranges, non-finite values, and protected organizer files before execution.
- `experiment/evaluator.py` loads the organizer evaluator read-only and can pin
  its SHA-256 digest.
- `experiment/logger.py` appends the required hypothesis, diff, metrics,
  critique, cost, and failure evidence as JSONL.
- `recommender/` defines immutable configuration transformations, feature/model
  registries, and the future `train_model()` boundary.

## Tree policy

The system uses a lightweight best-first tree, not full MCTS:

```text
priority = primary score
         + exploration bonus
         + branch novelty bonus
         - runtime penalty
```

Only the strongest node from each of the top three branches remains on the active
frontier. This preserves alternative hypotheses while staying practical within
the 50-iteration and six-hour competition limits.

## Safety invariants

1. The Planner returns an `ExperimentSpec`; it does not directly execute code.
2. Safe operations create a new immutable `ModelConfig` from a parent config.
3. Duplicate candidate configurations are rejected before execution.
4. The official evaluator and baseline metadata are protected paths.
5. Failed or worse experiments remain evidence but cannot erase the best node.
6. The Critic receives validation metrics only and must distinguish observation
   from interpretation.
7. Novel patch mode and work delegation are disabled in `configs/agent.json`.

## Current versus deferred

Currently runnable: deterministic and LLM selection, FM/DeepFM/multi-task/DCNv2/
sequence/LightGBM/ensemble training, safe config operations, isolated evaluation,
placebo controls, rolling and paired-seed confirmation, diagnostics, structured
memory, JSONL logging, budgets, dual convergence, and intervention accounting.

Deferred: a restricted Capability Builder, graph-model skill, and learned or
search-based higher-level planning. Arbitrary LLM code patches remain disabled.
Currently runnable **in this prototype**: schemas, tree selection, deterministic
Planner, safe config operations, validation, dry-run Runner, isolated official-FM
validation backend, grounded Critic, memory, JSONL logging, budgets, convergence,
and tests. The real backend discards post-validation rows before reading
`long_view` and stores each checkpoint/prediction in its experiment directory.

The live competition agent (`src/techjam_agent/` via `scripts/run_agent.py`) already
includes allow-listed LLM proposals, persistent evidence, and the isolated NumPy
training backends. Do not treat the deferred list below as the status of that stack.

Deferred in this prototype: repair generation and novel code patches.
