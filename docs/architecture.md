# Planner/Critic Research Architecture

This is the control architecture for the autonomous ML researcher. It is set up
before connecting an LLM or building new recommender models, so orchestration,
safety, memory, and evidence formats can be tested independently.

```mermaid
flowchart TD
    M[Research Manager] --> S[Select experiment-tree parent]
    S --> P[Planner]
    P --> V[Validator and safe config builder]
    V --> R[Isolated experiment runner]
    R --> E[Official evaluator wrapper]
    E --> C[Grounded Critic]
    C --> X[Experiment memory and JSONL evidence]
    X --> M
```

## Component boundaries

- `agent/manager.py` owns the loop, budgets, convergence, duplicate prevention,
  and final best-node designation.
- `agent/planner.py` defines the structured Planner input/output contract. The
  deterministic implementation is only an architecture fixture; an LLM adapter
  will implement the same interface later.
- `agent/critic.py` separates measured observations from interpretations,
  confidence, decisions, and follow-up tests.
- `agent/tree.py` keeps the best node from up to three research branches and
  scores candidates using exploitation, exploration, novelty, and runtime cost.
- `agent/memory.py` stores parent/child lineage, configurations, results, lessons,
  visit counts, and the best-node marker.
- `experiment/runner.py` is the canonical experiment boundary. A real backend
  will later train, predict, and call the official evaluator. The current dry-run
  backend returns clearly labelled simulated metrics only.
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

Currently runnable: schemas, tree selection, deterministic Planner, safe config
operations, validation, dry-run Runner, isolated official-FM validation backend,
grounded Critic, memory, JSONL logging, budgets, convergence, and tests. The real
backend discards post-validation rows before reading `long_view` and stores each
checkpoint/prediction in its experiment directory.

Deferred to the next build phase: LLM provider integration, new feature
transforms, additional model families, repair generation, and novel code patches.
