# Research Foundations

This document records the published work that informed the design of our autonomous
ML research agent, and — just as importantly — the parts of that work we chose not to
build. It describes the authoritative implementation in `src/techjam_agent/`, driven by
`scripts/run_agent.py`.

We did not reproduce any of these papers. None of our components is a reimplementation,
and we report no comparison against any paper's published results. The relationship is
conceptual: each paper clarified a design question we faced, and we adopted a much
smaller mechanism that fits a 50-iteration, six-hour budget on a single machine.

## Research motivation

Our task is not recommendation itself. It is autonomous *experimentation* on a
recommendation benchmark: an agent must choose the next experiment, train a real model,
read validation metrics, and decide what to try next, without a human in the loop.

That reframing is what makes the LLM-agent recommender literature useful to us but never
directly reusable. Those systems plan over *user turns* and treat a recommendation as the
output. We plan over *experiments* and treat a trained model's validation score as the
output. Four recurring problems in that literature map cleanly onto ours:

1. **How does an agent avoid re-deriving what it already learned?** Every step should be
   conditioned on all previously explored states, not just the last one (RecMind).
2. **How should planning be separated from execution?** A planner that also executes tends
   to be unsafe and hard to audit; and both successful and failed attempts should be
   reused rather than discarded (TAIRA).
3. **How much search is worth it?** Deliberate tree-structured planning beats greedy
   choice, but full tree search has a cost that must be justified (SAPIENT).
4. **When is slow, deliberate reasoning worth its training cost?** (STARec).

Our answers are deliberately conservative, because in our setting one "simulation" is a
real model training run costing minutes, not a cheap forward pass.

## Paper-to-component mapping

| Paper | Idea we drew on | Our corresponding component | Relationship |
| --- | --- | --- | --- |
| RecMind (Wang et al., Findings of NAACL 2024) | Self-Inspiring planning: at each step the agent reviews *all* previously explored states rather than a single trajectory | `experiment_history.jsonl` and `tree_snapshot.json` as the full evidence log; `memory.build_memory_summary()` compacting it into promising / uncertain / negative / failed buckets; `critic.review()` verdicts; `proposals.build_planner_prompt()` passing that summary plus the remaining search space to the Planner | Conceptually aligned. We keep the "review prior states before planning" principle but implement it as a deterministic structured summary, not as RecMind's prompting algorithm or tool use. |
| TAIRA (Yu et al., 2025) | Manager/Executor separation; reuse of both successful and corrected/failed trajectories | `controller.Controller` as manager (loop, budgets, convergence, duplicate rejection, best-checkpoint custody); `runner.py` / `isolated.py` / `worker.py` as executor in a separate process; `critic.review()` labelling each outcome; `memory.py` retaining failures as first-class evidence; the evidence-driven fallback in `proposals.DeterministicResearcher`, which turns recorded failures into hard blocks and soft preferences via `memory.evidence_directions()` | Inspired by. The manager/executor split is real. Thought Pattern Distillation is **not** implemented exactly, or even approximately — see below. |
| SAPIENT (Du et al., NAACL 2025) | Strategic tree-based planning instead of greedy action selection | `tree.TreeSearchPolicy` and `tree.TreePolicyConfig`: a best-first frontier keeping the strongest node from each of up to three active branches | Inspired by the motivation only. **Our TreeSearchPolicy is not MCTS.** See the explicit disclaimer below. |
| STARec (Wu et al., CIKM 2025) | Autonomous deliberate reasoning; separating fast response from slow reasoning | The Critic's forced separation of observation from interpretation, with an explicit confidence level and a concrete `next_test`, so a proposal must be justified against measured validation evidence | Conceptually aligned at the level of "make the reasoning step explicit and auditable". Anchored reinforcement training and user-agent slow thinking are **not** implemented. |

## What we implemented

**Full-history evidence, not last-step memory.** Every iteration appends a complete record
to `experiment_history.jsonl`: hypothesis, applied changes, resulting config, validation
metrics, decision, critique, parent lineage, parent-selection score breakdown, errors, and
token usage. `build_memory_summary()` derives a compact planner view from that log —
baseline reference, best observed, per-verdict lesson buckets, and hashed signatures of
already-tried configurations. The Planner sees this summary on every call, so planning is
conditioned on the whole search so far rather than the previous step.

**Manager/executor separation.** The Controller never trains anything itself. It selects a
parent, obtains a proposal, validates and applies the changes, and hands an immutable
config to the Runner, which trains and evaluates in an isolated child process under a
900-second timeout. A crashed or hanging experiment is recorded as evidence and the loop
continues. The Planner returns JSON describing an experiment; it is never given permission
to execute code or edit the repository.

**Failures are reused, not discarded.** A failed run produces a `failed` verdict, is kept in
the memory summary's `failed` bucket with its error type, and is surfaced to the Planner.
The tree policy independently penalizes parents whose children failed
(`failed_child_penalty`) and parents the Critic rejected (`rejected_node_penalty`), so a
bad direction loses priority through two separate paths.

**A grounded Critic.** `critic.review()` returns a structured verdict — `promote`, `noise`,
`reject`, or `failed` — alongside a measured `observation`, a separate `interpretation`, a
`confidence` level, and a `next_test` string. It compares against the parent using the
convergence epsilon of `0.002`, and explicitly labels single-seed gains as observations
rather than significance tests. It sees validation metrics only.

**A lightweight multi-branch tree.** `TreeSearchPolicy` groups successful nodes into
branches (`baseline`, `features`, `model`, `ranking_objective`, `optimization`), keeps the
strongest node per branch, and expands at most three branches. Parent priority is a single
additive score on the same scale as validation Primary:

```text
priority = validation Primary
         + exploration bonus        (visit-count based)
         + branch novelty bonus
         - runtime penalty
         - repetition penalty
         - failed-child penalty
         - rejected-node penalty
```

This preserves competing hypotheses instead of collapsing onto the current leader, while
costing no extra training runs.

**Determinism and recovery.** The LLM Planner retries up to three times with a repair
prompt, and any remaining failure falls back to `DeterministicResearcher`, which walks a
fixed candidate order and raises `StopIteration` when the space is exhausted. Identical
history therefore always yields an identical fallback proposal, and an API outage costs no
iterations. Duplicate configurations are blocked by canonical `experiment_key()` hashing
before execution.

**A fallback that still reads the evidence.** Losing the LLM does not mean losing what the
run has learned. `memory.evidence_directions()` derives an `EvidenceDirections` summary from
structured history fields only — `critique.verdict`, `error.type`, `error.message`, `model`,
`training_objective`, and `changes` — never from free-form model prose. It separates two
kinds of evidence:

- **Hard blocks** are directions that cannot run: a structurally unavailable model, a
  missing dependency, an unsupported model/configuration combination, or an exact duplicate.
- **Soft disfavored evidence** is a preference, not a rule: a model that failed repeatedly
  for generic reasons such as timeouts, or a mechanism the Critic rejected.

Selection runs the same candidate order twice. The first pass avoids both categories. If
nothing survives, the second pass relaxes only the soft evidence, while hard blocks and
duplicates remain enforced. If every remaining candidate is hard-blocked or duplicated, the
fallback raises `StopIteration` rather than knowingly launching a model that cannot run.
When evidence causes a higher-priority direction to be skipped, the reason is appended to
the proposal's `reason` string so the decision stays auditable. This is the closest our
system comes to TAIRA's reuse of failed trajectories: failures change what the planner is
allowed to propose, not merely what it is told.

## What we deliberately did not implement

**TAIRA's Thought Pattern Distillation.** TPD distills reusable, structured thought
patterns (task description, solution description, thought template) from successful runs,
corrected failures, and human expert insight, then retrieves the most similar pattern for a
new query. We implement none of that. Our memory is a fixed-schema summary of concrete
experiment records with no distillation step, no similarity retrieval, and no expert
correction loop. We reuse *outcomes*; TAIRA reuses *abstracted reasoning patterns*.

**SAPIENT's MCTS.** Our TreeSearchPolicy is explicitly **not** Monte Carlo Tree Search. It
has no rollouts, no simulation phase, no value backpropagation up the tree, and no
self-training loop between planner and agent. It is a one-step best-first scoring function
over a frontier of at most three nodes. The exploration term is a visit-count bonus
inspired by UCT-style balancing, but it is applied once at selection time and is never used
to propagate estimated values. Calling it MCTS would misrepresent it.

**STARec's anchored reinforcement training and user-agent slow thinking.** We considered
both and rejected them. Anchored reinforcement training is a two-stage paradigm combining
knowledge distillation from stronger reasoning models with preference-aligned reward
shaping; user-agent slow thinking models each *user* as an agent with parallel fast and
slow cognition. Neither matches our problem: we have no user-simulation loop to reason
over, and our agent's decisions are about experiment configurations, not user preferences.
Both would also require a training corpus of reasoning traces and a reward model, which
exceeds this project's compute and time budget by a wide margin.

**RecMind's tool use and zero-shot recommendation.** RecMind is an agent that produces
recommendations using external knowledge and tools. Our agent produces experiments. We took
the self-inspiring planning principle and left the rest.

## Why this design fits the 50-iteration and six-hour limits

The binding constraint is that our unit of search is a real training run. `configs/project.json`
sets `max_iterations` to 50, `max_wall_clock_hours` to 6, and `experiment_timeout_seconds`
to 900. In the worst case where every experiment runs to its timeout, six hours admits only
about 24 experiments, so the Controller reserves one full timeout before starting each
iteration and stops early rather than beginning a run it cannot finish.

That budget rules out MCTS on arithmetic alone. MCTS derives its advantage from many cheap
simulations per decision; here every simulation would be a real model training. At the
900-second per-experiment ceiling, ten training rollouts could consume up to 2.5 hours for
one planning decision. Two such decisions could consume up to 5 hours before accounting for
planning, evaluation, and finalization overhead, making full MCTS impractical within the
six-hour budget. A best-first policy spends zero training runs on planning and puts the
whole budget into experiments that produce real validation evidence.

The same reasoning applies to the training-based methods. Anchored reinforcement training
would need a corpus of reasoning traces and a reward model before the first useful
experiment; with at most 50 experiments available in total, there is no data regime in
which that pays back.

Three further choices protect the budget directly. Capping the frontier at three active
branches bounds parent-selection cost and stops the agent from thinly spreading its
experiments. Convergence stops the run after three consecutive flat expansions of the
global-best node — weak-parent exploration is deliberately neutral, so exploring a side
branch never falsely triggers a stop. And the deterministic fallback guarantees an LLM
outage degrades planning quality rather than burning iterations on retries.

## Evaluation discipline

Every point above operates on validation metrics only. Iterations expose validation GAUC,
nDCG@5, and their mean; the Planner, the Critic, the memory summary, and the tree policy
never receive test-split metrics. Test evaluation happens once, after research concludes,
on the validation-best checkpoint, and its results are never fed back. All ranking metrics
come from the unmodified organizer evaluator in `kuairand-starter-kit/`, verified against a
pinned SHA-256 digest.

## References

- Yancheng Wang, Ziyan Jiang, Zheng Chen, Fan Yang, Yingxue Zhou, Eunah Cho, Xing Fan,
  Yanbin Lu, Xiaojiang Huang, and Yingzhen Yang. *RecMind: Large Language Model Powered
  Agent For Recommendation.* Findings of the ACL: NAACL 2024, pages 4351–4364.
  <https://aclanthology.org/2024.findings-naacl.271/>
- Haocheng Yu, Yaxiong Wu, Hao Wang, Wei Guo, Yong Liu, Yawen Li, Yuyang Ye, Junping Du,
  and Enhong Chen. *Thought-Augmented Planning for LLM-Powered Interactive Recommender
  Agent* (TAIRA). arXiv:2506.23485, 2025. <https://arxiv.org/abs/2506.23485>
  — code: <https://github.com/USTC-StarTeam/TAIRA>
- Hanwen Du, Bo Peng, and Xia Ning. *SAPIENT: Mastering Multi-turn Conversational
  Recommendation with Strategic Planning and Monte Carlo Tree Search.* NAACL 2025 (Volume 1:
  Long Papers), pages 2629–2648. <https://aclanthology.org/2025.naacl-long.133/>
- Chenghao Wu, Ruiyang Ren, Junjie Zhang, Ruirui Wang, Zhongrui Ma, Qi Ye, and
  Wayne Xin Zhao. *STARec: An Efficient Agent Framework for Recommender Systems via
  Autonomous Deliberate Reasoning.* CIKM 2025. arXiv:2508.18812.
  <https://arxiv.org/abs/2508.18812>

Related internal documents: [architecture](architecture.md).
