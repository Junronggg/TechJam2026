# Autonomous ML Research Agent for Recommender Systems

## Project overview

We are building an autonomous machine-learning research agent for recommender
systems. The agent improves a recommendation model through repeated, validation-only
experiments instead of requiring a person to manually choose every model, feature,
or hyperparameter.

The system is designed for TikTok-style short-video recommendation, where the goal
is to rank videos that a user is likely to watch for a long time. It uses the
KuaiRand-Pure benchmark and reports the official GAUC and nDCG@5 metrics.

## How our solution addresses the problem

The agent combines a Planner, a metric-grounded Critic, persistent research memory,
and a progressive tree search:

1. A leakage-safe data profile identifies useful patterns, weak user or content
   segments, metric trade-offs, and runtime bottlenecks.
2. The Planner diagnoses the current result and proposes one legal experiment from
   a compatible candidate space. It can select a registered model or submit a
   compact, validated code branch for a new model or feature transformation.
3. The Controller executes the experiment in an isolated worker using only the
   training and validation splits.
4. The Critic compares GAUC, nDCG@5, and the Primary score against the global
   validation incumbent, records the hypothesis outcome, and identifies the next
   research direction.
5. The tree-search policy balances exploitation of promising branches with
   exploration of novel model families. Successful improvements receive positive
   reward; weak or failed experiments remain evidence but are not treated as
   successes.
6. The process continues until the experiment or time budget is exhausted or the
   search converges.

The LLM chooses and explains experiments, while deterministic code controls
execution, validation, promotion, rollback, budget enforcement, and the test-data
firewall. If the LLM is unavailable, the same interface uses a deterministic
researcher as an explicit fallback.

## Models explored

The experiment registry supports several complementary recommender families:

- Linear and factorization-machine baselines
- Field-aware factorization machines (FFM)
- FM ensembles
- LightGBM and grouped LambdaRank
- DeepFM
- Categorical and dense DCNv2
- Two-tower candidate/history models
- Sequential FM and FPMC
- SASRec and metadata-aware SASRec
- Candidate-aware DIN
- LightGCN and a LightGCN/FM hybrid
- Ranking-aware multitask learning
- A heterogeneous DCNv2/FM/tree blend

The open-ended mode also allows the Planner to propose a small new model or feature
module. Generated code must pass static validation and implement the repository's
`fit_validate` and `finalize` contracts before it can be executed.

## Feature engineering

Features are created from fields and interaction history that are available before
the prediction event. The current feature space includes:

- User, video, author, tab, and duration representations
- Train-fitted fine-grained duration buckets
- Raw video tags from the training vocabulary
- Request hour and weekday
- Upload age and freshness signals
- User activity statistics
- Video type
- Time-decayed item, author, and tag popularity
- Hierarchical user-tag and user-author affinity
- Recent-history-to-candidate similarity
- Leakage-safe user, item, tag, author, and user-author long-view statistics
- Candidate-aware sequential features for DIN and metadata-aware SASRec

All target statistics and sequential features are fitted using training data only.
Validation and test labels never update these features. Encoded data, feature
matrices, histories, and sequential representations are fingerprinted and cached to
avoid repeatedly processing the full 1.1-million-row training set.

## Evaluation and research integrity

The official KuaiRand evaluator is kept unchanged and verified by SHA-256 before
experiments run. Research decisions use the validation split only. The official
Primary metric is:

```text
Primary = mean(GAUC, nDCG@5)
```

The final test prediction is generated only after the validation-selected model is
frozen. Test metrics are never returned to the Planner and are not used to choose
experiments.

Our strongest completed validation run currently records:

```text
GAUC:    0.6732213
nDCG@5:  0.5385662
Primary: 0.6058938
```

These are validation results, not hidden-test results. The competition organizers
perform the hidden evaluation after submission.

The explicit final evaluation of the frozen validation winner on the local test split
returned GAUC `0.6665823`, nDCG@5 `0.5314327`, and Primary `0.5990075` across 170,588
rows and 23,875 users. Test metrics were computed after research and were not exposed
to the Planner; the hidden competition score is determined by the organizers.

## Development tools

- Visual Studio Code for implementation and inspection
- Windows PowerShell for environment setup and experiment execution
- Python 3.11 virtual environment
- Git and GitHub for version control and collaboration
- JSON/JSONL run logs for reproducible experiment auditing

## APIs used

- OpenRouter's OpenAI-compatible API for LLM-guided experiment planning
- The `/api/v1/chat/completions` endpoint with structured JSON proposals

The API key is loaded from an environment variable and is never stored in the
repository or in run artifacts. The system remains runnable without the API through
its deterministic researcher and fallback path.

## Libraries and frameworks

- PyTorch for neural recommender models and training
- LightGBM for tree-based ranking models
- NumPy for numerical computation and feature processing
- scikit-learn for supporting preprocessing and utilities
- Python standard library for the controller, HTTP client, logging, validation, and
  isolated experiment execution

## Datasets and assets

- KuaiRand-Pure, downloaded from the official Zenodo distribution
- The organizer-provided KuaiRand starter kit and evaluator
- KuaiRand user, video, and interaction-log files supplied with the benchmark

No hidden test data or hidden labels are included in the repository or used during
research.

## Reproducibility

Every experiment records its hypothesis, evidence, parent branch, configuration or
code diff, validation metrics, errors, recovery actions, runtime, and promotion
decision. The run directory also contains the research trajectory, tree snapshot,
LLM request audit, and the run's best checkpoint. This makes the agent's decisions
inspectable from the first baseline reproduction through the final validation winner.
