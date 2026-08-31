# Run & Iteration Log

This is the consolidated research log for the KuaiRand recommendation system.
The entries are ordered by experiment sequence, not by calendar time. Date, clock
time, and runtime are intentionally omitted. The log includes the model, loss,
feature, validation, robustness, and ensemble experiments recorded in `TRY.md`.

All ordinary scores below are **validation** scores. The official research metric is

```text
Primary = (GAUC + nDCG@5) / 2
```

The validation split contains `124,909` rows. Test labels were not used to choose
any research candidate. A final test evaluation was run once after research only.
`KEEP` means retain as a candidate, not automatically claim that the result is a
statistically confirmed generalization improvement.

## Iteration 0 — FM/BCE official baseline

**Hypothesis.** Reproduce the official FM baseline before changing the model,
objective, or features.

**Change.** No code or feature change. FM with the original categorical fields,
embedding dimension `16`, BCE, learning rate `0.001`, batch size `8192`, seed `0`.

**Result.** GAUC `0.667133`, nDCG@5 `0.535806`, Primary **`0.601470`**.

**Finding / decision.** The evaluator and data pipeline are reproducible. This is
the reference for every later delta. `REFERENCE`.

**Error/recovery.** None.

## Iteration 1 — FM pairwise BPR and learning-rate ablation

**Hypothesis.** GAUC and nDCG@5 rank items within each user, so pairwise BPR should
fit the evaluation objective better than pointwise BCE. A lower learning rate may
make the pairwise optimization more stable.

**Changes.** Reused the FM runner, same features and split; changed only the loss
and learning rate.

| Configuration | Primary | Delta vs. BCE | Decision |
|---|---:|---:|---|
| FM+BCE, lr `0.001` | 0.601470 | — | Reference |
| FM+BPR, lr `0.001` | 0.603396 | +0.001926 | Keep |
| FM+BPR, lr `0.0005` | 0.603696 | +0.002226 | Keep |
| FM+BPR, lr `0.0003` | **0.603963** | **+0.002493** | Keep; seed-0 reference |

Paired seeds `0–3` gave an average BPR-over-BCE improvement of `+0.002344`.

**Finding.** BPR is the first change that improves both ranking metrics by a clear
amount. The main gain is explained by objective alignment, not only by one seed.

**Error/recovery.** None.

## Iteration 2 — More negatives, semi-hard negatives, and hybrid loss

**Hypothesis.** More negatives per positive or semi-hard negatives might provide a
stronger ranking signal. A BCE+BPR mixture might combine calibration and ranking.

**Changes.** Kept FM+BPR and original features fixed; varied only the sampler or
loss mixture.

| Experiment | Primary | Delta vs. BPR lr `0.0003` | Decision |
|---|---:|---:|---|
| 2 negatives / positive | 0.603379 | -0.000584 | Reject |
| 4 negatives / positive | 0.602794 | -0.001169 | Reject |
| Semi-hard pool `2` | 0.601855 | -0.002108 | Reject |
| Semi-hard pool `4` | 0.587747 | -0.016216 | Reject |
| BCE+BPR hybrid, best weight in `0.25/0.5/0.75` | 0.603962 | -0.000001 | Reject |

**Finding.** More pairs increased training work without adding useful information.
The current semi-hard definition introduces noisy or excessively difficult pairs.
The hybrid objective does not improve over pure BPR.

**Error/recovery.** None. The sampler and hybrid branches completed normally.

## Iteration 3 — Global historical rates and explicit user/context crosses

**Hypothesis.** A user's overall long-view tendency, an item's historical quality,
or a user-specific preference for a tab/author may add information to FM.

**Changes.** Added leakage-safe train-only statistics or categorical crosses. For
training rows, leave-one-out statistics were used where applicable; validation used
train-only statistics.

| Experiment | Primary | Decision | Interpretation |
|---|---:|---|---|
| `user_long_view_rate` bucket | 0.600448 | Reject | Global average too coarse |
| `item_long_view_rate` bucket | 0.591682 | Reject | Large degradation |
| `user×tab_long_view_rate` cross | 0.602869 | Reject | Sparse/noisy cross |
| `user×author_long_view_rate` cross | 0.602180 | Reject | Sparse/noisy cross |
| Both crosses together | 0.601198 | Reject | Adding fields did not add complementarity |

**Finding.** The implementation is leakage-safe, but these aggregate or sparse
crosses do not provide reliable incremental ranking information.

**Error/recovery.** None.

## Iteration 4 — LightGBM representation check

**Hypothesis.** The historical statistics might be useful as continuous values even
if FM bucketization was a poor representation. LightGBM is a natural tabular check.

**Changes.** Trained LightGBM on original categorical encodings, then on continuous
rates and log-counts. FM evaluator and data split remained unchanged.

| Experiment | Primary | Delta vs. FM+BCE | Decision |
|---|---:|---:|---|
| LightGBM, original fields | 0.599817 | -0.001653 | Reject |
| LightGBM + continuous rates/counts | 0.590084 | -0.011386 | Reject |
| LightGBM + user×tab statistics | 0.597528 | -0.003942 | Reject |

**Finding.** The weak result is not only caused by FM's bucketization. These global
historical statistics are low-value for this time-split ranking task.

**Error/recovery.** None.

## Iteration 5 — DeepFM objective comparison

**Hypothesis.** Nonlinear feature interactions may add signal beyond FM. Compare
DeepFM+BCE and DeepFM+BPR while keeping the original fields fixed.

**Changes.** Enabled the existing DeepFM runner; compared objectives without adding
new target features.

| Model | Primary | Delta vs. FM+BCE | Decision |
|---|---:|---:|---|
| DeepFM+BCE | 0.603862 | +0.002392 | Keep |
| DeepFM+BPR | 0.603530 | +0.002060 | Reject vs. DeepFM+BCE |

**Finding.** DeepFM's nonlinear path helps, but the BPR gain observed in FM does not
transfer automatically to DeepFM. DeepFM+BCE remains a useful complementary model.

**Error/recovery.** None.

## Iteration 6 — Two-model ensemble weight search

**Hypothesis.** FM+BPR and DeepFM+BCE make different errors. A small blend may be
better than either model alone.

**Changes.** Trained the same two component models and changed only the DeepFM blend
weight.

| DeepFM weight | Ensemble Primary | Delta vs. FM+BPR | Decision |
|---:|---:|---:|---|
| 0.3 | 0.604562 | +0.000599 | Candidate |
| 0.4 | **0.604713** | **+0.000750** | Keep |
| 0.5 | 0.604203 | +0.000240 | Reject |

The weight `0.4` improved over FM+BPR in all three rolling folds; rolling mean
delta was `+0.001123`.

**Finding.** Model complementarity, rather than a stronger standalone model, is the
first reliable ensemble improvement.

**Error/recovery.** None.

## Iteration 7 — Three-day temporal counts

**Hypothesis.** Recent user activity and recent item exposure may reflect changing
interest and popularity better than all-history counts.

**Changes.** Added train-only `user_recent_3d_activity` and
`item_recent_3d_exposure` counts.

**Result.** The single official split reached **`0.605010`**, apparently above the
ensemble. Rolling validation improved only `1/3` folds, with mean delta **`-0.000246`**.

**Finding / decision.** The single-split increase is time-window dependent and does
not generalize across rolling folds. `REJECT`.

**Error/recovery.** None; rolling was used as the recovery/check against a misleading
single-split result.

## Iteration 8 — Multi-task auxiliary feedback

**Hypothesis.** Click, like, or completion behavior may provide auxiliary supervision
for the main `long_view` task.

**Changes.** Added an auxiliary training head while keeping long-view as the only
inference target.

| Auxiliary signal | Primary | Delta vs. DeepFM | Decision |
|---|---:|---:|---|
| Click only | 0.604034 | +0.000172 | Small; reject as final |
| Like only | **0.604400** | **+0.000538** | Keep standalone; rolling 3/3 |
| Completion only | 0.603876 | +0.000014 | Reject; label overlap |
| Click + like | 0.604259 | +0.000397 | Like-only is better |
| Click + like + completion | 0.604382 | +0.000520 | More complex, not better |

**Finding.** Like is the only auxiliary signal with a useful, repeatable indication.
Completion is too close to the long-view definition; click dilutes the like gain.

**Error/recovery.** None.

## Iteration 9 — Candidate-specific history features

**Hypothesis.** A user's previous positive interaction with the exact candidate video
or author should be more specific than a global user/item rate.

**Changes.** Added strict-past, train-only fields:
`prior_video_positive` and `author_positive_recency`; then tested them together.

| Feature configuration | Primary | Initial rolling | Decision |
|---|---:|---:|---|
| Prior video positive | 0.604205 | 3/3 | Reinterpret |
| Author positive recency | 0.604199 | 3/3 | Reinterpret |
| Both fields | 0.604169 | — | Reject |

Coverage was very low (about `0.0304%` for the video-positive field). A constant
field placebo reached `0.604394`, higher than either real history field. Later
target-free versions also failed:

| Target-free feature | Primary | Decision |
|---|---:|---|
| Prior same-video count | 0.603936 | Reject |
| Previous interaction same author | 0.604009 | Reject |

**Finding.** The temporal implementation is strict and leakage-safe, but the small
initial gains are better explained by adding an FM field than by behavioral history.
The placebo control prevents a false positive conclusion.

**Error/recovery.** No code error. The recovery was an automatic constant-field
control and coverage audit, which changed the interpretation to `REINTERPRET`.

## Iteration 10 — Low-rank DCNv2

**Hypothesis.** Explicit low-rank cross layers may learn useful interactions that
DeepFM does not represent as directly.

**Changes.** Enabled two cross layers with rank `16`; original fields and evaluator
were unchanged.

**Result.** Primary **`0.604164`**, `+0.000302` vs. DeepFM. Rolling improved `3/3`
folds with mean delta `+0.000248`.

**Finding / decision.** Positive standalone candidate, but not yet an ensemble
member. `KEEP` standalone.

**Error/recovery.** None.

## Iteration 11 — DCNv2 ensemble compatibility

**Hypothesis.** DCNv2 may add complementary errors to the existing FM/DeepFM blend.

**Changes.** Tested DCNv2 as a replacement for DeepFM, then as a third ensemble
member; no feature or evaluator changes.

| Ensemble | Primary | Rolling | Decision |
|---|---:|---:|---|
| FM+BPR + DCNv2 (replace DeepFM) | 0.604317 | 1/3 | Reject |
| FM+BPR + DeepFM + DCNv2 | 0.604616 | 0/3 | Reject |

DCNv2 and DeepFM prediction correlation was `0.9925–0.9963`, so they largely make
the same errors despite different model names.

**Finding.** A good standalone score does not imply ensemble value. Prediction
diversity is required before adding a model to the blend.

**Error/recovery.** None.

## Iteration 11A — Conditional error complementarity and history gate

**Hypothesis.** A candidate model may be useful only for particular user-history,
item-popularity, duration, or time slices. Overall Primary and prediction
correlation alone may hide that conditional value.

**Changes.** Added fixed slice evaluation and pair-error accounting for the current
champion versus Like-only Multi-task DeepFM and DCNv2. Then tested a pre-declared
rule: for users with history length `<=2`, blend `50%` champion with `50%` DCNv2;
use the champion elsewhere.

| Candidate vs. champion | Overall delta | Correlation | Best slice | Worst slice |
|---|---:|---:|---|---|
| Like-only Multi-task | -0.000313 | 0.9721 | Short video `+0.001340` | Cold history `-0.002673` |
| DCNv2 | -0.000549 | 0.9708 | Cold history `+0.009087` | Medium history `-0.002251` |

The DCNv2 cold-user gain covered only `1.30%` of validation rows. It recovered about
`7.24%` of champion error pairs but introduced about `3.56%` new errors; Like-only
recovered `7.36%` and introduced `3.59%`.

**Result.** The rule-based gate changed Primary by **-0.000017**; rolling was `2/3`
with mean delta `+0.000012`.

**Finding / decision.** Slice gains do not automatically become overall ranking
gains. `REJECT` as noise, while retaining slice/error analysis for future planning.

**Error/recovery.** No implementation error. The negative result prevented a
slice-specific model from being promoted based on one subgroup alone.

## Iteration 11B — Lightweight causal sequence model

**Hypothesis.** Strictly past interactions, item/author identity, behavior type,
time-gap buckets, and candidate-conditioned attention may add order information not
available to static FM fields.

**Changes.** Added a lightweight last-`16` sequence path with strict timestamp
causality, one attention layer, and the existing DeepFM scorer. Same-timestamp rows
do not read each other; validation labels are not inserted into history.

| Model | Primary | Delta vs. DeepFM | Decision |
|---|---:|---:|---|
| DeepFM+BCE control | 0.603862 | — | Reference |
| Lightweight sequence DeepFM | 0.603369 | -0.000493 | Reject |

The sequence prediction correlation with the champion was `0.9646`; it recovered
`8.96%` of champion error pairs but introduced `4.39%` new errors. Its best slice
was short video `+0.001460`, while tail items lost `-0.006721`. A fixed `90%`
champion + `10%` sequence blend changed Primary by `-0.000023`.

**Finding / decision.** The strict sequence implementation is reusable and
leakage-safe, but this architecture is weaker and less efficient than the static
models. `REJECT` for the current sequence path; do not continue only by scanning
sequence length or blend weights.

**Error/recovery.** An early validation-prediction batch-size memory issue was fixed
by capping prediction batches at the training batch size. The loss/metric trajectory
was unchanged after the fix, so it was a performance correction rather than a model
result change.

## Iteration 11C — Standard-exposure versus random-exposure robustness

**Hypothesis.** The ensemble improvement should remain useful if the exposure policy
changes, rather than only fitting the standard logging policy.

**Changes.** Evaluated the same FM+BPR and champion ensemble on a separate
random-exposure development split. No model or evaluator change was made.

| Exposure policy | Model | GAUC | nDCG@5 | Primary |
|---|---|---:|---:|---:|
| Standard | FM+BPR | 0.670593 | 0.537333 | 0.603963 |
| Standard | Champion ensemble | 0.671480 | 0.537945 | 0.604713 |
| Random | FM+BPR | 0.592296 | 0.147733 | 0.370014 |
| Random | Champion ensemble | 0.591785 | 0.147934 | 0.369860 |

**Finding / decision.** The ensemble delta is `+0.000750` on standard exposure but
`-0.000155` on random exposure. Absolute Primary values cannot be compared across
the two policies because candidate sets and positive rates differ. `KEEP` the
standard-split champion for the benchmark, but report that policy robustness is not
established; FM+BPR is the safer random-exposure fallback.

**Error/recovery.** None. The random split was treated as a robustness control, not
used to select the official validation winner.

## Iteration 11D — Pairwise multi-task objective

**Hypothesis.** Multi-task DeepFM may benefit from BPR on the long-view main task
while retaining like BCE as an auxiliary task.

**Changes.** Kept the like auxiliary head and changed only the long-view main loss
from BCE to same-user BPR. A second configuration changed embedding size and
auxiliary weight once; no broad parameter sweep was performed.

| Configuration | Primary | Delta vs. matched BCE | Decision |
|---|---:|---:|---|
| Like Multi-task + BCE, k16/aux0.1 | **0.604400** | — | Reference |
| Like Multi-task + BPR, k16/aux0.1 | 0.603322 | -0.001079 | Reject |
| Like Multi-task + BPR, k32/aux0.3 | 0.603610 | -0.000791 | Reject |

**Finding / decision.** The BPR improvement is specific to the FM branch and does not
transfer to this multi-task architecture. Mark the two tested configurations as
`STOP_DIRECTION` at their exact scope; do not generalize the result to every future
multi-task sampler or auxiliary target.

**Error/recovery.** None.

## Iteration 12 — Additional scalar history and watch-time regression

**Hypothesis.** Counts or capped/log watch time may be smoother and more useful than
binary history features.

**Changes and results.**

| Experiment | Primary | Delta vs. relevant baseline | Decision |
|---|---:|---:|---|
| Prior same-video count | 0.603936 | -0.000027 vs. FM+BPR | Reject |
| Previous same-author interaction | 0.604009 | +0.000046 vs. FM+BPR | Reject; noise-sized |
| Capped log-watch regression | 0.603891 | +0.000029 vs. DeepFM | Reject |

**Finding.** These scalar versions provide no reliable new information.

**Error/recovery.** None.

## Iteration 13 — Constant-field placebo becomes `global_context`

**Hypothesis.** A learned global context embedding may act as a useful bias/intercept
field even though the original behavioral feature failed its placebo test.

**Changes.** Formalized the constant-field branch as `global_context`, with no label
or future behavior in the field.

**Results.** FM+BPR plus `global_context` reached `0.604394` (`+0.000431`). Rolling
improved `3/3` folds with mean `+0.000810`. Paired seeds were positive for `3/4`,
mean delta `+0.000333`, but the approximate confidence interval crossed zero.
Replacing the FM component inside the champion gave `0.604674`, slightly below
`0.604713`.

**Finding / decision.** This is a promising but uncertain model-field effect, not
proof that candidate history works. Keep as a candidate; do not replace the robust
ensemble.

**Error/recovery.** The placebo result was not discarded; it was converted into a
separate, explicitly named hypothesis and then tested with rolling and paired seeds.

## Iteration 14 — One-sided censored watch-time objective

**Hypothesis.** A censored watch-time target may provide auxiliary information beyond
the capped/log regression, because completed videos reveal only a lower bound.

**Changes.** Implemented one-sided censored loss: incomplete views use observed
`log1p(play_time)`; completed views penalize predictions only below the duration
lower bound. Watch-time is training-only and never an inference feature.

| Configuration | Primary | Delta vs. DeepFM | Decision |
|---|---:|---:|---|
| DeepFM+BCE control | 0.603862 | — | Reference |
| BCE + censored-watch auxiliary | 0.603939 | +0.000077 | Insufficient |
| BPR + censored-watch auxiliary | 0.602523 | -0.001339 | Reject |

**Finding.** The censored objective is now genuinely tested, but its pointwise gain
is too small and its pairwise version declines. It cannot replace the champion.

**Error/recovery.** None. This was a controlled follow-up, not the earlier capped
watch-time approximation.

## Iteration 15 — Static video metadata

**Hypothesis.** Label-free, prediction-time metadata may add side information without
the leakage and sparsity problems of target-rate statistics.

**Changes and results.**

| Feature/model | Primary | Delta vs. FM+BPR | Decision |
|---|---:|---:|---|
| `video_music_type` + FM+BPR | 0.604077 | +0.000114 | Not champion |
| `video_tag_components` + FM+BPR | 0.604302 | +0.000339 | Candidate only |
| `video_tag_components` ensemble | 0.604415 | -0.000298 vs. champion | Reject |

**Finding.** Splitting multi-valued tags is better than the exact representation,
but it still does not provide ensemble complementarity.

**Error/recovery.** None.

## Iteration 16 — Rank calibration of the ensemble

**Hypothesis.** Because evaluation is within-user ranking, calibrating the DeepFM
component by user-level rank may improve the blend without changing either model.

**Changes.** Reused the cached FM+BPR and DeepFM+BCE predictions; changed only the
combination from two-sided z-score to `fm_zscore_deepfm_rank`.

| Calibration | DeepFM weight | Primary | Delta vs. original ensemble |
|---|---:|---:|---:|
| z-score + z-score | 0.40 | 0.604713 | — |
| FM z-score + DeepFM rank | 0.40 | 0.604746 | +0.000033 |
| FM z-score + DeepFM rank | 0.65 | **0.605291** | **+0.000578** |

**Finding.** The local validation peak improved without new features or a new model.
Rolling was `2/3` with mean `+0.000357`; paired seeds were `3/4` positive with an
interval crossing zero. Status: `UNCERTAIN / ELIGIBLE`, not statistically confirmed.

**Error/recovery.** None.

## Iteration 17 — Autonomous calibration follow-up

**Hypothesis.** After a positive calibration result, the planner should test a nearby
weight rather than stop or repeat a failed feature family.

**Change.** The controller automatically followed the rank-calibration result with
`ensemble_deepfm_weight=0.65`; no human selected this action.

**Result.** Primary **`0.605291`**, `+0.003821` vs. FM+BCE and `+0.000578` vs. the
original `0.604713` ensemble.

**Finding / decision.** This demonstrates autonomous local follow-up and expands the
submission candidate pool. It is still subject to the rolling and paired-seed caveat
from Iteration 16.

**Error/recovery.** None; manual interventions `0`.

## Iteration 18 — Strict prior-exposure and author-recency follow-up

**Hypothesis.** If exact candidate history carries signal, strict past exposure and
same-author recency should improve FM+BPR without using future labels.

**Results.** `prior_video_exposure` reached `0.604123` (`+0.000160` vs. FM+BPR);
`author_recency` reached `0.604005` (`+0.000042`). Neither approached the champion.

**Finding / decision.** These fields are correctly leakage-safe but have no reliable
incremental value on this dataset. `REJECT / RESEARCH_ONLY`.

**Error/recovery.** None.

## Iteration 19 — Censored watch-time head with FM+BPR

**Hypothesis.** A watch-time head may help an FM+BPR representation even when the
same auxiliary target was weak for DeepFM.

**Results.**

| Configuration | Primary | Delta vs. FM+BPR | Decision |
|---|---:|---:|---|
| FM+BPR + censored watch-time | 0.604108 | +0.000145 | Research-only |
| Plus prior exposure + author recency | 0.604222 | +0.000259 | Research-only |

An additional lambda `0.2` run reached `0.604201`; adding the two history fields gave
`0.604154`. The small gains did not exceed the ensemble or receive rolling/seed
confirmation.

**Finding.** The watch-time direction remains a possible research branch, but these
specific auxiliary configurations are not final candidates.

**Error/recovery.** None.

## Iteration 20 — Paired-seed confirmation of rank-calibrated weight `0.65`

**Hypothesis.** The `0.605291` validation candidate should be compared with the
original ensemble under identical seeds before being treated as robust.

**Change.** Same component checkpoints, data, and evaluator; only the calibration and
weight differed. Seeds `0–3` were paired.

| Seed | Reference | Candidate | Delta |
|---:|---:|---:|---:|
| 0 | 0.604713 | 0.605291 | +0.000578 |
| 1 | 0.604061 | 0.604734 | +0.000673 |
| 2 | 0.604362 | 0.604723 | +0.000362 |
| 3 | 0.604435 | 0.604433 | -0.000002 |

Mean delta was `+0.000403`; `3/4` seeds were positive and the approximate interval
was `[-0.000074, +0.000880]`.

**Finding / decision.** `UNCERTAIN / ELIGIBLE`: acceptable as a validation submission
candidate, but not a confirmed generalization improvement.

**Error/recovery.** None.

## Iteration 21 — Local rank-calibration weight scan and shadow validation

**Hypothesis.** A small, pre-bounded neighborhood around `0.65` may find a better
blend without retraining the component models.

**Changes.** Added only weights `0.63` and `0.64` to the cached prediction scan; the
three calibration directions were fixed before evaluation.

| Configuration | Primary | Evidence |
|---|---:|---|
| Rank-calibrated weight `0.63` | **0.605365** | New validation peak |
| Rank-calibrated weight `0.64` | 0.605352 | Near-peak |
| Weight `0.63`, paired seeds | Mean delta `+0.000430` | 4/4 positive; interval crosses 0 |
| Weight `0.63`, shadow windows | Mean delta `+0.000085` | 3/4 windows positive |

**Finding / decision.** `0.605365` is the highest single validation score, but the
shadow and paired-seed margins are noise-sized and one time window is negative. Keep
it as `UNCERTAIN / ELIGIBLE`; retain `0.604713` as the rolling 3/3 fallback.

**Error/recovery.** None. The shadow validation is a robustness control, not another
leaderboard score.

## Iteration 21A — Expanded action-space model smoke ablation

**Hypothesis.** A broader executable model space may reveal a model family whose
errors differ from FM/DeepFM, even if its standalone score is not competitive.

**Changes.** Added validation-only runners for LambdaRank, ADT, and LightGCN using
the original fields and the official evaluator. These were exploratory skills, not
hard-coded replacements for the planner. Test labels were not loaded.

| Model/action | GAUC | nDCG@5 | Primary | Decision |
|---|---:|---:|---:|---|
| LambdaRank | 0.662384 | 0.534602 | 0.598493 | Research-only |
| ADT | 0.616908 | 0.515149 | 0.566029 | Reject |
| LightGCN (initial smoke) | 0.491757 | 0.465515 | 0.478636 | Reject |
| LightGCN (fixed runner check) | 0.500850 | 0.469940 | 0.485395 | Reject |

**Finding.** LambdaRank is weaker alone than FM+BPR, but its prediction diversity
made it useful in the later cached heterogeneous ensemble. ADT and LightGCN were
far below the FM-family baselines and were not added to the final blend.

**Error/recovery.** The first LightGCN smoke result was followed by a fixed-runner
check; the score remained far below the reference, so this was classified as a
negative model-family result rather than a planner failure.

## Iteration 22 — Cached heterogeneous ensemble search

**Hypothesis.** A weaker standalone model can still help if its errors are genuinely
different. Search a small, fixed subset/normalization/weight space using cached
validation predictions rather than repeatedly retraining.

**Candidate components.** FM+BPR `0.603963`; DeepFM+BCE `0.603862`; like-only
multi-task `0.604400`; censored watch-time BCE `0.603939`; LambdaRank `0.598493`;
DCNv2 `0.604164`; lightweight sequence `0.603369`.

**Results.**

| Cached ensemble | Primary | Delta vs. `0.604713` | Decision |
|---|---:|---:|---|
| Like multitask + LambdaRank + DCNv2 | 0.605263 | +0.000550 | Research-only |
| FM watch-time + LambdaRank + DCNv2 | **0.605309** | **+0.000596** | Research-only |

LambdaRank's standalone score is low, but its correlation with the other components
was about `0.872–0.880`, lower than most FM/DeepFM-family correlations. This explains
why it can enter a diverse blend.

**Finding / decision.** Different information sources can beat same-family additions,
but these cached combinations have not passed rolling or paired-seed confirmation.
They do not replace the robust champion.

**Error/recovery.** None. The search used validation caches only and did not read
test labels.

## Iteration 23 — Latest autonomous controller run

This is the latest end-to-end agent trajectory, recorded separately from the larger
offline ablation chronology above. It confirms that the controller can select a
local follow-up on its own.

| Agent iteration | Action | Primary | Delta from previous |
|---:|---|---:|---:|
| 0 | FM+BCE baseline | 0.601470 | — |
| 1 | FM+BPR, lr `0.0003` | 0.603963 | +0.002493 |
| 2 | FM+BPR + DeepFM+BCE, weight `0.4` | 0.604713 | +0.000750 |
| 3 | FM z-score + DeepFM rank calibration | 0.604746 | +0.000033 |
| 4 | Same calibration, weight `0.6` | **0.605248** | +0.000503 |

**Hypothesis progression.** The planner first selected the proven BPR objective, then
the complementary DeepFM blend, then a score-calibration follow-up, and finally a
nearby blend-weight follow-up. No person selected an action during this trajectory.

**Code/configuration diffs.** All four candidate actions used existing runners; the
changes were configuration-only: BPR objective/lower learning rate, ensemble mode,
rank calibration, and weight `0.4 -> 0.6`.

**Errors/recovery.** All iteration records contain `error: null`. There was no retry,
timeout, deterministic fallback, or manual restart.

## Finalization after research

After the validation search converged, the controller evaluated the validation-best
checkpoint on the test split exactly once and wrote `submissions/final.csv`. This
evaluation was not available to the planner and did not change the selected config.

| GAUC | nDCG@5 | Primary | Rows |
|---:|---:|---:|---:|
| 0.666354 | 0.532377 | **0.599365** | 170,588 |

The final CSV has the required columns `row_id,user_id,video_id,score` and 170,588
prediction rows. The complete test result is recorded in
`artifacts/final_test_metrics.json`.

## Final research summary

| Reference | Primary | Interpretation |
|---|---:|---|
| FM+BCE baseline | 0.601470 | Official reference |
| FM+BPR, lr `0.0003` | 0.603963 | Main objective improvement |
| Rolling-stable ensemble, weight `0.4` | **0.604713** | Robust fallback; rolling 3/3 |
| Latest autonomous run, weight `0.6` | 0.605248 | Highest in latest autonomous run |
| Rank-calibrated local scan, weight `0.63` | **0.605365** | Highest single validation candidate; not confirmed |

The main lessons are: align the loss with the ranking metric; use heterogeneous
models rather than redundant fields; require rolling/paired-seed evidence for small
gains; and use placebo controls when a sparse feature appears to help.

## Manual-intervention summary

For the latest autonomous run, manual interventions were **0**. No human changed
code/configuration, chose an experiment, repaired a failed process, or restarted the
agent while it was running. The offline ablations were also executed as recorded
experiments without changing the official evaluator.

## Evidence and reproduction files

- Model/feature chronology: [`TRY.md`](TRY.md)
- Agent planning and autonomy chronology: [`AGENT-TRY.md`](AGENT-TRY.md)
- Final raw agent records: `logs/run_20260831T160804Z/iteration_000.json` through `iteration_004.json`
- Final agent summary: `logs/run_20260831T160804Z/summary.json`
- Final research trajectory: `logs/run_20260831T160804Z/research_trajectory.json`
- Final artifact manifest: `artifacts/best_manifest.json`
- Final submission: `submissions/final.csv`
- Typical model ablation commands are listed in the reproduction section of `TRY.md`.
