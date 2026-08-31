# Leakage-safe EDA and data profiling

Run the profile before starting an LLM research session:

```powershell
.\.venv\Scripts\python.exe -X utf8 .\scripts\profile_dataset.py
```

The command writes three ignored, reproducible artifacts under
`artifacts/data-profile/`:

- `profile.md`: short human-readable findings.
- `profile.json`: full segment, cold-start, coverage, and drift tables.
- `planner_context.json`: compact evidence supplied to the LLM Planner.

`scripts/run_agent.py --researcher llm` automatically loads the planner context
when it exists. A different generated profile can be selected with
`--data-profile PATH`.

## Research boundary

The profiler follows the organizer split exactly:

- Train: 2022-04-08 through 2022-04-21.
- Validation: 2022-04-22 through 2022-04-28.
- Test: 2022-04-29 through 2022-05-08.

Rows outside the requested split are rejected by date before `long_view` is
read. The profile therefore uses training and validation labels, but never test
labels. It does not open the random-exposure log or video-statistics aggregates
for analysis; those sources remain quarantined pending a competition-rules and
time-provenance audit.

Current-row outcomes are never proposed as features. This exclusion covers
`long_view`, play time, clicks, likes, follows, comments, forwards, hates, and
profile/comment engagement. Static request-time metadata and train-only or
strictly-prior histories are the safe feature sources.

## What the profile measures

- Row, user, item, author, tag, and music counts by split.
- Long-view prevalence by date, hour, weekday, tab, activity, video type, tag,
  duration, and upload age.
- Users eligible for GAUC and users contributing zero to nDCG because they have
  no positives.
- Validation cold-start rates for users, videos, authors, tags, and music.
- Train-to-validation coverage of user-author, user-tag, user-music, and
  user-tab pairs.
- Train/validation label drift and categorical total-variation distance.
- Metadata coverage and an explicit feature-source leakage audit.

The segment rates are descriptive associations, not proof that a feature will
improve within-user ranking. Each candidate direction still needs a controlled
validation experiment and, for label-derived features, leave-one-out or
strictly-prior construction.

## Current profile implications

The current local dataset profile shows:

- Validation long-view prevalence is about 2.33 percentage points below train.
- User-tab history covers about 93.3% of validation rows.
- User-tag history covers about 73.2% of validation rows.
- User-author and user-music histories each cover only about 3.4% of rows.
- Item/author/tag cold start is very small; unseen validation users affect about
  1.6% of rows.

This makes user-tag affinity, request-time context, user activity, video type,
and time-aware/recency-aware histories more promising next directions than
blindly stacking sparse author or music pair rates. These are hypotheses for
the research loop, not guaranteed gains.

Raw tag, past-only user-tag impression count, and past-only smoothed user-tag
long-view rate are now executable feature operators. Raw tag vocabulary is fitted
on training only with an UNK bucket. For training target statistics, each event sees
only earlier timestamps; equal-timestamp events cannot update one another. Validation
and test use aggregates fitted from training labels only.
