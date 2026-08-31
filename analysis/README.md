# Leakage-safe EDA

`01_eda.ipynb` is the exploratory notebook. Do not use its full `log_late`
statistics for model selection because that file also contains the official test
period beginning on 2022-04-29.

Run the development-only audit with:

```bash
python analysis/leakage_safe_eda.py
```

The script uses all early-standard rows, but discards rows after 2022-04-28 from
both late-standard and random-exposure data before reading their behavior labels.

Candidate-history coverage and feature-only ranking audit:

```bash
python analysis/candidate_history_audit.py
```

This audit also excludes every row after 2022-04-28. It showed that exact prior
positive video history covers only 0.0304% of the official validation rows, so
added-field placebo controls are required before attributing FM gains to the signal.

Conditional model diagnostics use strictly earlier target-free history lengths and
fixed cold/medium/rich, popularity, duration, and time slices:

```bash
python scripts/analyze_conditional_complementarity.py
python scripts/evaluate_history_gated_ensemble.py
```

Slice improvements generate hypotheses only. A model or rule gate is accepted only
after evaluating the complete within-user ranking and rolling folds.
