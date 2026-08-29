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
