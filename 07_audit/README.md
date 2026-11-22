# 07 — Audit

Provenance tracking and manuscript cross-checking.

| script | does |
|---|---|
| `audit_numbers.py` | walks every output file and writes a CSV of each reported quantity with its provenance — case, latent dimension, calibration length, timestep, trim setting, source file — flagging quantities whose sources disagree |
| `audit_numbers_v3.py` | `audit_numbers.py` plus the revision outputs as sources and a claims check of every number quoted in the revised text against its data source (reads `sensitivity_table_v2.json`) |
| `check_manuscript.py` | cross-checks every number printed in the LaTeX source against that CSV, in both directions: coverage, and unmatched claims |

```bash
python3 audit_numbers.py --data-dir .
python3 audit_numbers_v3.py --data-dir . --quiet     # revised manuscript
python3 check_manuscript.py --csv audit_numbers.csv --tex <bundle-dir>
```

Some flagged disagreements are expected. The calibration-length sweep stores the
same quantity at six different window lengths, and these group together as an
apparent conflict; read the `ncal` column before drawing a conclusion.

Expect a long unmatched list from `check_manuscript.py`. Range endpoints, derived
percentages and numbers from sources the audit does not read all appear there
legitimately. **A number that should be traceable and is not is the finding.**

This tooling caught errors that manual checking missed, including a figure
plotting one quantity while labelled another, a table assembled from two
incompatible parameter sweeps, and a drift rate that survived a manual
search-and-replace because the search terms did not include it.
