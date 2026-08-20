#!/usr/bin/env python3
"""
check_manuscript.py
===================
Cross-checks the numbers written in the manuscript against the audited values.

Run audit_numbers.py first; this consumes its CSV.

Two directions, because they catch different failures:

  COVERAGE  For each audited quantity, does the value appear anywhere in the
            .tex? A value that never appears is not an error, but a value that
            appears in a DIFFERENT form than audited usually is -- that is how a
            sinusoid entry ends up where a phase entry belongs.

  CLAIMS    For each percentage-like number in the .tex, is there an audited row
            within tolerance? Numbers with no match are listed for inspection.
            Many will be legitimate (derived quantities, ranges, round numbers),
            so this is a worklist, not a verdict.

PRECEDENCE
----------
Where two sources report the same quantity, this script applies the rulings we
have established rather than treating them as equal:

  * phasecorr_*.npy is authoritative for correction efficacies. The
    improvement_pct field of pipeline_summary_*.npy comes from an older inline
    computation and is deprecated as a source.
  * trim=0 is the correct seed-ensemble variant; trim=10 rows are excluded.
  * the calibration window is 150 steps unless a passage is explicitly about the
    calibration-length study, so rows with other ncal are set aside by default.

Usage
-----
    python3 check_manuscript.py --csv audit_numbers.csv --tex <DIR>
    python3 check_manuscript.py --csv audit_numbers.csv --tex <DIR> --all-ncal
    python3 check_manuscript.py --csv audit_numbers.csv --tex <DIR> --tol 0.5
"""

import os, re, csv, glob, argparse
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="audit_numbers.csv")
ap.add_argument("--tex", required=True, help="directory holding the .tex files")
ap.add_argument("--tol", type=float, default=0.15,
                help="absolute tolerance when matching a printed number to an "
                     "audited one (in the printed units)")
ap.add_argument("--ncal", type=int, default=150,
                help="calibration window the manuscript reports")
ap.add_argument("--all-ncal", action="store_true",
                help="do not restrict to the manuscript's calibration window")
ap.add_argument("--show-matched", action="store_true")
args = ap.parse_args()

# ── load the audit ──────────────────────────────────────────────────────────
if not os.path.exists(args.csv):
    raise SystemExit(f"audit CSV not found: {args.csv}  (run audit_numbers.py)")

rows = []
with open(args.csv) as f:
    for r in csv.DictReader(f):
        try:
            r["value"] = float(r["value"])
        except ValueError:
            continue
        rows.append(r)

DEPRECATED_NOTE = "run_pipeline improvement_pct is the SINUSOID correction"

def keep(r):
    # deprecated source: superseded by phasecorr
    if DEPRECATED_NOTE in (r.get("note") or ""):
        return False
    # the mislabelled DMD key is retained in the CSV for the record, not for matching
    if "MISLABELLED" in r["quantity"]:
        return False
    # wrong seed-ensemble variant
    if (r.get("trim") or "") not in ("", "0"):
        return False
    # calibration window
    if not args.all_ncal and (r.get("ncal") or "") not in ("", str(args.ncal)):
        return False
    return True

live = [r for r in rows if keep(r)]
dropped = len(rows) - len(live)
print("=" * 78)
print(f"MANUSCRIPT CROSS-CHECK")
print("=" * 78)
print(f"  audited values      : {len(rows)}")
print(f"  in scope after rulings: {len(live)}   ({dropped} set aside: deprecated "
      f"source, trim!=0, or ncal!={args.ncal})")

# ── load the manuscript ─────────────────────────────────────────────────────
tex_files = sorted(glob.glob(os.path.join(os.path.expanduser(args.tex), "*.tex")))
if not tex_files:
    raise SystemExit(f"no .tex files in {args.tex}")
text = {}
for p in tex_files:
    text[os.path.basename(p)] = open(p, encoding="utf-8", errors="replace").read()
print(f"  manuscript files    : {len(tex_files)} "
      f"({', '.join(sorted(text)[:6])}{'...' if len(text) > 6 else ''})")

# strip comments so commented-out numbers do not count
def strip_comments(s):
    return re.sub(r"(?<!\\)%.*", "", s)

body = {k: strip_comments(v) for k, v in text.items()}
alltex = "\n".join(body.values())

# ── direction 1: coverage ───────────────────────────────────────────────────
# a printed number may be rounded; look for the value at 0, 1 and 2 decimals
def printed_forms(v):
    out = set()
    for nd in (0, 1, 2, 3):
        s = f"{abs(v):.{nd}f}"
        if s.endswith(".0") or s.endswith(".00") or s.endswith(".000"):
            s = s.rstrip("0").rstrip(".")
        out.add(s)
    return {s for s in out if s and s != "0"}

def appears(v):
    for s in printed_forms(v):
        # word-boundary-ish: not preceded/followed by another digit or a dot-digit
        if re.search(r"(?<![\d.])" + re.escape(s) + r"(?![\d])", alltex):
            return True
    return False

by_q = defaultdict(list)
for r in live:
    by_q[r["quantity"]].append(r)

print("\n" + "-" * 78)
print("COVERAGE — audited values not found anywhere in the text")
print("-" * 78)
missing = []
for q in sorted(by_q):
    for r in by_q[q]:
        if not appears(r["value"]):
            missing.append(r)
if missing:
    print(f"  {len(missing)} of {len(live)} audited values do not appear:\n")
    cur = None
    for r in sorted(missing, key=lambda r: (r["quantity"], r["case"])):
        if r["quantity"] != cur:
            cur = r["quantity"]; print(f"  {cur}")
        print(f"      {r['value']:>12.4f}   {r['case']:<28} {r['source']}")
    print("\n  Not necessarily wrong: many audited quantities are intermediate.")
    print("  Worth checking any whose sibling values DO appear -- a partially")
    print("  reported group is where a wrong column tends to hide.")
else:
    print("  every audited value appears somewhere in the text")

# ── direction 2: claims ─────────────────────────────────────────────────────
# percentages and decimal quantities that look like reported results
NUM = re.compile(r"(?<![\d.])(-?\d+\.\d+|\d{1,3})\s*\\?%")
DEC = re.compile(r"\$(-?\d+\.\d{2,4})\$")

# numbers that are structural rather than data
IGNORE_EXACT = {"100", "0", "1", "2", "3", "4", "5", "10", "20", "25", "50",
                "52", "8", "40", "60", "70", "90", "95", "99"}

# ranges written "$72$--$83\\%$" print two numbers that are endpoints of a set of
# audited values, not values in their own right. Collect them separately.
RANGE = re.compile(r"\$(-?\d+(?:\.\d+)?)\$\s*--\s*\$(-?\d+(?:\.\d+)?)(?:\\?%)?\$")
range_tokens = set()
for _fn, _s in body.items():
    for m in RANGE.finditer(_s):
        range_tokens.add(m.group(1)); range_tokens.add(m.group(2))

claims = []
for fn, s in body.items():
    for m in NUM.finditer(s):
        tok = m.group(1)
        if tok in IGNORE_EXACT or tok in range_tokens:
            continue
        ctx = re.sub(r"\s+", " ", s[max(0, m.start() - 70):m.end() + 30]).strip()
        claims.append((fn, float(tok), tok + "%", ctx))
    for m in DEC.finditer(s):
        ctx = re.sub(r"\s+", " ", s[max(0, m.start() - 70):m.end() + 30]).strip()
        claims.append((fn, float(m.group(1)), m.group(1), ctx))

vals = [r["value"] for r in live]

def matched(v):
    return any(abs(v - w) <= args.tol for w in vals)

unmatched = [c for c in claims if not matched(c[1])]
print("\n" + "-" * 78)
print(f"CLAIMS — printed numbers with no audited value within {args.tol}")
print("-" * 78)
print(f"  {len(claims) - len(unmatched)} of {len(claims)} printed numbers matched an audited value")
print(f"  ({len(range_tokens)} range endpoints set aside: they bound a set of audited\n"
      f"   values rather than naming one)\n")
if unmatched:
    cur = None
    for fn, v, tok, ctx in sorted(unmatched, key=lambda c: (c[0], c[1])):
        if fn != cur:
            cur = fn; print(f"  --- {fn} ---")
        print(f"    {tok:>9}   ...{ctx[-88:]}")
    print("\n  Expect many benign entries here: ranges, derived quantities,")
    print("  percentages of percentages, and numbers from sources this audit does")
    print("  not read. Treat it as a worklist. A number that should be traceable")
    print("  and is not is the finding.")

if args.show_matched:
    print("\n" + "-" * 78)
    print("MATCHED CLAIMS")
    print("-" * 78)
    for fn, v, tok, ctx in sorted(set(claims) - set(unmatched), key=lambda c: (c[0], c[1])):
        print(f"    {tok:>9}   ...{ctx[-80:]}")

print(f"""
{'-' * 78}
Tolerance is {args.tol} in printed units, so 82.8 matches an audited 82.79 but not
84.17. Raise it with --tol if rounding is coarser than that; lower it to find
values that agree loosely but not exactly.
""")
