#!/usr/bin/env python3
"""
heldout_threshold.py
====================
Out-of-sample validation of the R^2 applicability threshold.

The paper fixes R^2 > 0.3 as the rule that decides whether the phase correction
should be applied, and reports the efficacy of the accepted subset. That threshold
was fixed on the single-run cases and the reported efficacy is therefore an
in-sample estimate. This script tests it out of sample, using only the per-seed
records already written by the seed sweeps -- no retraining.

Three independent tests, each answering a different objection:

  A. STRATIFIED RANDOM SPLIT (repeated)
     Fit the rule on half the networks, validate on the other half. Answers:
     "is the accepted-subset efficacy an artefact of having chosen the threshold
     on the same data?" Gives an out-of-sample efficacy with error bars.

  B. LEAVE-ONE-CONFIGURATION-OUT
     Fit on all configurations but one, validate on the held-out configuration.
     Answers the stronger question: "does a threshold calibrated on some flows
     transfer to a flow it has never seen?" -- including, if the square cylinder
     is present, transfer across geometry.

  C. THRESHOLD SWEEP
     Rather than assuming 0.3, sweep candidate thresholds on the fit half, select
     one by an explicit objective, and validate the SELECTED threshold out of
     sample. Answers: "was 0.3 even the right number, and would an honestly
     chosen threshold have landed near it?"

Metrics reported out of sample:
  * mean efficacy (I_pha) over accepted networks vs over all networks
  * FALSE ACCEPT rate: accepted but the correction made things worse (I_pha < 0).
    The paper's claim is that there are none; this measures it out of sample.
  * false reject rate: rejected although the correction would have helped well.
  * the same for the field metric G where field files are available.

Input files (whichever exist), from --data-dir:
    seed_correction_Re{RE}_L{NZ}_summary.json    (latent efficacy per seed)
    seed_field_Re{RE}_L{NZ}_summary.json         (field G per seed)

IMPORTANT: seed_field_* files carry a "trim" field. Only trim == 0 is correct
(the untrimmed phase regression). Files with trim != 0 are REJECTED by this
script, because a stale trim=10 sweep produces very different numbers and must
not be mixed in.

Usage
-----
    python3 heldout_threshold.py --dry-run
    python3 heldout_threshold.py --data-dir ~/cylinder_lstm_data
    python3 heldout_threshold.py --threshold 0.3 --n-splits 2000
"""

import os, sys, json, glob, argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", default="~/cylinder_lstm_data")
ap.add_argument("--threshold", type=float, default=0.3,
                help="the threshold as published, tested out of sample")
ap.add_argument("--n-splits", type=int, default=2000,
                help="number of random stratified splits for test A")
ap.add_argument("--sweep", type=float, nargs=3, default=[0.05, 0.60, 0.05],
                metavar=("LO", "HI", "STEP"), help="threshold grid for test C")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="heldout_threshold_results.json")
ap.add_argument("--dry-run", action="store_true")
args = ap.parse_args()

rng = np.random.default_rng(args.seed)
DATA = os.path.expanduser(args.data_dir)

# ────────────────────────────────────────────────────────────────────────────
#  Load per-seed records
# ────────────────────────────────────────────────────────────────────────────
def load_records(data_dir):
    """Walk data_dir (results live in per-run SUBFOLDERS) and collect per-seed
    records. Returns (records, skipped)."""
    recs, skipped = [], []

    # gather every json anywhere under data_dir
    candidates = []
    for root, dirs, files in os.walk(data_dir):
        for fn in files:
            if fn.endswith(".json"):
                candidates.append(os.path.join(root, fn))
    candidates.sort()

    for path in candidates:
        rel = os.path.relpath(path, data_dir)
        low = rel.lower()

        # -- only the seed sweeps we want -----------------------------------
        # folder/file names use both 'seed_corr' and 'seed_correction'
        is_field  = "seed_field" in low
        is_latent = ("seed_corr" in low) and not is_field
        if not (is_field or is_latent):
            continue

        # -- exclude dry runs (synthetic smoke tests) ------------------------
        if "_dry" in low or "/dry" in low:
            skipped.append((rel, "dry-run output, excluded")); continue

        # -- exclude the stale trim=10 sweeps by path as well as by field ----
        if "trim10" in low:
            skipped.append((rel, "trim10 sweep (only trim=0 is correct), excluded"))
            continue

        try:
            d = json.load(open(path))
        except Exception as e:
            skipped.append((rel, f"unreadable: {e}")); continue
        if not isinstance(d, dict) or "per_seed" not in d:
            continue                      # not a summary file; ignore quietly
        if not d["per_seed"]:
            skipped.append((rel, "empty per_seed")); continue

        cfg = f"Re{d.get('re','?')}_L{d.get('latent','?')}"

        if is_field:
            # the trim footgun: only trim == 0 is the correct sweep
            trim = d.get("trim", None)
            if trim is None:
                skipped.append((rel, "no 'trim' field -- cannot verify, REJECTED"))
                continue
            if int(trim) != 0:
                skipped.append((rel, f"trim={trim}, REJECTED")); continue

        for s in d["per_seed"]:
            if not isinstance(s, dict):
                continue
            if "r2_mean" not in s or "I_pha" not in s:
                continue
            recs.append(dict(config=cfg,
                             kind=("field" if is_field else "latent"),
                             seed=s.get("seed"),
                             r2=float(s["r2_mean"]), I=float(s["I_pha"]),
                             G=(float(s["G"]) if ("G" in s and s["G"] is not None) else None),
                             source=rel))

    # -- de-duplicate: the same (config, kind, seed) may appear in more than one
    #    summary file if a sweep was re-run. Keep the first, report the rest.
    seen, uniq, dups = set(), [], 0
    for r in recs:
        key = (r["config"], r["kind"], r["seed"])
        if key in seen:
            dups += 1; continue
        seen.add(key); uniq.append(r)
    if dups:
        skipped.append(("(de-duplication)", f"{dups} duplicate (config,kind,seed) records dropped"))
    return uniq, skipped


def synthetic_records(rng):
    """Dry-run data with a genuine r2/efficacy relationship plus noise."""
    recs = []
    for cfg, n in [("Re300_L4", 10), ("Re400_L4", 10), ("Re800_L8", 10), ("Re101_L4", 10)]:
        for s in range(n):
            r2 = float(np.clip(rng.beta(2, 3), 0.01, 0.95))
            I = 100 * r2 + rng.normal(0, 18) - 10          # correlated, noisy
            recs.append(dict(config=cfg, kind="latent", seed=s, r2=r2, I=I,
                             G=float(np.clip(100 * r2 + rng.normal(0, 15), -200, 100)),
                             source="synthetic"))
    return recs


# ────────────────────────────────────────────────────────────────────────────
#  Metrics
# ────────────────────────────────────────────────────────────────────────────
def evaluate(recs, thr, key="I"):
    """Apply threshold to a set of records; return out-of-sample metrics."""
    vals = np.array([r[key] for r in recs if r[key] is not None], dtype=float)
    r2s  = np.array([r["r2"]  for r in recs if r[key] is not None], dtype=float)
    if len(vals) == 0:
        return None
    acc = r2s > thr
    out = dict(n=int(len(vals)), n_accepted=int(acc.sum()),
               mean_all=float(vals.mean()),
               mean_accepted=float(vals[acc].mean()) if acc.any() else float("nan"),
               mean_rejected=float(vals[~acc].mean()) if (~acc).any() else float("nan"),
               # false accept: accepted, but the correction HURT
               n_false_accept=int((acc & (vals < 0)).sum()),
               false_accept_rate=float((acc & (vals < 0)).sum() / max(acc.sum(), 1)),
               # false reject: rejected, but the correction would have helped a lot
               n_false_reject=int(((~acc) & (vals > 50)).sum()),
               false_reject_rate=float(((~acc) & (vals > 50)).sum() / max((~acc).sum(), 1)))
    return out


def pick_threshold(recs, grid, key="I", objective="youden", harm=0.0, marginal=25.0):
    """Choose a threshold on the FIT half only, by an explicit objective.

    objective:
      'min_safe'  smallest threshold admitting no harmful network (efficacy<harm).
                  This is a SAFETY FLOOR, not an optimum: if nothing is harmful
                  above the grid floor it returns the floor, by construction.
      'youden'    treat gating as classification of 'worth correcting'
                  (efficacy > marginal) and maximise sensitivity+specificity-1.
                  Non-degenerate: penalised both for accepting poor networks and
                  for rejecting good ones.
      'max_total' maximise total efficacy delivered, sum over accepted.
      'max_mean'  maximise mean efficacy of accepted (degenerate upward).
    """
    vals = np.array([r[key] for r in recs if r[key] is not None], dtype=float)
    r2s  = np.array([r["r2"]  for r in recs if r[key] is not None], dtype=float)
    if len(vals) == 0:
        return None
    best, best_score = None, -np.inf
    for thr in grid:
        acc = r2s > thr
        if objective == "min_safe":
            if acc.sum() == 0:
                continue
            harmful = (acc & (vals < harm)).sum()
            score = (0 if harmful else 1) * 1000 + acc.sum() - thr
        elif objective == "youden":
            # 'worth correcting' label. The fixed cutoff can fail to split the
            # data (e.g. field G is 90-99 everywhere, so nothing is 'poor').
            # Fall back to the median, which always yields two non-empty classes.
            good = vals > marginal
            if good.sum() == 0 or (~good).sum() == 0:
                good = vals > np.median(vals)
            if good.sum() == 0 or (~good).sum() == 0:
                continue                     # degenerate even at the median
            sens = (acc & good).sum() / good.sum()      # true positive rate
            spec = ((~acc) & (~good)).sum() / (~good).sum()
            score = sens + spec - 1.0
        elif objective == "max_total":
            score = vals[acc].sum() if acc.any() else -np.inf
        elif objective == "max_mean":
            score = vals[acc].mean() if acc.any() else -np.inf
        else:
            raise ValueError(objective)
        if score > best_score:
            best_score, best = score, float(thr)
    return best


def threshold_response(by_cfg, configs, grid, key, n_splits, rng):
    """Out-of-sample response curve: for each threshold, average over random
    stratified splits the coverage, accepted efficacy and false-accept rate."""
    rows = []
    for thr in grid:
        cov, mu, fa, anyfa = [], [], [], []
        for _ in range(n_splits):
            hold = []
            for c in configs:
                sub = [r for r in by_cfg[c] if r[key] is not None]
                if len(sub) < 2:
                    continue
                idx = rng.permutation(len(sub))
                hold += [sub[i] for i in idx[len(sub) // 2:]]
            m = evaluate(hold, thr, key)
            if m is None or m["n_accepted"] == 0:
                continue
            cov.append(m["n_accepted"] / m["n"])
            mu.append(m["mean_accepted"])
            fa.append(m["false_accept_rate"])
            anyfa.append(1.0 if m["n_false_accept"] > 0 else 0.0)
        if not mu:
            continue
        rows.append(dict(threshold=float(thr), coverage=float(np.mean(cov)),
                         mean_accepted=float(np.mean(mu)),
                         std_accepted=float(np.std(mu)),
                         false_accept_rate=float(np.mean(fa)),
                         frac_splits_with_false_accept=float(np.mean(anyfa))))
    return rows


# ────────────────────────────────────────────────────────────────────────────
#  Load
# ────────────────────────────────────────────────────────────────────────────
print("=" * 74, flush=True)
print("HELD-OUT VALIDATION OF THE R^2 APPLICABILITY THRESHOLD", flush=True)
print("=" * 74, flush=True)

if args.dry_run:
    print("\n[DRY RUN] synthetic records, structure test only\n", flush=True)
    recs, skipped = synthetic_records(rng), []
else:
    recs, skipped = load_records(DATA)

if skipped:
    print("\nFiles skipped:", flush=True)
    for p, why in skipped:
        print(f"   {os.path.basename(p)}: {why}", flush=True)

if not recs:
    sys.exit("\nNo usable per-seed records found. Check --data-dir.")

configs = sorted({r["config"] for r in recs})
print(f"\nLoaded {len(recs)} per-seed records across {len(configs)} configurations:", flush=True)
for c in configs:
    sub = [r for r in recs if r["config"] == c]
    kinds = sorted({r['kind'] for r in sub})
    nG = sum(1 for r in sub if r["G"] is not None)
    print(f"   {c:12s} n={len(sub):3d}  kinds={','.join(kinds):13s} with G: {nG}", flush=True)

has_G = any(r["G"] is not None for r in recs)
metrics = ["I"] + (["G"] if has_G else [])
grid = np.arange(args.sweep[0], args.sweep[1] + 1e-9, args.sweep[2])
results = {}

# ────────────────────────────────────────────────────────────────────────────
#  TEST A — stratified random split, published threshold
# ────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 74, flush=True)
print(f"TEST A — stratified random 50/50 split, threshold fixed at "
      f"{args.threshold}, {args.n_splits} repeats", flush=True)
print("─" * 74, flush=True)

by_cfg = {c: [r for r in recs if r["config"] == c] for c in configs}
for key in metrics:
    accs, alls, fa, nacc = [], [], [], []
    for _ in range(args.n_splits):
        hold = []
        for c in configs:                     # stratify: half of each config
            sub = [r for r in by_cfg[c] if r[key] is not None]
            if len(sub) < 2:
                continue
            idx = rng.permutation(len(sub))
            hold += [sub[i] for i in idx[len(sub) // 2:]]
        m = evaluate(hold, args.threshold, key)
        if m is None or np.isnan(m["mean_accepted"]):
            continue
        accs.append(m["mean_accepted"]); alls.append(m["mean_all"])
        fa.append(m["false_accept_rate"]); nacc.append(m["n_accepted"])
    if not accs:
        print(f"  [{key}] too few records to split", flush=True); continue
    accs, alls, fa = np.array(accs), np.array(alls), np.array(fa)
    print(f"  [{key}]  held-out accepted : {accs.mean():6.1f} "
          f"+/- {accs.std():4.1f}   (5-95%: {np.percentile(accs,5):.1f}"
          f"–{np.percentile(accs,95):.1f})", flush=True)
    print(f"        held-out all      : {alls.mean():6.1f} +/- {alls.std():4.1f}", flush=True)
    print(f"        gain from gating  : {accs.mean()-alls.mean():+6.1f} points", flush=True)
    print(f"        false-accept rate : {fa.mean()*100:5.2f}%  "
          f"(splits with any false accept: {100*np.mean(fa>0):.1f}%)", flush=True)
    print(f"        mean accepted n   : {np.mean(nacc):.1f}", flush=True)
    results[f"A_{key}"] = dict(mean=float(accs.mean()), std=float(accs.std()),
                               p5=float(np.percentile(accs, 5)),
                               p95=float(np.percentile(accs, 95)),
                               mean_all=float(alls.mean()),
                               false_accept_rate=float(fa.mean()))

# ────────────────────────────────────────────────────────────────────────────
#  TEST B — leave-one-configuration-out
# ────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 74, flush=True)
print("TEST B — leave-one-configuration-out (threshold fitted on the others)", flush=True)
print("─" * 74, flush=True)

if len(configs) < 3:
    print("  need at least 3 configurations; skipping", flush=True)
else:
    for key in metrics:
        print(f"\n  [{key}]", flush=True)
        rows = []
        for c in configs:
            fit  = [r for r in recs if r["config"] != c and r[key] is not None]
            hold = [r for r in recs if r["config"] == c and r[key] is not None]
            if not fit or not hold:
                continue
            thr_sel = pick_threshold(fit, grid, key)
            if thr_sel is None:
                print(f"    held out {c:12s} no threshold selectable on the "
                      f"remaining data; showing published only", flush=True)
                m_pub = evaluate(hold, args.threshold, key)
                if m_pub is not None:
                    print(f"       with published  : accepted {m_pub['n_accepted']}/{m_pub['n']}"
                          f"  mean {m_pub['mean_accepted']:6.1f}"
                          f"  false-accept {m_pub['n_false_accept']}", flush=True)
                    rows.append((c, float("nan"), None, m_pub))
                continue
            m_sel = evaluate(hold, thr_sel, key)
            m_pub = evaluate(hold, args.threshold, key)
            if m_sel is None or m_pub is None:
                continue
            rows.append((c, thr_sel, m_sel, m_pub))
            print(f"    held out {c:12s} threshold chosen on others = {thr_sel:.2f}", flush=True)
            print(f"       with chosen thr : accepted {m_sel['n_accepted']}/{m_sel['n']}"
                  f"  mean {m_sel['mean_accepted']:6.1f}  false-accept {m_sel['n_false_accept']}",
                  flush=True)
            print(f"       with published  : accepted {m_pub['n_accepted']}/{m_pub['n']}"
                  f"  mean {m_pub['mean_accepted']:6.1f}  false-accept {m_pub['n_false_accept']}",
                  flush=True)
        results[f"B_{key}"] = [dict(held_out=c, thr_selected=t,
                                    selected=ms, published=mp) for c, t, ms, mp in rows]
        if rows:
            tot_fa = sum(r[3]["n_false_accept"] for r in rows)
            mus = [r[3]["mean_accepted"] for r in rows if not np.isnan(r[3]["mean_accepted"])]
            print(f"    --> published threshold, pooled over held-out configs: "
                  f"mean accepted {np.mean(mus):.1f}, total false accepts {tot_fa}", flush=True)

# ────────────────────────────────────────────────────────────────────────────
#  TEST C — threshold sweep, selected on fit half, validated out of sample
# ────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 74, flush=True)
print("TEST C — threshold response curve and selection objectives", flush=True)
print("─" * 74, flush=True)

for key in metrics:
    print(f"\n  [{key}] out-of-sample response to the threshold "
          f"(averaged over {args.n_splits // 4} stratified splits)", flush=True)
    rows = threshold_response(by_cfg, configs, grid, key, args.n_splits // 4, rng)
    if not rows:
        print("    insufficient data", flush=True); continue
    print(f"    {'thr':>5} {'coverage':>9} {'accepted eff.':>15} {'false acc.':>11}", flush=True)
    for r in rows:
        mark = "  <- published" if abs(r["threshold"] - args.threshold) < 1e-9 else ""
        print(f"    {r['threshold']:5.2f} {100*r['coverage']:8.1f}% "
              f"{r['mean_accepted']:9.1f} +/-{r['std_accepted']:4.1f} "
              f"{100*r['false_accept_rate']:10.2f}%{mark}", flush=True)
    results[f"C_curve_{key}"] = rows

    # the safety floor: lowest threshold with no false accept in ANY split
    safe = [r["threshold"] for r in rows if r["frac_splits_with_false_accept"] == 0.0]
    floor = min(safe) if safe else None
    if floor is not None:
        print(f"    minimum threshold with zero false accepts in every split: "
              f"{floor:.2f}", flush=True)
        print(f"    the published {args.threshold} is therefore "
              f"{'conservative' if args.threshold > floor else 'at or below the floor'}"
              f" by {args.threshold - floor:+.2f}", flush=True)
    results[f"C_safefloor_{key}"] = floor

    # selection objectives, each fitted on half and validated on the other half
    print(f"    selection objectives (fitted on half, validated out of sample):", flush=True)
    for obj in ["youden", "min_safe", "max_total"]:
        sel, oos, fa = [], [], []
        for _ in range(args.n_splits // 4):
            fit, hold = [], []
            for c in configs:
                sub = [r for r in by_cfg[c] if r[key] is not None]
                if len(sub) < 2:
                    continue
                idx = rng.permutation(len(sub))
                fit  += [sub[i] for i in idx[:len(sub) // 2]]
                hold += [sub[i] for i in idx[len(sub) // 2:]]
            if not fit or not hold:
                continue
            thr = pick_threshold(fit, grid, key, objective=obj)
            if thr is None:
                continue
            m = evaluate(hold, thr, key)
            if m is None or np.isnan(m["mean_accepted"]):
                continue
            sel.append(thr); oos.append(m["mean_accepted"]); fa.append(m["false_accept_rate"])
        if not sel:
            print(f"      {obj:10s}: insufficient data", flush=True); continue
        sel, oos = np.array(sel), np.array(oos)
        print(f"      {obj:10s}: chose {sel.mean():.2f} +/- {sel.std():.2f} "
              f"(median {np.median(sel):.2f})  -> held-out efficacy "
              f"{oos.mean():5.1f} +/-{oos.std():4.1f}, false accepts "
              f"{100*np.mean(fa):.2f}%", flush=True)
        results[f"C_obj_{obj}_{key}"] = dict(thr_mean=float(sel.mean()),
                                             thr_median=float(np.median(sel)),
                                             oos_mean=float(oos.mean()),
                                             false_accept_rate=float(np.mean(fa)))

# ────────────────────────────────────────────────────────────────────────────
out_path = os.path.join(DATA if not args.dry_run else ".", args.out)
json.dump(dict(threshold=args.threshold, n_records=len(recs),
               configs=configs, results=results), open(out_path, "w"),
          indent=2, default=float)
print("\n" + "=" * 74, flush=True)
print(f"Saved -> {out_path}", flush=True)
print("""
Reading the output:
  * If TEST A's held-out accepted mean is close to the in-sample 74-80%, the
    published figure is not a selection artefact.
  * If TEST B shows the threshold transferring to a configuration it was not
    fitted on -- especially the square cylinder, a different geometry -- that is
    the strongest form of the claim.
  * If TEST C's chosen threshold clusters near 0.3, the published value was a
    reasonable choice rather than a tuned one.
  * A false-accept rate of zero out of sample is the claim worth stating: the
    diagnostic never admits a network the correction would harm.
""", flush=True)
