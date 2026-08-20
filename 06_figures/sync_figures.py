#!/usr/bin/env python3
"""Regenerate figures, sync into the Overleaf bundle, verify includes resolve."""
import argparse, os, re, shutil, subprocess, sys

DATA   = os.path.expanduser("~/cylinder_lstm_data")
# The bundle directory is whichever one under Paper_tex_files actually holds
# main.tex. Repeated downloads leave numbered duplicates ("... (1)"), and an
# earlier husk may still contain a figs/ folder, so locate by content not name.
def _find_bundle():
    root = os.path.join(DATA, "Paper_tex_files")
    hits = [d for d, _, fs in os.walk(root) if "main.tex" in fs]
    if not hits:
        sys.exit("no main.tex found under %s" % root)
    return max(hits, key=os.path.getmtime)

BUNDLE = _find_bundle()
FIGS   = os.path.join(BUNDLE, "figs")

GENERATORS = [
    ("fig01_configurations.pdf",        ["python3", "make_fig01.py"]),
    ("fig03_spectrum_cylinder.pdf",     ["python3", "make_paper_figures.py"]),
    ("fig04_spectrum_square.pdf",       None),
    ("fig05_decomposition.pdf",         None),
    ("fig06_correction.pdf",            None),
    ("fig07_field.pdf",                 None),
    ("fig08_dmd_baseline.pdf",          ["python3", "plot_dmd_baseline.py"]),
    ("fig09_tangential_transverse.pdf", ["python3", "tangential_transverse.py",
                                         "--out-prefix", "fig09_tangential_transverse"]),
]
SEARCH = [os.path.join(DATA, "paper_figs"), DATA]

def run(cmd):
    print("  $ " + " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=DATA, capture_output=True, text=True)
    if r.returncode != 0:
        tail = (r.stderr.strip().splitlines() or ["<no stderr>"])[-1]
        print("  FAILED rc=%d: %s" % (r.returncode, tail), flush=True)

def newest(name):
    hits = [os.path.join(d, name) for d in SEARCH
            if os.path.exists(os.path.join(d, name))]
    return max(hits, key=os.path.getmtime) if hits else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regen", action="store_true")
    args = ap.parse_args()
    os.makedirs(FIGS, exist_ok=True)

    if args.regen:
        print("REGENERATING", flush=True)
        for name, cmd in GENERATORS:
            if cmd:
                run(cmd)
        print(flush=True)

    print("SYNCING -> figs/", flush=True)
    missing = []
    for name, _ in GENERATORS:
        src = newest(name)
        if src is None:
            stem = name[:-4]
            stray = sorted({f for d in SEARCH for f in os.listdir(d)
                            if f.startswith(stem) and f != name})
            note = "  (found instead: %s)" % ", ".join(stray) if stray else ""
            print("  MISSING  %s%s" % (name, note), flush=True)
            missing.append(name)
            continue
        dst = os.path.join(FIGS, name)
        fresh = (os.path.exists(dst)
                 and os.path.getmtime(dst) >= os.path.getmtime(src)
                 and os.path.getsize(dst) == os.path.getsize(src))
        shutil.copy2(src, dst)
        print("  %s %s  <- %s" % ("ok     " if fresh else "UPDATED",
                                  name, os.path.relpath(src, DATA)), flush=True)

    print("\nVERIFYING includes", flush=True)
    main_tex = open(os.path.join(BUNDLE, "main.tex")).read()
    roots = [BUNDLE] + [os.path.join(BUNDLE, p)
                        for p in re.findall(r"\\graphicspath\{\{(.*?)\}\}", main_tex)]
    bad = 0
    for tex in sorted(f for f in os.listdir(BUNDLE) if f.endswith(".tex")):
        body = open(os.path.join(BUNDLE, tex)).read()
        for ref in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", body):
            cands = [ref] if os.path.splitext(ref)[1] else \
                    [ref + e for e in (".pdf", ".png", ".jpg", ".eps")]
            if not any(os.path.exists(os.path.join(r, c))
                       for r in roots for c in cands):
                print("  UNRESOLVED  %s: %s" % (tex, ref), flush=True)
                bad += 1
    if bad == 0:
        print("  all includes resolve", flush=True)

    print("\n%d missing, %d unresolved" % (len(missing), bad), flush=True)
    sys.exit(1 if (missing or bad) else 0)

if __name__ == "__main__":
    main()
