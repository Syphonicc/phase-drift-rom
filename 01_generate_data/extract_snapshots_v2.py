#!/usr/bin/env python3
"""
extract_snapshots_v2.py  —  Re-parameterized version
====================================================
Extracts U (ux, uy) and p from an OpenFOAM cylinder case, interpolates
onto a structured 128x64 grid, saves as numpy. Takes Reynolds number
and case directory as arguments so one script serves all cases.

Usage:
    python3 extract_snapshots_v2.py --re 400 \
        --case ~/OpenFOAM/run/Cylinder2D_study/cylinder2D_Re=400_LSTM

    # Re=300 (defaults to its case path if you follow the naming pattern)
    python3 extract_snapshots_v2.py --re 300

Output:
    ~/cylinder_lstm_data/snapshots_Re{RE}.npy
    ~/cylinder_lstm_data/metadata_Re{RE}.npy
"""

import os, re, sys, argparse, numpy as np
from pathlib import Path

try:
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
    from scipy.spatial import Delaunay
except ImportError:
    sys.exit("extract_snapshots_v2.py needs scipy: pip install -r requirements.txt")

# ── Arguments ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--re",   type=int, required=True, help="Reynolds number")
parser.add_argument("--case", type=str, default=None,
                    help="Case directory (default: standard naming pattern)")
parser.add_argument("--tstart", type=float, default=50.01,
                    help="First snapshot time to keep (discard transient)")
parser.add_argument("--nx", type=int, default=128)
parser.add_argument("--ny", type=int, default=64)
args = parser.parse_args()

RE      = args.re
NX, NY  = args.nx, args.ny
T_START = args.tstart

if args.case:
    CASE_DIR = os.path.expanduser(args.case)
else:
    CASE_DIR = os.path.expanduser(
        f"~/OpenFOAM/run/Cylinder2D_study/cylinder2D_Re={RE}_LSTM"
    )

OUT_DIR = os.path.expanduser("~/cylinder_lstm_data")
os.makedirs(OUT_DIR, exist_ok=True)

print(f"Re        : {RE}", flush=True)
print(f"Case dir  : {CASE_DIR}", flush=True)
print(f"Grid      : {NX} x {NY}", flush=True)
print(f"T_start   : {T_START}", flush=True)

if not os.path.isdir(CASE_DIR):
    print(f"ERROR: case directory not found: {CASE_DIR}", flush=True)
    sys.exit(1)

# ── Parsers ───────────────────────────────────────────────────────────────────
def read_of_scalar(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    m = re.search(
        r'internalField\s+nonuniform\s+List<scalar>\s*\n\s*(\d+)\s*\n\s*\((.*?)\)\s*;',
        content, re.DOTALL)
    if m is None:
        raise ValueError(f"Cannot find scalar internalField in {filepath}")
    return np.fromstring(m.group(2), sep='\n', dtype=np.float64)

def read_of_vector(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    m = re.search(
        r'internalField\s+nonuniform\s+List<vector>\s*\n\s*(\d+)\s*\n\s*\((.*?)\)\s*;',
        content, re.DOTALL)
    if m is None:
        raise ValueError(f"Cannot find vector internalField in {filepath}")
    rows = re.findall(
        r'\(\s*([-+eE0-9.]+)\s+([-+eE0-9.]+)\s+([-+eE0-9.]+)\s*\)', m.group(2))
    return np.array([[float(a), float(b), float(c)]
                     for a, b, c in rows], dtype=np.float64).T

def parse_int_list(content):
    m = re.search(r'\b(\d+)\s*\n\s*\(([^)]+)\)', content, re.DOTALL)
    return np.array(m.group(2).split(), dtype=np.int32) if m else None

def parse_float_list_3d(content):
    m = re.search(r'\b(\d+)\s*\n\s*\((.+?)\n\)', content, re.DOTALL)
    if not m: return None
    rows = re.findall(
        r'\(\s*([-+eE0-9.]+)\s+([-+eE0-9.]+)\s+([-+eE0-9.]+)\s*\)', m.group(2))
    return np.array([[float(a), float(b), float(c)]
                     for a, b, c in rows], dtype=np.float64)

def compute_cell_centres(case_dir):
    mesh = Path(case_dir) / "constant" / "polyMesh"
    print("  Reading polyMesh/points ...", flush=True)
    with open(mesh / "points") as f: pts = parse_float_list_3d(f.read())
    print("  Reading polyMesh/faces ...", flush=True)
    with open(mesh / "faces") as f: fcs = f.read()
    m = re.search(r'\b(\d+)\s*\n\s*\((.+?)\n\)', fcs, re.DOTALL)
    faces = [list(map(int, r.split()))
             for r in re.findall(r'\d+\(([^)]+)\)', m.group(2).strip())]
    print("  Reading polyMesh/owner ...", flush=True)
    with open(mesh / "owner") as f: owner = parse_int_list(f.read())
    n_cells = int(owner.max()) + 1
    cx = np.zeros(n_cells); cy = np.zeros(n_cells); cnt = np.zeros(n_cells, dtype=np.int32)
    for fi, verts in enumerate(faces):
        if fi >= len(owner): break
        c = owner[fi]
        for vid in verts:
            cx[c] += pts[vid, 0]; cy[c] += pts[vid, 1]; cnt[c] += 1
    mask = cnt > 0
    cx[mask] /= cnt[mask]; cy[mask] /= cnt[mask]
    print(f"    {n_cells} cell centres  X:[{cx.min():.3f},{cx.max():.3f}]  "
          f"Y:[{cy.min():.3f},{cy.max():.3f}]", flush=True)
    return cx, cy

# ── Step 1: time dirs ─────────────────────────────────────────────────────────
print("\n[1] Scanning time directories ...", flush=True)
time_dirs = {}
for d in Path(CASE_DIR).iterdir():
    try:
        t = float(d.name)
        if t >= T_START: time_dirs[t] = d.name
    except ValueError: continue
all_times    = sorted(time_dirs.keys())
all_dirnames = [time_dirs[t] for t in all_times]
print(f"    {len(all_times)} snapshots, t={all_times[0]:.2f}→{all_times[-1]:.2f}",
      flush=True)

# ── Step 2: cell centres ──────────────────────────────────────────────────────
print("\n[2] Cell centres ...", flush=True)
cx, cy  = compute_cell_centres(CASE_DIR)
cell_xy = np.column_stack([cx, cy])

# ── Step 3: grid + triangulation ──────────────────────────────────────────────
print("\n[3] Grid + triangulation ...", flush=True)
x_lin = np.linspace(cx.min(), cx.max(), NX)
y_lin = np.linspace(cy.min(), cy.max(), NY)
xx, yy   = np.meshgrid(x_lin, y_lin)
grid_pts = np.column_stack([xx.ravel(), yy.ravel()])
tri = Delaunay(cell_xy)
print("    done.", flush=True)

# ── Step 4: extract ───────────────────────────────────────────────────────────
print("\n[4] Extracting ...", flush=True)
n = len(all_times)
data = np.zeros((n, 3, NY, NX), dtype=np.float32)
for i, (t, dn) in enumerate(zip(all_times, all_dirnames)):
    td = Path(CASE_DIR) / dn
    try:
        U = read_of_vector(str(td / "U")); p = read_of_scalar(str(td / "p"))
    except Exception as e:
        print(f"    SKIP t={t}: {e}", flush=True); continue
    for ch, fld in enumerate([U[0], U[1], p]):
        g = LinearNDInterpolator(tri, fld)(grid_pts).reshape(NY, NX)
        nan = np.isnan(g)
        if nan.any():
            g[nan] = NearestNDInterpolator(cell_xy, fld)(grid_pts[nan.ravel()])
        data[i, ch] = g.astype(np.float32)
    if (i+1) % 200 == 0 or i == 0:
        print(f"    [{i+1}/{n}] t={t:.2f}", flush=True)

# ── Step 5: save ──────────────────────────────────────────────────────────────
print("\n[5] Saving ...", flush=True)
print(f"    shape={data.shape}  NaNs={np.isnan(data).sum()}", flush=True)
print(f"    ux:[{data[:,0].min():.3f},{data[:,0].max():.3f}]  "
      f"p:[{data[:,2].min():.3f},{data[:,2].max():.3f}]", flush=True)
np.save(os.path.join(OUT_DIR, f"snapshots_Re{RE}.npy"), data)
np.save(os.path.join(OUT_DIR, f"metadata_Re{RE}.npy"),
    {'times': np.array(all_times), 'x_grid': x_lin, 'y_grid': y_lin,
     'nx': NX, 'ny': NY, 't_start': T_START, 're': RE,
     'channels': ['ux','uy','p'], 'shape': data.shape}, allow_pickle=True)
print(f"\nDONE → snapshots_Re{RE}.npy  ({data.nbytes/1e6:.1f} MB)  shape={data.shape}",
      flush=True)
