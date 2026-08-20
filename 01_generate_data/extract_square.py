#!/usr/bin/env python3
"""
extract_square.py
=================
Extracts (ux, uy, p) from the SQUARE CYLINDER OpenFOAM case onto a
structured 128x64 grid, cropped to the body + near-wake window.

Differences from the circular-cylinder extractor:
  - different case path
  - CROPPED extraction window (Option A) so resolution sits where the
    dynamics are, and the frame is comparable to the cylinder data
  - cells inside the solid square are masked (set to 0) after interpolation

Extraction window:
    x: -0.5 to 2.0    (~4D upstream, ~17D downstream of the body)
    y: -0.6 to 0.6    (10D tall, body centred)
  -> 2.5 x 1.2 domain on a 128x64 grid = cells 0.0195 x 0.0188 (near-square)

Output (drops straight into the existing pipeline):
    ~/cylinder_lstm_data/snapshots_SQ100.npy      (N, 3, 64, 128)
    ~/cylinder_lstm_data/metadata_SQ100.npy

Usage:
    python3 extract_square.py
    python3 extract_square.py --tstart 60.0 --nx 128 --ny 64
"""

import os, re, sys, argparse, numpy as np
from pathlib import Path
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.spatial import Delaunay

ap = argparse.ArgumentParser()
ap.add_argument("--case", type=str,
    default="~/OpenFOAM/run/SquareCylinder2D_study/squareCyl_Re100")
ap.add_argument("--tag",    type=str,   default="SQ100")
ap.add_argument("--tstart", type=float, default=60.0)
ap.add_argument("--nx",     type=int,   default=128)
ap.add_argument("--ny",     type=int,   default=64)
# crop window (Option A)
ap.add_argument("--xmin", type=float, default=-0.5)
ap.add_argument("--xmax", type=float, default= 2.0)
ap.add_argument("--ymin", type=float, default=-0.6)
ap.add_argument("--ymax", type=float, default= 0.6)
# square body half-width, for masking
ap.add_argument("--halfD", type=float, default=0.06)
args = ap.parse_args()

CASE = os.path.expanduser(args.case)
OUT  = os.path.expanduser("~/cylinder_lstm_data")
os.makedirs(OUT, exist_ok=True)
NX, NY = args.nx, args.ny
TAG = args.tag

print(f"Case   : {CASE}",  flush=True)
print(f"Window : x[{args.xmin},{args.xmax}]  y[{args.ymin},{args.ymax}]", flush=True)
print(f"Grid   : {NX} x {NY}", flush=True)
print(f"t_start: {args.tstart}", flush=True)

if not os.path.isdir(CASE):
    sys.exit(f"ERROR: case not found: {CASE}")

# ── OpenFOAM parsers ──────────────────────────────────────────────────────────
def read_scalar(fp):
    c = open(fp).read()
    m = re.search(r'internalField\s+nonuniform\s+List<scalar>\s*\n\s*(\d+)\s*\n\s*\((.*?)\)\s*;',
                  c, re.DOTALL)
    if not m:
        mu = re.search(r'internalField\s+uniform\s+([-\deE.+]+)\s*;', c)
        if mu: return None, float(mu.group(1))
        raise ValueError(f"no scalar internalField in {fp}")
    return np.fromstring(m.group(2), sep='\n', dtype=np.float64), None

def read_vector(fp):
    c = open(fp).read()
    m = re.search(r'internalField\s+nonuniform\s+List<vector>\s*\n\s*(\d+)\s*\n\s*\((.*?)\)\s*;',
                  c, re.DOTALL)
    if not m:
        raise ValueError(f"no vector internalField in {fp}")
    rows = re.findall(r'\(\s*([-+eE0-9.]+)\s+([-+eE0-9.]+)\s+([-+eE0-9.]+)\s*\)', m.group(2))
    return np.array([[float(a),float(b),float(c_)] for a,b,c_ in rows]).T

def parse_int_list(c):
    m = re.search(r'\b(\d+)\s*\n\s*\(([^)]+)\)', c, re.DOTALL)
    return np.array(m.group(2).split(), dtype=np.int32) if m else None

def parse_points(c):
    m = re.search(r'\b(\d+)\s*\n\s*\((.+?)\n\)', c, re.DOTALL)
    rows = re.findall(r'\(\s*([-+eE0-9.]+)\s+([-+eE0-9.]+)\s+([-+eE0-9.]+)\s*\)', m.group(2))
    return np.array([[float(a),float(b),float(c_)] for a,b,c_ in rows])

def cell_centres(case):
    mesh = Path(case)/"constant"/"polyMesh"
    print("  reading points ...", flush=True)
    pts = parse_points(open(mesh/"points").read())
    print("  reading faces ...", flush=True)
    fc = open(mesh/"faces").read()
    m = re.search(r'\b(\d+)\s*\n\s*\((.+?)\n\)', fc, re.DOTALL)
    faces = [list(map(int, r.split()))
             for r in re.findall(r'\d+\(([^)]+)\)', m.group(2).strip())]
    print("  reading owner ...", flush=True)
    owner = parse_int_list(open(mesh/"owner").read())
    n = int(owner.max())+1
    cx = np.zeros(n); cy = np.zeros(n); cnt = np.zeros(n, dtype=np.int32)
    for fi, verts in enumerate(faces):
        if fi >= len(owner): break
        c = owner[fi]
        for v in verts:
            cx[c] += pts[v,0]; cy[c] += pts[v,1]; cnt[c] += 1
    msk = cnt > 0
    cx[msk] /= cnt[msk]; cy[msk] /= cnt[msk]
    print(f"    {n} cells  X:[{cx.min():.2f},{cx.max():.2f}] Y:[{cy.min():.2f},{cy.max():.2f}]",
          flush=True)
    return cx, cy

# ── times ─────────────────────────────────────────────────────────────────────
print("\n[1] scanning time dirs ...", flush=True)
tdirs = {}
for d in Path(CASE).iterdir():
    try:
        t = float(d.name)
        if t >= args.tstart: tdirs[t] = d.name
    except ValueError:
        continue
times = sorted(tdirs)
print(f"    {len(times)} snapshots, t={times[0]:.2f} -> {times[-1]:.2f}", flush=True)

# ── mesh ──────────────────────────────────────────────────────────────────────
print("\n[2] cell centres ...", flush=True)
cx, cy = cell_centres(CASE)
xy = np.column_stack([cx, cy])

# ── target grid (CROPPED window) ──────────────────────────────────────────────
print("\n[3] building grid + triangulation ...", flush=True)
x_lin = np.linspace(args.xmin, args.xmax, NX)
y_lin = np.linspace(args.ymin, args.ymax, NY)
xx, yy = np.meshgrid(x_lin, y_lin)
gpts = np.column_stack([xx.ravel(), yy.ravel()])

# mask: grid points INSIDE the solid square get zeroed
inside = (np.abs(xx) <= args.halfD) & (np.abs(yy) <= args.halfD)
print(f"    {inside.sum()} of {NX*NY} grid points lie inside the square body (masked)",
      flush=True)

tri = Delaunay(xy)
print("    done.", flush=True)

# ── extract ───────────────────────────────────────────────────────────────────
print("\n[4] extracting ...", flush=True)
N = len(times)
data = np.zeros((N, 3, NY, NX), dtype=np.float32)
for i, t in enumerate(times):
    td = Path(CASE)/tdirs[t]
    try:
        U = read_vector(str(td/"U"))
        p, p_uniform = read_scalar(str(td/"p"))
        if p is None: p = np.full(len(cx), p_uniform)
    except Exception as e:
        print(f"    SKIP t={t}: {e}", flush=True)
        continue
    for ch, fld in enumerate([U[0], U[1], p]):
        g = LinearNDInterpolator(tri, fld)(gpts).reshape(NY, NX)
        nan = np.isnan(g)
        if nan.any():
            g[nan] = NearestNDInterpolator(xy, fld)(gpts[nan.ravel()])
        g[inside] = 0.0                      # zero the solid body
        data[i, ch] = g.astype(np.float32)
    if (i+1) % 200 == 0 or i == 0:
        print(f"    [{i+1}/{N}] t={t:.2f}", flush=True)

# ── save ──────────────────────────────────────────────────────────────────────
print("\n[5] saving ...", flush=True)
print(f"    shape={data.shape}  NaNs={np.isnan(data).sum()}", flush=True)
print(f"    ux:[{data[:,0].min():.3f},{data[:,0].max():.3f}]  "
      f"uy:[{data[:,1].min():.3f},{data[:,1].max():.3f}]  "
      f"p:[{data[:,2].min():.3f},{data[:,2].max():.3f}]", flush=True)

np.save(os.path.join(OUT, f"snapshots_{TAG}.npy"), data)
np.save(os.path.join(OUT, f"metadata_{TAG}.npy"), {
    'times': np.array(times), 'x_grid': x_lin, 'y_grid': y_lin,
    'nx': NX, 'ny': NY, 't_start': args.tstart,
    'geometry': 'square_cylinder', 'Re': 100, 'D': 0.12, 'U': 1.0,
    'St_measured': 0.1480, 'f_shed_measured': 1.2333, 'Cd_measured': 1.5016,
    'window': (args.xmin, args.xmax, args.ymin, args.ymax),
    'channels': ['ux','uy','p'], 'shape': data.shape,
    'body_mask': inside,
}, allow_pickle=True)

print(f"\nDONE -> snapshots_{TAG}.npy  ({data.nbytes/1e6:.1f} MB)  {data.shape}", flush=True)
print(f"        f_shed = 1.2333 Hz  (St = 0.148)  <- the frequency the LSTM error should inherit",
      flush=True)
