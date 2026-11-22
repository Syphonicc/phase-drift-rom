#!/usr/bin/env python3
"""
extract_snapshots_v5.py -- v4 concatenated processor cell centres, whose owner-face
approximation differs slightly from the serial mesh (0.05% rms). v5 scatters each
processor field into SERIAL cell order with cellProcAddressing and uses the serial
mesh in constant/polyMesh: identical to reconstructPar + v3 by construction.
(v3 docstring follows)
extract_snapshots_v3.py -- as v2 (same parsers, linear Delaunay interpolation of
cell-centre values with nearest-neighbour fill), plus:
  * reads gzip-compressed OpenFOAM fields (writeCompression on)
  * interpolates onto a FIXED grid, by default the production 128x64 grid of
    metadata_Re400.npy, so fine-mesh / long-domain cases share identical points
  * --tend, --out-tag, --out-dir; never writes the production snapshots_Re*.npy
  * parallel over snapshots
Usage:
  python3 extract_snapshots_v3.py --case DIR --out-tag Re400_fine --tstart 50.01 --tend 90
"""
import os
for _v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): os.environ[_v] = "1"
import re, sys, gzip, argparse, numpy as np
from pathlib import Path
from multiprocessing import Pool
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.spatial import Delaunay

ap = argparse.ArgumentParser()
ap.add_argument("--case", required=True)
ap.add_argument("--out-tag", required=True)
ap.add_argument("--out-dir", default=os.path.join(os.path.expanduser(os.environ.get("PDROM_DATA_DIR", "~/cylinder_lstm_data")), "revision"))
ap.add_argument("--tstart", type=float, default=50.01)
ap.add_argument("--tend", type=float, default=1e9)
ap.add_argument("--grid-from", default=os.path.join(os.path.expanduser(os.environ.get("PDROM_DATA_DIR", "~/cylinder_lstm_data")), "metadata_Re400.npy"))
ap.add_argument("--workers", type=int, default=6)
ap.add_argument("--decomposed", action="store_true")
args = ap.parse_args()
CASE_DIR = os.path.expanduser(args.case)

def _open(path):
    path = str(path)
    if os.path.exists(path): return open(path, 'r')
    if os.path.exists(path + '.gz'): return gzip.open(path + '.gz', 'rt')
    raise FileNotFoundError(path)

# ── Parsers ───────────────────────────────────────────────────────────────────
def read_of_scalar(filepath):
    with _open(filepath) as f:
        content = f.read()
    m = re.search(
        r'internalField\s+nonuniform\s+List<scalar>\s*\n\s*(\d+)\s*\n\s*\((.*?)\)\s*;',
        content, re.DOTALL)
    if m is None:
        raise ValueError(f"Cannot find scalar internalField in {filepath}")
    return np.fromstring(m.group(2), sep='\n', dtype=np.float64)

def read_of_vector(filepath):
    with _open(filepath) as f:
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
    with _open(mesh / "points") as f: pts = parse_float_list_3d(f.read())
    print("  Reading polyMesh/faces ...", flush=True)
    with _open(mesh / "faces") as f: fcs = f.read()
    m = re.search(r'\b(\d+)\s*\n\s*\((.+?)\n\)', fcs, re.DOTALL)
    faces = [list(map(int, r.split()))
             for r in re.findall(r'\d+\(([^)]+)\)', m.group(2).strip())]
    print("  Reading polyMesh/owner ...", flush=True)
    with _open(mesh / "owner") as f: owner = parse_int_list(f.read())
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


# ── time directories / mesh ───────────────────────────────────────────────────
PROCS = sorted(Path(CASE_DIR).glob("processor*"), key=lambda p: int(p.name[9:])) if args.decomposed else [Path(CASE_DIR)]
times = {}
for d in PROCS[0].iterdir():
    try: t = float(d.name)
    except ValueError: continue
    if args.tstart <= t <= args.tend + 1e-9: times[t] = d.name
all_times = sorted(times); names = [times[t] for t in all_times]
print(f"{len(all_times)} snapshots t={all_times[0]:.2f}->{all_times[-1]:.2f}  ({len(PROCS)} mesh parts)", flush=True)
cell_xy = np.column_stack(compute_cell_centres(CASE_DIR))        # serial mesh
if args.decomposed:
    ADDR = [np.array(_open(P / "constant/polyMesh/cellProcAddressing").read().split("(")[1].split(")")[0].split(), int)
            for P in PROCS]
    assert sum(len(a) for a in ADDR) == len(cell_xy), "cellProcAddressing does not cover the serial mesh"
meta = np.load(args.grid_from, allow_pickle=True).item()
x_lin, y_lin = meta['x_grid'], meta['y_grid']
NX, NY = len(x_lin), len(y_lin)
xx, yy = np.meshgrid(x_lin, y_lin)
grid_pts = np.column_stack([xx.ravel(), yy.ravel()])
tri = Delaunay(cell_xy)

def one(dn):
    if args.decomposed:
        U = np.zeros((3, len(cell_xy))); p = np.zeros(len(cell_xy))
        for P, a in zip(PROCS, ADDR):
            U[:, a] = read_of_vector(str(P / dn / "U")); p[a] = read_of_scalar(str(P / dn / "p"))
    else:
        U = read_of_vector(str(PROCS[0] / dn / "U")); p = read_of_scalar(str(PROCS[0] / dn / "p"))
    out = np.zeros((3, NY, NX), np.float32)
    for ch, fld in enumerate([U[0], U[1], p]):
        g = LinearNDInterpolator(tri, fld)(grid_pts).reshape(NY, NX)
        nan = np.isnan(g)
        if nan.any(): g[nan] = NearestNDInterpolator(cell_xy, fld)(grid_pts[nan.ravel()])
        out[ch] = g
    return out

import multiprocessing as _mp
with _mp.get_context("fork").Pool(args.workers) as pool:
    data = np.stack(pool.map(one, names, chunksize=8))
os.makedirs(args.out_dir, exist_ok=True)
np.save(os.path.join(args.out_dir, f"snapshots_{args.out_tag}.npy"), data)
np.save(os.path.join(args.out_dir, f"metadata_{args.out_tag}.npy"),
        {'times': np.array(all_times), 'x_grid': x_lin, 'y_grid': y_lin, 'nx': NX, 'ny': NY,
         't_start': args.tstart, 'case': CASE_DIR, 'channels': ['ux','uy','p'],
         'shape': data.shape, 'grid_from': args.grid_from}, allow_pickle=True)
print(f"DONE snapshots_{args.out_tag}.npy shape={data.shape} NaN={np.isnan(data).sum()}", flush=True)
