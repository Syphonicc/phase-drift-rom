#!/usr/bin/env python3
"""v2: wall normal oriented into the fluid (v1 marched tangentially).
R3.4(i): near-wall resolution of each mesh, from polyMesh geometry.
For every face of the body patch, the owner cell's height normal to the wall is
the largest distance of that cell's points from the wall-face plane (exact for
the hexahedral wall cells used here).  The number of cells inside the laminar
boundary-layer scale delta = D Re^-1/2 is counted along the wall normal of each
wall face by marching outward through cells (face-neighbour walk) until the
cumulative height exceeds delta.  Reports min/mean/max first-cell height / D and
the min/mean number of cells within delta, for each Re that uses the mesh."""
import os, re, json, gzip, numpy as np

from paths import ARCHIVE as E, OUT_REV
MESHES = {   # name: (case dir, body patch, D, [Re list using this mesh])
    "circular_baseline": (f"{E}/cylinder2D_Re=400_LSTM", "cylinder", 0.12, [300, 400, 500, 600, 700, 800, 1000]),
    "circular_fine":     (f"{E}/Re400_gridfine",         "cylinder", 0.12, [400, 800, 1000]),
    "square":            (f"{E}/squareCyl_Re100",        None,       0.12, [100]),
}

def rd(p):
    return gzip.open(p + ".gz", "rt").read() if not os.path.exists(p) and os.path.exists(p + ".gz") else open(p).read()

def body(txt):
    m = re.search(r'\n(\d+)\s*\n\(', txt); n = int(m.group(1)); return n, txt[m.end():]

def read_points(d):
    n, b = body(rd(d + "/points"))
    a = np.array(re.findall(r'\(([-+\deE.]+)\s+([-+\deE.]+)\s+([-+\deE.]+)\)', b)[:n], float); return a

def read_faces(d):
    n, b = body(rd(d + "/faces"))
    return [list(map(int, f.split())) for f in re.findall(r'\d+\(([^)]*)\)', b)[:n]]

def read_labels(d, name):
    n, b = body(rd(d + "/" + name)); return np.array(b.split()[:n], int)

def read_boundary(d):
    txt = rd(d + "/boundary"); out = {}
    for m in re.finditer(r'(\w+)\s*\{([^}]*)\}', txt):
        g = m.group(2)
        if 'startFace' in g:
            out[m.group(1)] = dict(type=re.search(r'type\s+(\w+)', g).group(1),
                                   nFaces=int(re.search(r'nFaces\s+(\d+)', g).group(1)),
                                   startFace=int(re.search(r'startFace\s+(\d+)', g).group(1)))
    return out

res = {}
for name, (case, patch, D, res_list) in MESHES.items():
    d = case + "/constant/polyMesh"
    pts, faces = read_points(d), read_faces(d)
    own, nei = read_labels(d, "owner"), read_labels(d, "neighbour")
    bnd = read_boundary(d)
    if patch is None:   # the wall patch that is not inlet/outlet/sides
        cand = [k for k, v in bnd.items() if v['type'] == 'wall'] or \
               [k for k, v in bnd.items() if v['type'] == 'patch' and k not in ('inlet', 'outlet')]
        patch = cand[0]
    bp = bnd[patch]
    ncell = own.max() + 1
    cell_faces = [[] for _ in range(ncell)]
    for f, c in enumerate(own): cell_faces[c].append(f)
    for f, c in enumerate(nei): cell_faces[c].append(f)
    def cell_pts(c): return np.unique(np.concatenate([faces[f] for f in cell_faces[c]]))
    h1, ncells_in = [], {re_: [] for re_ in res_list}
    for f in range(bp['startFace'], bp['startFace'] + bp['nFaces']):
        fp = pts[faces[f]]; fc = fp.mean(0)
        nrm = np.cross(fp[1] - fp[0], fp[2] - fp[0]); nrm /= np.linalg.norm(nrm)
        if (pts[cell_pts(own[f])].mean(0) - fc) @ nrm < 0: nrm = -nrm   # point into the fluid
        c = own[f]; h = np.abs((pts[cell_pts(c)] - fc) @ nrm).max(); h1.append(h)
        # march outward: next cell = face-neighbour across the face whose centre is farthest along nrm
        for re_ in res_list:
            delta = D / np.sqrt(re_); cum, cc, cnt, prev = 0.0, c, 0, f
            heights = []
            while True:
                hp = np.abs((pts[cell_pts(cc)] - fc) @ nrm).max()
                cnt += 1
                if hp >= delta or cnt > 200: break
                best, bestd = None, -1
                for g in cell_faces[cc]:
                    if g >= len(nei): continue
                    dd = (pts[faces[g]].mean(0) - fc) @ nrm
                    if dd > bestd: bestd, best = dd, g
                if best is None: break
                cc = nei[best] if own[best] == cc else own[best]
            ncells_in[re_].append(cnt)
    h1 = np.array(h1) / D
    res[name] = dict(patch=patch, n_wall_faces=int(bp['nFaces']), n_cells=int(ncell),
                     h1_over_D_min=float(h1.min()), h1_over_D_mean=float(h1.mean()), h1_over_D_max=float(h1.max()),
                     cells_in_delta={str(k): dict(min=int(min(v)), mean=float(np.mean(v)), delta_over_D=float(1/np.sqrt(k)))
                                     for k, v in ncells_in.items()})
    print(name, json.dumps(res[name], indent=1))
json.dump(res, open(os.path.join(OUT_REV, "nearwall_resolution_v2.json"), "w"), indent=1)
