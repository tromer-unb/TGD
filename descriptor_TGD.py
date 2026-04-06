import os
import re
import numpy as np
import pandas as pd
from ase.io import read
from ase.neighborlist import neighbor_list
from collections import defaultdict

# ============================================================
# CONFIGURATION
# ============================================================
STRUCT_DIR = "/home/tromer/Carbon_allotropes/fourier/descritor/juncao/structures/"
OUT_CSV = "descriptors.csv"

Nx, Ny = 10, 10                  # supercell
Rmin, Rmax = 3, 30               # can be increased for cey-graphene (e.g.: 22+)
r_max = 3.5
k_candidates = 12
bond_factor = 1.40               # graphene ~1.25 | phagraphene ~1.35-1.45

# ============================================================
# CORRECT NUMERICAL ORDER: 1,2,3,...,10,11...
# ============================================================
def numeric_key(fname):
    m = re.match(r"(\d+)", fname)
    if m:
        return (0, int(m.group(1)))
    return (1, fname)

# ============================================================
# AUXILIARY FUNCTIONS
# ============================================================
def angle_ccw(u, v):
    # CCW angle from u->v in [0,2pi)
    return np.mod(np.arctan2(u[0]*v[1] - u[1]*v[0], u @ v), 2*np.pi)

def canonical_cycle(c):
    # canonicalization to avoid duplicates by rotation/reversal
    m = len(c)
    reps = []
    for i in range(m):
        r = c[i:] + c[:i]
        reps.append(tuple(r))
        reps.append(tuple(reversed(r)))
    return min(reps)

def topo_entropy_from_counts(counts):
    # counts: dict R->count
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    ent = 0.0
    for c in counts.values():
        if c <= 0:
            continue
        p = c / total
        ent -= p * np.log(p + 1e-15)
    return float(ent)

# ============================================================
# PROCESS A STRUCTURE -> returns a feature dictionary
# ============================================================
def process_structure(path):
    atoms = read(path).repeat((Nx, Ny, 1))
    atoms.set_pbc([True, True, False])  # PBC for faces (effective rings)
    pos2d = atoms.get_positions()[:, :2]
    n = len(atoms)

    # -------------------------
    # Automatic chemical graph
    # -------------------------
    i_idx, j_idx, d = neighbor_list("ijd", atoms, r_max)

    cand = [[] for _ in range(n)]
    all_d = []
    for i, j, dist in zip(i_idx, j_idx, d):
        if i != j:
            cand[i].append((dist, j))
            all_d.append(dist)

    if len(all_d) == 0:
        raise RuntimeError(f"No neighbors detected in {path}. Adjust r_max.")

    d0 = min(all_d)
    bond_max = bond_factor * d0

    knn = [sorted(c, key=lambda x: x[0])[:k_candidates] for c in cand]
    knn_sets = [set(j for dist, j in lst if dist <= bond_max) for lst in knn]

    neighbors = {i: [] for i in range(n)}
    for i in range(n):
        for j in knn_sets[i]:
            if i in knn_sets[j]:
                neighbors[i].append(j)

    def mic(i, j):
        # 2D vector with MIC (PBC)
        return atoms.get_distance(i, j, mic=True, vector=True)[:2]

    # -------------------------
    # Face extraction (B2)
    # -------------------------
    def next_left(prev, curr):
        vin = mic(curr, prev)
        best, best_ang = None, None
        for k in neighbors[curr]:
            if k == prev:
                continue
            vout = mic(curr, k)
            ang = angle_ccw(vin, vout)
            if best_ang is None or ang < best_ang:
                best, best_ang = k, ang
        return best

    faces = []
    areas = []
    visited = set()

    for i in range(n):
        for j in neighbors[i]:
            if (i, j) in visited:
                continue

            face = []
            a, b = i, j

            for _ in range(3000):  # safety
                visited.add((a, b))
                face.append(a)

                c = next_left(a, b)
                if c is None:
                    break

                a, b = b, c

                if (a, b) == (i, j):
                    break

            if len(face) < Rmin:
                continue

            can = canonical_cycle(face)
            if can in faces:
                continue

            # unwrap to compute area
            coords = np.zeros((len(can), 2), dtype=float)
            coords[0] = pos2d[can[0]]
            for t in range(1, len(can)):
                coords[t] = coords[t - 1] + mic(can[t - 1], can[t])

            area = 0.5 * abs(np.sum(
                coords[:, 0] * np.roll(coords[:, 1], -1) -
                coords[:, 1] * np.roll(coords[:, 0], -1)
            ))

            faces.append(can)
            areas.append(area)

    if len(faces) == 0:
        raise RuntimeError(f"No face detected in {path}. Adjust parameters.")

    faces = [list(f) for f in faces]
    areas = np.array(areas, dtype=float)
    Rs = np.array([len(f) for f in faces], dtype=int)

    # -------------------------
    # Counts / fractions
    # -------------------------
    total_faces = len(Rs)
    counts = {R: int(np.sum(Rs == R)) for R in range(Rmin, Rmax + 1)}
    # main fractions
    pR5 = counts.get(5, 0) / total_faces
    pR6 = counts.get(6, 0) / total_faces
    pR7 = counts.get(7, 0) / total_faces
    pR8plus = int(np.sum(Rs >= 8)) / total_faces

    # topological entropy (improvement #1)
    topo_entropy = topo_entropy_from_counts({R: c for R, c in counts.items() if c > 0})

    # -------------------------
    # Topological charge q = 6 - R
    # -------------------------
    q = 6 - Rs
    mean_q = float(np.mean(q))
    mean_abs_q = float(np.mean(np.abs(q)))
    var_q = float(np.var(q))

    # -------------------------
    # Face geometry (improvement #2: percentiles)
    # -------------------------
    mean_area = float(np.mean(areas))
    median_area = float(np.median(areas))
    area_std = float(np.std(areas))
    area_cv = float(area_std / (mean_area + 1e-15))
    area_p25 = float(np.percentile(areas, 25))
    area_p75 = float(np.percentile(areas, 75))
    max_area = float(np.max(areas))

    frac_area_Rge10 = float(np.sum(areas[Rs >= 10]) / (np.sum(areas) + 1e-15))

    # -------------------------
    # Dual graph: face adjacencies
    # -------------------------
    face_edges = defaultdict(set)
    for idx, f in enumerate(faces):
        m = len(f)
        for t in range(m):
            e = tuple(sorted((f[t], f[(t + 1) % m])))
            face_edges[e].add(idx)

    adj_pairs = []
    for fs in face_edges.values():
        if len(fs) == 2:
            a, b = list(fs)
            adj_pairs.append((Rs[a], Rs[b]))

    adj_pairs = np.array(adj_pairs, dtype=int)
    if len(adj_pairs) > 0:
        frac_adj_5_7 = float(np.mean(
            ((adj_pairs[:, 0] == 5) & (adj_pairs[:, 1] == 7)) |
            ((adj_pairs[:, 0] == 7) & (adj_pairs[:, 1] == 5))
        ))
        frac_adj_defect_defect = float(np.mean(
            (adj_pairs[:, 0] != 6) & (adj_pairs[:, 1] != 6)
        ))
    else:
        frac_adj_5_7 = 0.0
        frac_adj_defect_defect = 0.0

    return {
        "system": os.path.basename(path),

        # Basic topology
        "pR5": pR5, "pR6": pR6, "pR7": pR7, "pR8plus": pR8plus,

        # Improvement #1
        "topo_entropy": topo_entropy,

        # Curvature/disclination
        "mean_q": mean_q, "mean_abs_q": mean_abs_q, "var_q": var_q,

        # Geometry
        "mean_area": mean_area,
        "median_area": median_area,
        "area_std": area_std,
        "area_cv": area_cv,
        "area_p25": area_p25,
        "area_p75": area_p75,
        "frac_area_Rge10": frac_area_Rge10,
        "max_area": max_area,

        # Dual graph
        "frac_adj_5_7": frac_adj_5_7,
        "frac_adj_defect_defect": frac_adj_defect_defect,

        # Useful debug
        "n_faces": int(total_faces),
        "bond_d0": float(d0),
        "bond_max": float(bond_max),
    }

# ============================================================
# MAIN: iterates through structures in ascending order and saves CSV
# ============================================================
def main():
    files = sorted(
        [f for f in os.listdir(STRUCT_DIR)
         if f.lower().endswith((".cif", ".xyz", ".vasp", ".poscar"))],
        key=numeric_key
    )

    if len(files) == 0:
        raise RuntimeError(f"No file found in '{STRUCT_DIR}'.")

    rows = []
    for f in files:
        full = os.path.join(STRUCT_DIR, f)
        print("Processing:", f)
        rows.append(process_structure(full))

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    print("\n✅ descriptors.csv successfully generated:", OUT_CSV)
    print("Processed order:")
    for i, f in enumerate(files, 1):
        print(f"{i:3d} -> {f}")

if __name__ == "__main__":
    main()
