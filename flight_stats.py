import glob, os, sys
import numpy as np
log_dir = sys.argv[1] if len(sys.argv) > 1 else "drone_logs"
rows = []
for f in sorted(glob.glob(os.path.join(log_dir, "episode_*.npz"))):
    d = np.load(f); tr, gates, hd, size, dt = d["traj"], d["gates"], d["headings"], float(d["gate_size"]), float(d["dt"])
    finished = int(d["gates_passed"]) == len(gates)
    t = len(tr) * dt
    dist = np.linalg.norm(np.diff(tr[:, :3], axis=0), axis=1).sum()
    offs = []  # how far from the gate centre (as % of half-size) at each pass
    for k in np.where(np.diff(tr[:, 3]) > 0)[0] + 1:
        g = int(tr[k, 3]) - 1
        if g >= len(gates): continue
        n = np.array([np.cos(hd[g]), np.sin(hd[g])])
        v = tr[k, :2] - gates[g, :2]
        lat = np.linalg.norm(v - (v @ n) * n)
        dz = abs(tr[k, 2] - gates[g, 2])
        offs.append(100 * max(lat, dz) / (size / 2))
    rows.append((finished, t, dist / t if t > 0 else 0, np.mean(offs) if offs else np.nan))
r = np.array(rows, dtype=float)
fin = r[r[:, 0] == 1]
print(f"episodes: {len(r)}   finished: {len(fin)} ({100 * len(fin) / len(r):.0f}%)")
if len(fin):
    print(f"finish time (finished runs): {fin[:, 1].mean():.1f} s")
    print(f"average speed (finished runs): {fin[:, 2].mean():.2f} m/s")
print(f"gate precision: {np.nanmean(r[:, 3]):.0f}% off-centre on average (0% = dead centre, 100% = touching the edge)")
