import glob, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
log_dir = sys.argv[1] if len(sys.argv) > 1 else "drone_logs"
files = sorted(glob.glob(os.path.join(log_dir, "episode_*.npz")))
n = len(files); cols = min(4, n); rows = int(np.ceil(n / cols))
fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)
for ax, f in zip(axes.flat, files):
    d = np.load(f); tr, gates, size = d["traj"], d["gates"], float(d["gate_size"])
    passed = int(d["gates_passed"])
    for i, g in enumerate(gates):
        ax.plot([g[0], g[0]], [g[2] - size / 2, g[2] + size / 2], color="#999999" if i < passed else "#f39c12", lw=3)
    ax.plot(tr[:, 0], tr[:, 2], color="#1f77b4", lw=1.5)
    ax.plot(tr[-1, 0], tr[-1, 2], "rx" if bool(d["crashed"]) else "g*", ms=8)
    ax.axhline(0.1, color="red", lw=0.8, ls="--")
    ax.set_ylim(0, 4); ax.grid(alpha=0.3)
    ax.set_title(f"{os.path.basename(f)[:-4]}: {passed}/{len(gates)} gates", fontsize=9)
for ax in list(axes.flat)[n:]:
    ax.axis("off")
fig.suptitle("SIDE view: height (z) vs forward distance (x). Red dashed = floor limit", fontsize=11)
fig.tight_layout(); out = os.path.join(log_dir, "side_view.png"); fig.savefig(out, dpi=120)
print("saved", out)
