"""Ben's flight animator: turns flight-recorder .npz files into GIFs + an overview image.
Usage: python animate_flight.py <log_dir>
"""
import glob, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

log_dir = sys.argv[1] if len(sys.argv) > 1 else "drone_logs"
files = sorted(glob.glob(os.path.join(log_dir, "episode_*.npz")))
if not files:
    sys.exit(f"No episode_*.npz files in {log_dir}")

GREEN, ORANGE, GREY, BLUE = "#2ecc71", "#f39c12", "#999999", "#1f77b4"

def gate_corners(c, heading, size):
    u = np.array([-np.sin(heading), np.cos(heading), 0.0])  # sideways
    z = np.array([0.0, 0.0, 1.0])
    h = size / 2
    pts = [c + u * h + z * h, c - u * h + z * h, c - u * h - z * h, c + u * h - z * h, c + u * h + z * h]
    return np.array(pts)

def gate_color(i, current):
    return GREY if i < current else (GREEN if i == current else ORANGE)

eps = []
for f in files:
    d = np.load(f)
    eps.append((os.path.basename(f), d))
    print(f"{os.path.basename(f)}: {int(d['gates_passed'])}/{len(d['gates'])} gates, "
          f"{len(d['traj']) * float(d['dt']):.1f}s, crashed={bool(d['crashed'])}")

# ---- overview: top-down view of every episode ----
n = len(eps)
cols = min(4, n)
rows = int(np.ceil(n / cols))
fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.2 * rows), squeeze=False)
for ax, (name, d) in zip(axes.flat, eps):
    tr, gates, hd, size = d["traj"], d["gates"], d["headings"], float(d["gate_size"])
    for i, (g, h) in enumerate(zip(gates, hd)):
        c = gate_corners(g, h, size)
        ax.plot(c[:2, 0], c[:2, 1], color=GREY if i < int(d["gates_passed"]) else ORANGE, lw=3)
    ax.plot(tr[:, 0], tr[:, 1], color=BLUE, lw=1.5)
    ax.plot(tr[0, 0], tr[0, 1], "ko", ms=4)
    ax.plot(tr[-1, 0], tr[-1, 1], "rx" if bool(d["crashed"]) else "g*", ms=8)
    ax.set_title(f"{name[:-4]}: {int(d['gates_passed'])}/{len(gates)} gates", fontsize=9)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.3)
for ax in list(axes.flat)[n:]:
    ax.axis("off")
fig.suptitle("Top-down view (grey = passed gate, orange = missed, x = crash)", fontsize=11)
fig.tight_layout()
overview = os.path.join(log_dir, "overview.png")
fig.savefig(overview, dpi=120)
plt.close(fig)
print("saved", overview)

# ---- 3D animation for the best episodes ----
best = sorted(eps, key=lambda e: (-int(e[1]["gates_passed"]), e[0]))[:3]
for name, d in best:
    tr, gates, hd, size, dt = d["traj"], d["gates"], d["headings"], float(d["gate_size"]), float(d["dt"])
    skip = max(1, int(round(0.04 / dt)))  # ~25 fps, real time
    frames = list(range(0, len(tr), skip)) + [len(tr) - 1]

    fig = plt.figure(figsize=(10, 5.5))
    ax = fig.add_subplot(111, projection="3d")
    allp = np.vstack([tr[:, :3], gates])
    lo, hi = allp.min(0) - 1.0, allp.max(0) + 1.0
    lo[2] = 0.0
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    aspect = hi - lo
    aspect[1] = max(aspect[1], aspect[0] * 0.35)  # keep it readable on long courses
    aspect[2] = max(aspect[2], aspect[0] * 0.25)
    try:
        ax.set_box_aspect(aspect, zoom=1.2)
    except TypeError:  # older matplotlib
        ax.set_box_aspect(aspect)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
    ax.view_init(elev=25, azim=-60)

    gate_lines = []
    for g, h in zip(gates, hd):
        c = gate_corners(g, h, size)
        (ln,) = ax.plot(c[:, 0], c[:, 1], c[:, 2], lw=3)
        gate_lines.append(ln)
    (path,) = ax.plot([], [], [], color=BLUE, lw=2)
    (drone,) = ax.plot([], [], [], "o", color="k", ms=7)
    title = ax.set_title("")

    def update(k):
        i = frames[k]
        cur = int(tr[i, 3])
        for gi, ln in enumerate(gate_lines):
            ln.set_color(gate_color(gi, cur))
        path.set_data(tr[: i + 1, 0], tr[: i + 1, 1]); path.set_3d_properties(tr[: i + 1, 2])
        drone.set_data([tr[i, 0]], [tr[i, 1]]); drone.set_3d_properties([tr[i, 2]])
        title.set_text(f"{name[:-4]}   t = {i * dt:4.1f}s   gates passed: {cur}/{len(gates)}")
        return gate_lines + [path, drone, title]

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 * dt * skip)
    out = os.path.join(log_dir, name.replace(".npz", ".gif"))
    anim.save(out, writer=PillowWriter(fps=max(1, int(round(1 / (dt * skip))))))
    plt.close(fig)
    print("saved", out)
print("DONE")
