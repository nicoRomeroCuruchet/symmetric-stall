"""Isolate the effect of POWER: same elevator (the DP's) in all three arms.

  DP    0.6 s ramp, from t = 0
  CAA   2.0 s ramp, from t = 0
  FAA   2.0 s ramp, from the moment the aircraft unstalls (alpha < 14 deg),
        which is what happens after the nose-down

delta_e comes from the policy in all three cases, evaluated at the current
state. So the only difference between the curves is WHEN and HOW FAST the power
comes in.

Uso: potencia_pura.py <policy.npz> <V0> [alpha0]
"""
import sys
import logging
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.disable(logging.INFO)

from symmetric_stall import train as main
from symmetric_stall.policy_iteration import PolicyIterationStall
from symmetric_stall.aircraft.symmetric_stall import SymmetricStall
from symmetric_stall.utils.utils import get_optimal_action
from symmetric_stall.procedures import RecoveryMonitor

POLICY = Path(sys.argv[1])
V0 = float(sys.argv[2])
A0 = float(sys.argv[3]) if len(sys.argv) > 3 else 20.0
ALPHA_UNSTALL = np.deg2rad(14.0)
MAX_TIME = 15.0

env = SymmetricStall()
pi = PolicyIterationStall.load(POLICY, env=env)
v_stall = env.airplane.STALL_AIRSPEED
dt = env.airplane.TIME_STEP

ARMS = [
    ("DP   (0.6 s ramp from t=0)",   0.6, "t0",      "#0072B2"),
    ("CAA  (2 s ramp from t=0)",     2.0, "t0",      "#D55E00"),
    ("FAA  (2 s ramp after nose-down)", 2.0, "unstall", "#009E73"),
]


def run_arm(ramp, start):
    obs, _ = env.specific_reset(0.0, V0, np.deg2rad(A0), 0.0)
    t, h, t_uns = 0.0, 0.0, None
    stop = RecoveryMonitor(dt)
    hs = [0.0]
    H = {k: [] for k in ("t", "gamma", "v_norm", "alpha", "q", "de", "dt", "h")}
    while t < MAX_TIME:
        de = float(get_optimal_action(obs, pi)[0][0])
        if t_uns is None and obs[2] < ALPHA_UNSTALL:
            t_uns = t
        t_pwr = 0.0 if start == "t0" else t_uns
        thr = 0.0 if t_pwr is None else float(np.clip((t - t_pwr) / ramp, 0.0, 1.0))

        for k, v in (("t", t), ("gamma", obs[0]), ("v_norm", obs[1]),
                     ("alpha", obs[2]), ("q", obs[3]), ("de", de),
                     ("dt", thr), ("h", h)):
            H[k].append(v)

        obs, _, _, _, _ = env.step(np.array([de, thr], dtype=np.float32))
        h += obs[1] * v_stall * np.sin(obs[0]) * dt
        t += dt
        hs.append(h)
        if stop.update(np.rad2deg(obs[0])):
            return dict(h=h, hmin=min(hs), t=t, est="recuperado", H=H, t_uns=t_uns)
        if obs[2] >= np.deg2rad(40) or obs[0] <= -np.pi + 0.05:
            return dict(h=h, hmin=min(hs), t=t, est="CHOQUE", H=H, t_uns=t_uns)
    return dict(h=h, hmin=min(hs), t=t, est="sin cerrar", H=H, t_uns=t_uns)


print("IC: gamma=0, V=%.2f Vs, alpha=%.0f deg, q=0" % (V0, A0))
print("elevator: the policy's in ALL THREE arms\n")
print("%-34s %9s %8s %9s %11s" % ("arm", "h_min", "dur", "t_nose-dn", "status"))
print("-" * 76)
res = {}
for name, ramp, start, col in ARMS:
    r = run_arm(ramp, start)
    res[name] = r
    print("%-34s %+9.3f %7.2fs %8.2fs %11s" % (
        name, r["hmin"], r["t"], r["t_uns"] if r["t_uns"] else -1, r["est"]))

base = res[ARMS[0][0]]["hmin"]
print("\npenalty against the DP (from the power alone):")
for name, _, _, _ in ARMS[1:]:
    print("  %-32s %+8.3f m" % (name, res[name]["hmin"] - base))

# ───────── figura ─────────
PAN = [("gamma", r"$\gamma$ (deg)", np.rad2deg), ("v_norm", r"$V/V_s$", lambda x: x),
       ("alpha", r"$\alpha$ (deg)", np.rad2deg), ("q", r"$q$ (deg/s)", np.rad2deg),
       ("de", r"$\delta_e$ (deg)", np.rad2deg), ("dt", r"$\delta_t$", lambda x: x),
       ("h", "altura (m)", lambda x: x)]
fig, axes = plt.subplots(len(PAN), 1, figsize=(7.4, 12.2), sharex=True)
for ax, (k, et, cv) in zip(axes, PAN):
    for name, ramp, start, col in ARMS:
        H = res[name]["H"]
        ax.plot(H["t"], cv(np.asarray(H[k])), lw=1.5, color=col, label=name, zorder=3)
        if k == "h":
            y = np.asarray(H[k]); i = int(np.argmin(y))
            ax.plot(H["t"][i], y[i], "o", ms=5, color=col, mec="white", mew=1.0,
                    zorder=4)
            ax.annotate("%.2f m" % y[i], xy=(H["t"][i], y[i]), xytext=(6, 0),
                        textcoords="offset points", fontsize=7.5, color=col,
                        va="center")
    if k == "alpha":
        ax.axhline(14.0, color="0.5", lw=0.8, ls="--", zorder=1)
        ax.annotate(r"$\alpha_s=14^\circ$ (triggers the FAA ramp)", xy=(0.99, 14.0),
                    xycoords=("axes fraction", "data"), ha="right", va="bottom",
                    fontsize=7, color="0.35")
    if k == "dt":
        ax.set_ylim(-0.02, 1.05)
        tu = res[ARMS[2][0]]["t_uns"]
        if tu:
            ax.axvline(tu, color="0.6", lw=0.8, ls=":", zorder=1)
            ax.annotate("nose-down\ncompleto", xy=(tu, 0.55), fontsize=7,
                        color="0.4", ha="left", va="center",
                        xytext=(4, 0), textcoords="offset points")
    if k in ("gamma", "h"):
        ax.axhline(0.0, color="0.85", lw=0.6, zorder=0)
    ax.set_ylabel(et, fontsize=9); ax.grid(True, color="0.92", lw=0.5)
    ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"): ax.spines[sp].set_color("0.6")
    ax.tick_params(labelsize=8, color="0.6")
axes[-1].set_xlabel("tiempo (s)", fontsize=9)
axes[0].legend(loc="lower left", fontsize=8, frameon=False, ncol=1,
               bbox_to_anchor=(0.0, 1.02))
fig.suptitle(r"Same elevator, different power — $V_0=%.2f\,V_s$, $\alpha_0=%.0f^\circ$"
             % (V0, A0), fontsize=10, y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.945])
out = main.RESULTS_DIR / ("potencia_pura_v%03d.png" % round(V0 * 100))
fig.savefig(out, dpi=200)
print("\n-> %s" % out)
