"""Automatic system against human pilot: each with its own DETECTION delay.

Before detection nobody touches anything (de = 0, throttle = 0): the aircraft enters the
stall and falls. On detecting, each arm starts acting:

  DP   detects at 0.33 s  (Bunge's detector with flow-angle sensors) and from
       there runs the policy in real time, throttle to full with a 0.6 s
       ramp
  CAA  detects at TAU_H s (pilot) and brings the throttle up over 2 s from
       that
       instante
  FAA  detects at TAU_H s but waits for the nose-down (alpha < 14) to start
       the 2 s ramp

delta_e comes from the POLICY in all three, once each has detected: the only
thing that tells them apart is WHEN they start and how they bring power up.

The engine carries Riley's first-order lag (A4) in all three.

Usage: deteccion.py <policy.npz> <V0> [tau_dp] [tau_human] [tau_engine]
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
TAU_DP = float(sys.argv[3]) if len(sys.argv) > 3 else 0.33
TAU_H = float(sys.argv[4]) if len(sys.argv) > 4 else 1.00
TAU_M = float(sys.argv[5]) if len(sys.argv) > 5 else 0.50
DE_NOREAC = np.deg2rad(float(sys.argv[6]) if len(sys.argv) > 6 else -25.0)
A0, ALPHA_UNSTALL, MAX_TIME = 20.0, np.deg2rad(14.0), 15.0

env = SymmetricStall()
pi = PolicyIterationStall.load(POLICY, env=env)
v_stall, dt = env.airplane.STALL_AIRSPEED, env.airplane.TIME_STEP

ARMS = [
    ("DP  automatic   (det. %.2f s, 0.6 s ramp)" % TAU_DP,
     TAU_DP, 0.6, "immediate", "#0072B2"),
    ("CAA pilot       (det. %.2f s, 2 s ramp)" % TAU_H,
     TAU_H, 2.0, "immediate", "#D55E00"),
    ("FAA pilot       (det. %.2f s, 2 s ramp after n-d)" % TAU_H,
     TAU_H, 2.0, "after_nose_down", "#009E73"),
]


def run_arm(t_det, ramp, when):
    obs, _ = env.specific_reset(0.0, V0, np.deg2rad(A0), 0.0)
    t, h, dt_ef, t_uns, t_pwr = 0.0, 0.0, 0.0, None, None
    stop, hs = RecoveryMonitor(dt), [0.0]
    H = {k: [] for k in ("t", "gamma", "v_norm", "alpha", "q", "de",
                         "dt_cmd", "dt_ef", "h")}
    while t < MAX_TIME:
        detectado = t >= t_det
        de = float(get_optimal_action(obs, pi)[0][0]) if detectado else float(DE_NOREAC)

        if detectado and t_uns is None and obs[2] < ALPHA_UNSTALL:
            t_uns = t
        if not detectado:
            cmd = 0.0
        else:
            if when == "immediate":
                t_pwr = t_det
            elif t_uns is not None:
                t_pwr = t_uns
            cmd = 0.0 if t_pwr is None else float(
                np.clip((t - t_pwr) / ramp, 0.0, 1.0))

        dt_ef = cmd if TAU_M <= 0 else dt_ef + (cmd - dt_ef) * (dt / TAU_M)

        for k, v in (("t", t), ("gamma", obs[0]), ("v_norm", obs[1]),
                     ("alpha", obs[2]), ("q", obs[3]), ("de", de),
                     ("dt_cmd", cmd), ("dt_ef", dt_ef), ("h", h)):
            H[k].append(v)

        obs, _, _, _, _ = env.step(np.array([de, dt_ef], dtype=np.float32))
        h += obs[1] * v_stall * np.sin(obs[0]) * dt
        t += dt
        hs.append(h)
        if stop.update(np.rad2deg(obs[0])):
            return dict(hmin=min(hs), t=t, est="recuperado", H=H, t_uns=t_uns)
        if obs[2] >= np.deg2rad(40) or obs[0] <= -np.pi + 0.05:
            return dict(hmin=min(hs), t=t, est="CHOQUE", H=H, t_uns=t_uns)
    return dict(hmin=min(hs), t=t, est="not closed", H=H, t_uns=t_uns)


print("IC: gamma=0, V=%.2f Vs, alpha=20 deg    engine tau_e=%.2f s" % (V0, TAU_M))
print("before detecting: de=%.1f deg (the pilot PULLS, he did not react), throttle=0\n"
      % np.rad2deg(DE_NOREAC))
print("%-46s %9s %8s %10s %11s" % ("arm", "h_min", "dur", "nose-down", "status"))
print("-" * 90)
res = {}
for name, t_det, ramp, when, col in ARMS:
    r = run_arm(t_det, ramp, when); res[name] = r
    print("%-46s %+9.3f %7.2fs %9.2fs %11s" % (
        name, r["hmin"], r["t"], r["t_uns"] if r["t_uns"] else -1, r["est"]))

base = res[ARMS[0][0]]["hmin"]
print("\npenalty against the automatic system:")
for name, *_ in ARMS[1:]:
    print("  %-44s %+8.3f m" % (name, res[name]["hmin"] - base))

# ───── figura ─────
PAN = [("gamma", r"$\gamma$ (deg)", np.rad2deg), ("v_norm", r"$V/V_s$", lambda x: x),
       ("alpha", r"$\alpha$ (deg)", np.rad2deg), ("q", r"$q$ (deg/s)", np.rad2deg),
       ("de", r"$\delta_e$ (deg)", np.rad2deg), ("dt", r"$\delta_t$", lambda x: x),
       ("h", "altitude (m)", lambda x: x)]
fig, axes = plt.subplots(len(PAN), 1, figsize=(7.4, 12.2), sharex=True)
for ax, (k, et, cv) in zip(axes, PAN):
    for name, t_det, ramp, when, col in ARMS:
        H = res[name]["H"]
        if k == "dt":
            ax.plot(H["t"], H["dt_cmd"], lw=1.0, ls=":", color=col, alpha=0.6, zorder=2)
            ax.plot(H["t"], H["dt_ef"], lw=1.5, color=col, label=name, zorder=3)
        else:
            ax.plot(H["t"], cv(np.asarray(H[k])), lw=1.5, color=col, label=name, zorder=3)
            if k == "h":
                y = np.asarray(H[k]); i = int(np.argmin(y))
                ax.plot(H["t"][i], y[i], "o", ms=5, color=col, mec="white", mew=1.0, zorder=4)
                ax.annotate("%.2f m" % y[i], xy=(H["t"][i], y[i]), xytext=(6, 0),
                            textcoords="offset points", fontsize=7.5, color=col, va="center")
    for tv, lab in ((TAU_DP, "system detects"), (TAU_H, "pilot detects")):
        ax.axvline(tv, color="0.65", lw=0.8, ls=":", zorder=1)
    if k == "alpha":
        ax.axhline(14.0, color="0.5", lw=0.8, ls="--", zorder=1)
    if k == "dt":
        ax.set_ylim(-0.02, 1.05)
        ax.annotate("dotted = throttle, solid = engine", xy=(0.99, 0.28),
                    xycoords=("axes fraction", "data"), ha="right", fontsize=7.5,
                    color="0.35")
    if k == "gamma":
        ax.annotate("%.2f s\ndetecta\nel sistema" % TAU_DP, xy=(TAU_DP, 0.02),
                    xycoords=("data", "axes fraction"), fontsize=6.5, color="0.4",
                    ha="center", va="bottom")
        ax.annotate("%.2f s\npilot\ndetects" % TAU_H, xy=(TAU_H, 0.02),
                    xycoords=("data", "axes fraction"), fontsize=6.5, color="0.4",
                    ha="center", va="bottom")
    if k in ("gamma", "h"):
        ax.axhline(0.0, color="0.85", lw=0.6, zorder=0)
    ax.set_ylabel(et, fontsize=9); ax.grid(True, color="0.92", lw=0.5)
    ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"): ax.spines[sp].set_color("0.6")
    ax.tick_params(labelsize=8, color="0.6")
axes[-1].set_xlabel("tiempo (s)", fontsize=9)
axes[0].legend(loc="lower left", fontsize=7.5, frameon=False, ncol=1,
               bbox_to_anchor=(0.0, 1.02))
fig.suptitle(r"Detection: system (%.2f s) vs pilot (%.2f s), engine $\tau_e=%.2f$ s"
             "\n$V_0=%.2f\\,V_s$, not reacting $\\delta_e=%.0f^\\circ$" % (TAU_DP, TAU_H, TAU_M, V0, np.rad2deg(DE_NOREAC)), fontsize=10, y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.94])
out = main.RESULTS_DIR / ("3_maniobras/deteccion_v%03d_de%+03d.png" % (round(V0*100), round(np.rad2deg(DE_NOREAC))))
fig.savefig(out, dpi=200)
print("\n-> %s" % out)
