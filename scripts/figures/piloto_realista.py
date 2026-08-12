"""Realistic pilot: follows the DP elevator with a human delay, and the engine
responds with a first-order lag (Riley, Appendix A, eq. A4).

Three ingredients, all three requested:
  1. the pilot applies the delta_e the DP asks for, but TAU_H seconds late
     (pure delay: he sees the state, decides, and the arm arrives afterwards);
  2. power comes up with Gratton's 2 s ramp, from t=0 (CAA) or from
     the unstall (FAA);
  3. the engine adds a first-order lag TAU_M on top of the throttle
     commanded:  d(dt_ef)/dt = (dt_cmd - dt_ef) / TAU_M.

The optimum also goes through the engine, so the comparison is fair: the only
advantage it keeps is having no human delay and commanding full from t=0.

Uso: piloto_realista.py <policy.npz> <V0> [tau_humano] [tau_motor ...]
"""
import sys
import logging
from pathlib import Path
from collections import deque

import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from symmetric_stall import train as main

logging.disable(logging.INFO)

from symmetric_stall.policy_iteration import PolicyIterationStall
from symmetric_stall.aircraft.symmetric_stall import SymmetricStall
from symmetric_stall.utils.utils import get_optimal_action
from symmetric_stall.procedures import RecoveryMonitor, GRATTON_RAMP_S, DE_DOWN

POLICY = Path(sys.argv[1])
V0 = float(sys.argv[2])
TAU_H = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
TAUS_M = [float(x) for x in sys.argv[4:]] or [0.0, 0.5, 1.0, 1.5]
A0 = 20.0
ALPHA_UNSTALL = np.deg2rad(14.0)
MAX_TIME = 15.0

env = SymmetricStall()
pi = PolicyIterationStall.load(POLICY, env=env)
v_stall = env.airplane.STALL_AIRSPEED
dt = env.airplane.TIME_STEP


def run_arm(mode, tau_m, tau_h=TAU_H):
    """mode: 'optimal' | 'caa' | 'faa'.  Returns dict with h_min, dur, etc."""
    obs, _ = env.specific_reset(0.0, V0, np.deg2rad(A0), 0.0)
    n_ret = int(round(tau_h / dt))
    cola = deque([np.float32(DE_DOWN)] * n_ret, maxlen=max(1, n_ret))
    dt_ef = 0.0 if mode != "optimal" else 0.0
    t, h, t_uns = 0.0, 0.0, None
    stop = RecoveryMonitor(dt)
    hs, hist = [0.0], {"t": [], "de": [], "dt_cmd": [], "dt_ef": [],
                       "alpha": [], "gamma": [], "h": [], "v_norm": [], "q": []}

    while t < MAX_TIME:
        de_opt = float(get_optimal_action(obs, pi)[0][0])
        if mode == "optimal":
            de = de_opt
        else:                       # the pilot applies what was asked tau_h ago
            de = float(cola[0]) if n_ret > 0 else de_opt
            cola.append(np.float32(de_opt))

        if t_uns is None and obs[2] < ALPHA_UNSTALL:
            t_uns = t
        if mode == "optimal":
            dt_cmd = 1.0
        elif mode == "caa":
            dt_cmd = min(t / GRATTON_RAMP_S, 1.0)
        else:                       # faa: the ramp starts on unstalling
            dt_cmd = 0.0 if t_uns is None else min((t - t_uns) / GRATTON_RAMP_S, 1.0)

        if tau_m > 0.0:             # first-order engine (Riley A4)
            dt_ef += (dt_cmd - dt_ef) * (dt / tau_m)
        else:
            dt_ef = dt_cmd

        for k, v in (("t", t), ("de", de), ("dt_cmd", dt_cmd), ("dt_ef", dt_ef),
                     ("alpha", obs[2]), ("gamma", obs[0]), ("h", h),
                     ("v_norm", obs[1]), ("q", obs[3])):
            hist[k].append(v)

        obs, _, _, _, _ = env.step(np.array([de, dt_ef], dtype=np.float32))
        h += obs[1] * v_stall * np.sin(obs[0]) * dt
        t += dt
        hs.append(h)

        if stop.update(np.rad2deg(obs[0])):
            return dict(h=h, hmin=min(hs), t=t, status="recovered", hist=hist)
        if obs[2] >= np.deg2rad(40) or obs[0] <= -np.pi + 0.05:
            return dict(h=h, hmin=min(hs), t=t, status="CRASH", hist=hist)
    return dict(h=h, hmin=min(hs), t=t, status="not closed", hist=hist)


print("IC: gamma=0, V=%.2f Vs, alpha=20 deg    human delay = %.2f s\n"
      % (V0, TAU_H))
print("%8s | %-24s | %-24s | %-24s" % ("tau_mot", "OPTIMUM (no human delay)",
                                       "CAA (DP de + 1s)", "FAA (DP de + 1s)"))
print("%8s | %8s %7s %7s | %8s %7s %7s | %8s %7s %7s" % (
    "(s)", "h_min", "dur", "est", "h_min", "dur", "est", "h_min", "dur", "est"))
print("-" * 92)
saved = {}
for tm in TAUS_M:
    row = "%8.2f |" % tm
    for mode in ("optimal", "caa", "faa"):
        r = run_arm(mode, tm)
        saved[(tm, mode)] = r
        row += " %8.3f %6.2fs %7s |" % (r["hmin"], r["t"], r["status"][:7])
    print(row)

print("\npilot penalty against the optimum, for each engine:")
for tm in TAUS_M:
    o = saved[(tm, "optimal")]["hmin"]
    c = saved[(tm, "caa")]["hmin"]
    f = saved[(tm, "faa")]["hmin"]
    print("  tau_motor=%.2f s:  CAA %+8.3f m   FAA %+8.3f m   (FAA-CAA %+6.3f m)"
          % (tm, c - o, f - o, f - c))

print("\nwhat a non-instantaneous engine costs the OPTIMUM:")
base = saved[(TAUS_M[0], "optimal")]["hmin"]
for tm in TAUS_M:
    print("  tau_motor=%.2f s:  %+8.3f m" % (tm, saved[(tm, "optimal")]["hmin"] - base))


# ───────────────────────── figura ─────────────────────────
SEL = [
    ("optimum, ideal engine",         (0.0, "optimal"), "#0072B2", "-"),
    ("optimum, engine tau=1.0 s",     (1.0, "optimal"), "#0072B2", "--"),
    ("CAA: DP de + 1 s, ideal engine", (0.0, "caa"),  "#D55E00", "-"),
    ("FAA: DP de + 1 s, ideal engine", (0.0, "faa"),  "#009E73", "-"),
]
SEL = [(n, k, c, l) for (n, k, c, l) in SEL if k in saved]

PAN = [("gamma", r"$\gamma$ (deg)", np.rad2deg),
       ("v_norm", r"$V/V_s$", lambda x: x),
       ("alpha", r"$\alpha$ (deg)", np.rad2deg),
       ("q", r"$q$ (deg/s)", np.rad2deg),
       ("de", r"$\delta_e$ (deg)", np.rad2deg),
       ("dt", r"$\delta_t$", lambda x: x),
       ("h", "altitude (m)", lambda x: x)]

fig, axes = plt.subplots(len(PAN), 1, figsize=(7.4, 12.2), sharex=True)
for ax, (k, et, cv) in zip(axes, PAN):
    for nom, key, col, ls in SEL:
        h = saved[key]["hist"]
        if k == "dt":
            ax.plot(h["t"], h["dt_cmd"], lw=1.0, ls=":", color=col, alpha=0.55,
                    zorder=2)
            ax.plot(h["t"], h["dt_ef"], lw=1.4, ls=ls, color=col, label=nom,
                    zorder=3)
        else:
            ax.plot(h["t"], cv(np.asarray(h[k])), lw=1.4, ls=ls, color=col,
                    label=nom, zorder=3)
    if k == "alpha":
        ax.axhline(14.0, color="0.5", lw=0.8, ls="--", zorder=1)
        ax.axhline(40.0, color="#b2182b", lw=0.8, ls="--", zorder=1)
        ax.annotate("crash: 40", xy=(0.99, 40.0),
                    xycoords=("axes fraction", "data"), ha="right", va="top",
                    fontsize=7, color="#b2182b")
    if k == "dt":
        ax.set_ylim(-0.02, 1.05)
        ax.annotate("dotted = commanded, solid = effective (engine)",
                    xy=(0.99, 0.32), xycoords=("axes fraction", "data"),
                    ha="right", fontsize=7.5, color="0.35")
    if k in ("gamma", "h"):
        ax.axhline(0.0, color="0.85", lw=0.6, zorder=0)
    ax.set_ylabel(et, fontsize=9); ax.grid(True, color="0.92", lw=0.5)
    ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"): ax.spines[sp].set_color("0.6")
    ax.tick_params(labelsize=8, color="0.6")
axes[-1].set_xlabel("tiempo (s)", fontsize=9)
axes[0].legend(loc="lower left", fontsize=7.5, frameon=False, ncol=2,
               bbox_to_anchor=(0.0, 1.02))
fig.suptitle(r"Pilot with %.1f s delay and first-order engine — $V_0=%.2f\,V_s$"
             % (TAU_H, V0), fontsize=10, y=0.988)
fig.tight_layout(rect=[0, 0, 1, 0.955])
out = main.RESULTS_DIR / ("piloto_realista_v%03d.png" % round(V0 * 100))
fig.savefig(out, dpi=200)
print("-> %s" % out)
