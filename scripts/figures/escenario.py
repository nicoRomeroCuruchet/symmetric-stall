"""Full scenario: a pilot who does not react, a detector, and trained pilots.

  t = 0        the aircraft stalls and the pilot does NOT react: he keeps
               PULLING (delta_e = DE_NOREAC, negative), which deepens the
               stall
  t = 0.33 s   the stall detector fires and engages the controller (DP).
               From there the DP commands the elevator continuously and brings up the
               throttle with a 0.6 s ramp
  t = 1.00 s   only then do the trained pilots react. Their elevator is
               ESCALONADO (heaviside), no modulado:
                   fase 1:  delta_e = +15 deg (picar a tope) hasta alpha < 14
                   fase 2:  delta_e = DE_PULL constante (tirar y sostener)
               CAA brings the throttle up over 2 s from reacting;
               FAA waits for the nose-down to start the ramp.

The engine carries Riley's first-order lag (A4) in all three arms.

Uso: escenario.py <npz> <V0> [t_dp] [t_pil] [tau_m] [de_noreac] [de_pull]
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

A = sys.argv
POLICY, V0 = Path(A[1]), float(A[2])
T_DP = float(A[3]) if len(A) > 3 else 0.33
T_PIL = float(A[4]) if len(A) > 4 else 1.00
TAU_M = float(A[5]) if len(A) > 5 else 0.50
DE_NOREAC = np.deg2rad(float(A[6]) if len(A) > 6 else -25.0)
DE_PULL = np.deg2rad(float(A[7]) if len(A) > 7 else -15.0)
# shape of the optimal elevator, measured on the policy itself:
#   +15 deg durante 0.19 s | pico -14.9 deg, dura 0.30 s | sostiene -6.2 deg
D_PUSH, D_PULL = 0.19, 0.30
DE_HOLD = np.deg2rad(-6.2)
DE_PUSH = np.deg2rad(15.0)
A0, ALPHA_UNSTALL, MAX_TIME = 20.0, np.deg2rad(14.0), 15.0

env = SymmetricStall()
pi = PolicyIterationStall.load(POLICY, env=env)
v_stall, dt = env.airplane.STALL_AIRSPEED, env.airplane.TIME_STEP


def run(mode):
    """mode: 'dp' | 'caa' | 'faa'."""
    t_det = T_DP if mode == "dp" else T_PIL
    obs, _ = env.specific_reset(0.0, V0, np.deg2rad(A0), 0.0)
    t, h, dt_ef, t_uns = 0.0, 0.0, 0.0, None
    stop, hs = RecoveryMonitor(dt), [0.0]
    H = {k: [] for k in ("t", "gamma", "v_norm", "alpha", "q", "de",
                         "dt_cmd", "dt_ef", "h")}
    while t < MAX_TIME:
        act = t >= t_det
        if not act:
            de = float(DE_NOREAC)                    # todavia tirando
        elif mode == "dp":
            de = float(get_optimal_action(obs, pi)[0][0])
        else:                                        # pilot: two steps
            dtr = t - t_det                      # three steps that follow
            if dtr < D_PUSH:                     # the optimum's shape
                de = float(DE_PUSH)
            elif dtr < D_PUSH + D_PULL:
                de = float(DE_PULL)
            else:
                de = float(DE_HOLD)

        if act and t_uns is None and obs[2] < ALPHA_UNSTALL:
            t_uns = t

        if not act:
            cmd = 0.0
        elif mode == "dp":
            cmd = float(np.clip((t - T_DP) / 0.6, 0.0, 1.0))
        elif mode == "caa":
            cmd = float(np.clip((t - T_PIL) / 2.0, 0.0, 1.0))
        else:
            cmd = 0.0 if t_uns is None else float(np.clip((t - t_uns) / 2.0, 0.0, 1.0))
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
    return dict(hmin=min(hs), t=t, est="sin cerrar", H=H, t_uns=t_uns)


BRAZOS = [("DP  automatico (det. %.2f s)" % T_DP,       "dp",  "#0072B2"),
          ("CAA pilot      (reacts %.2f s)" % T_PIL, "caa", "#D55E00"),
          ("FAA pilot      (reacts %.2f s)" % T_PIL, "faa", "#009E73")]

print("V=%.2f Vs, alpha0=20 deg   |   engine tau=%.2f s" % (V0, TAU_M))
print("not reacting: de=%.0f deg (PULLING)   |   pilot: +15 deg until the "
      "nose-down, then %.0f deg fixed\n"
      % (np.rad2deg(DE_NOREAC), np.rad2deg(DE_PULL)))
print("%-36s %9s %8s %10s %9s %11s" % ("arm", "h_min", "dur", "nose-down",
                                       "alpha_max", "status"))
print("-" * 88)
res = {}
for name, mode, col in BRAZOS:
    r = run(mode); res[name] = r
    print("%-36s %+9.3f %7.2fs %9.2fs %8.1f %11s" % (
        name, r["hmin"], r["t"], r["t_uns"] if r["t_uns"] else -1,
        np.rad2deg(max(r["H"]["alpha"])), r["est"]))
base = res[BRAZOS[0][0]]["hmin"]
print("\npenalty against the automatic controller:")
for name, _, _ in BRAZOS[1:]:
    print("  %-34s %+8.3f m" % (name, res[name]["hmin"] - base))

# ── figura ──
PAN = [("gamma", r"$\gamma$ (deg)", np.rad2deg), ("v_norm", r"$V/V_s$", lambda x: x),
       ("alpha", r"$\alpha$ (deg)", np.rad2deg), ("q", r"$q$ (deg/s)", np.rad2deg),
       ("de", r"$\delta_e$ (deg)", np.rad2deg), ("dt", r"$\delta_t$", lambda x: x),
       ("h", "altitude (m)", lambda x: x)]
fig, axes = plt.subplots(len(PAN), 1, figsize=(7.4, 12.2), sharex=True)
for ax, (k, et, cv) in zip(axes, PAN):
    for name, mode, col in BRAZOS:
        H = res[name]["H"]
        if k == "dt":
            ax.plot(H["t"], H["dt_cmd"], lw=1.0, ls=":", color=col, alpha=0.6, zorder=2)
            ax.plot(H["t"], H["dt_ef"], lw=1.5, color=col, label=name, zorder=3)
        else:
            ax.plot(H["t"], cv(np.asarray(H[k])), lw=1.5, color=col, label=name, zorder=3)
            if k == "h":
                y = np.asarray(H[k]); i = int(np.argmin(y))
                ax.plot(H["t"][i], y[i], "o", ms=5, color=col, mec="white", mew=1.0, zorder=4)
                ax.annotate("%.1f m" % y[i], xy=(H["t"][i], y[i]), xytext=(6, 0),
                            textcoords="offset points", fontsize=7.5, color=col, va="center")
    for tv in (T_DP, T_PIL):
        ax.axvline(tv, color="0.65", lw=0.8, ls=":", zorder=1)
    if k == "alpha":
        ax.axhline(14.0, color="0.5", lw=0.8, ls="--", zorder=1)
    if k == "de":
        ax.axhline(0.0, color="0.85", lw=0.6, zorder=0)
        ax.annotate("arriba = picar", xy=(0.99, 0.92), xycoords="axes fraction",
                    ha="right", fontsize=7, color="0.4")
    if k == "dt":
        ax.set_ylim(-0.02, 1.05)
    if k in ("gamma", "h"):
        ax.axhline(0.0, color="0.85", lw=0.6, zorder=0)
    ax.set_ylabel(et, fontsize=9); ax.grid(True, color="0.92", lw=0.5)
    ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"): ax.spines[sp].set_color("0.6")
    ax.tick_params(labelsize=8, color="0.6")
axes[-1].set_xlabel("time (s)", fontsize=9)
axes[0].legend(loc="lower left", fontsize=7.5, frameon=False, ncol=1,
               bbox_to_anchor=(0.0, 1.02))
fig.suptitle("Escenario completo: no-reaccion, detector a %.2f s, pilotos a %.2f s"
             "\n$V_0=%.2f\\,V_s$, stepped pilot following the DP shape" % (T_DP, T_PIL, V0),
             fontsize=10, y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.94])
out = main.RESULTS_DIR / ("3_maniobras/escenario_pd_v%03d.png" % round(V0 * 100))
fig.savefig(out, dpi=200)
print("\n-> %s" % out)
