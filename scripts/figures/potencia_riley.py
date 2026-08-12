"""Mismo elevador (el del DP), distinta potencia, CON LA DINAMICA DE MOTOR DE RILEY.

Riley, Apendice A, ec. (A4):   delta_t = 1/(tau_e s + 1) * delta_t,c
o sea el empuje efectivo sigue al comando de palanca con un retardo de primer
orden. Riley NO publica tau_e (solo lo define en la lista de simbolos), asi que
se barre.

  palanca:  DP   rampa 0.6 s desde t=0
            CAA  rampa 2.0 s desde t=0
            FAA  rampa 2.0 s desde el nose-down (alpha < 14)
  motor:    d(dt_ef)/dt = (dt_cmd - dt_ef) / tau_e     en los TRES

Uso: potencia_riley.py <policy.npz> <V0> <tau_figura> [tau ...]
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
TAU_FIG = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
TAUS = [float(x) for x in sys.argv[4:]] or [0.0, 0.25, 0.5, 1.0]
if TAU_FIG not in TAUS:
    TAUS.append(TAU_FIG)
A0, ALPHA_UNSTALL, MAX_TIME = 20.0, np.deg2rad(14.0), 15.0

env = SymmetricStall()
pi = PolicyIterationStall.load(POLICY, env=env)
v_stall, dt = env.airplane.STALL_AIRSPEED, env.airplane.TIME_STEP

BRAZOS = [("DP   (palanca 0.6 s)",        0.6, "t0",      "#0072B2"),
          ("CAA  (palanca 2 s)",          2.0, "t0",      "#D55E00"),
          ("FAA  (palanca 2 s tras n-d)", 2.0, "unstall", "#009E73")]


def rodar(rampa, inicio, tau_e):
    obs, _ = env.specific_reset(0.0, V0, np.deg2rad(A0), 0.0)
    t, h, t_uns, dt_ef = 0.0, 0.0, None, 0.0
    stop, hs = RecoveryMonitor(dt), [0.0]
    H = {k: [] for k in ("t", "gamma", "v_norm", "alpha", "q", "de",
                         "dt_cmd", "dt_ef", "h")}
    while t < MAX_TIME:
        de = float(get_optimal_action(obs, pi)[0][0])
        if t_uns is None and obs[2] < ALPHA_UNSTALL:
            t_uns = t
        t_pwr = 0.0 if inicio == "t0" else t_uns
        cmd = 0.0 if t_pwr is None else float(np.clip((t - t_pwr) / rampa, 0.0, 1.0))
        dt_ef = cmd if tau_e <= 0 else dt_ef + (cmd - dt_ef) * (dt / tau_e)

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


print("IC: gamma=0, V=%.2f Vs, alpha=20 deg   |   elevador: el del DP en los tres\n"
      % V0)
print("%8s | %-22s | %-22s | %-22s" % ("tau_e", "DP", "CAA", "FAA"))
print("%8s | %9s %6s %5s | %9s %6s %5s | %9s %6s %5s" % (
    "(s)", "h_min", "dur", "est", "h_min", "dur", "est", "h_min", "dur", "est"))
print("-" * 84)
G = {}
for tau in sorted(TAUS):
    fila = "%8.2f |" % tau
    for nom, rampa, inicio, _ in BRAZOS:
        r = rodar(rampa, inicio, tau); G[(tau, nom)] = r
        fila += " %9.3f %5.2fs %5s |" % (r["hmin"], r["t"], r["est"][:5])
    print(fila)

print("\npenalizacion contra el DP, para cada motor:")
for tau in sorted(TAUS):
    d = G[(tau, BRAZOS[0][0])]["hmin"]
    print("  tau_e=%.2f s:  CAA %+8.3f m   FAA %+8.3f m   (FAA-CAA %+7.3f m)" % (
        tau, G[(tau, BRAZOS[1][0])]["hmin"] - d,
        G[(tau, BRAZOS[2][0])]["hmin"] - d,
        G[(tau, BRAZOS[2][0])]["hmin"] - G[(tau, BRAZOS[1][0])]["hmin"]))

# ───── figura con tau = TAU_FIG ─────
PAN = [("gamma", r"$\gamma$ (deg)", np.rad2deg), ("v_norm", r"$V/V_s$", lambda x: x),
       ("alpha", r"$\alpha$ (deg)", np.rad2deg), ("q", r"$q$ (deg/s)", np.rad2deg),
       ("de", r"$\delta_e$ (deg)", np.rad2deg), ("dt", r"$\delta_t$", lambda x: x),
       ("h", "altura (m)", lambda x: x)]
fig, axes = plt.subplots(len(PAN), 1, figsize=(7.4, 12.2), sharex=True)
for ax, (k, et, cv) in zip(axes, PAN):
    for nom, rampa, inicio, col in BRAZOS:
        H = G[(TAU_FIG, nom)]["H"]
        if k == "dt":
            ax.plot(H["t"], H["dt_cmd"], lw=1.0, ls=":", color=col, alpha=0.6, zorder=2)
            ax.plot(H["t"], H["dt_ef"], lw=1.5, color=col, label=nom, zorder=3)
        else:
            ax.plot(H["t"], cv(np.asarray(H[k])), lw=1.5, color=col, label=nom, zorder=3)
            if k == "h":
                y = np.asarray(H[k]); i = int(np.argmin(y))
                ax.plot(H["t"][i], y[i], "o", ms=5, color=col, mec="white", mew=1.0, zorder=4)
                ax.annotate("%.2f m" % y[i], xy=(H["t"][i], y[i]), xytext=(6, 0),
                            textcoords="offset points", fontsize=7.5, color=col, va="center")
    if k == "alpha":
        ax.axhline(14.0, color="0.5", lw=0.8, ls="--", zorder=1)
    if k == "dt":
        ax.set_ylim(-0.02, 1.05)
        ax.annotate("punteado = palanca,  lleno = motor (lag de Riley)",
                    xy=(0.99, 0.30), xycoords=("axes fraction", "data"),
                    ha="right", fontsize=7.5, color="0.35")
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
fig.suptitle(r"Motor de Riley (A4), $\tau_e=%.2f$ s — $V_0=%.2f\,V_s$, $\alpha_0=20^\circ$"
             % (TAU_FIG, V0), fontsize=10, y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.945])
out = main.RESULTS_DIR / ("potencia_riley_v%03d_tau%03d.png" % (round(V0*100), round(TAU_FIG*100)))
fig.savefig(out, dpi=200)
print("\n-> %s" % out)
