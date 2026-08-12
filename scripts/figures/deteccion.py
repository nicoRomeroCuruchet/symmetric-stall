"""Sistema automatico contra piloto humano: cada uno con su retardo de DETECCION.

Antes de detectar, nadie toca nada (de = 0, palanca = 0): el avion entra en
perdida y cae. Al detectar, cada brazo empieza a actuar:

  DP   detecta a los 0.33 s  (el detector de Bunge con sensores de angulo de
       flujo) y desde ahi ejecuta la politica en tiempo real, palanca a full
       con rampa de 0.6 s
  CAA  detecta a los TAU_H s (piloto) y sube la palanca en 2 s desde ese
       instante
  FAA  detecta a los TAU_H s pero espera al nose-down (alpha < 14) para
       empezar la rampa de 2 s

El delta_e sale de la POLITICA en los tres, una vez que cada uno detecto: lo
unico que los distingue es CUANDO empiezan y como suben la potencia.

El motor tiene el retardo de primer orden de Riley (A4) en los tres.

Uso: deteccion.py <policy.npz> <V0> [tau_dp] [tau_humano] [tau_motor]
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
from paper_procedures import RecoveryMonitor

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

BRAZOS = [
    ("DP  automatico  (det. %.2f s, rampa 0.6 s)" % TAU_DP,
     TAU_DP, 0.6, "inmediata", "#0072B2"),
    ("CAA piloto      (det. %.2f s, rampa 2 s)" % TAU_H,
     TAU_H, 2.0, "inmediata", "#D55E00"),
    ("FAA piloto      (det. %.2f s, rampa 2 s tras n-d)" % TAU_H,
     TAU_H, 2.0, "tras_nose_down", "#009E73"),
]


def rodar(t_det, rampa, cuando):
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
            if cuando == "inmediata":
                t_pwr = t_det
            elif t_uns is not None:
                t_pwr = t_uns
            cmd = 0.0 if t_pwr is None else float(
                np.clip((t - t_pwr) / rampa, 0.0, 1.0))

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


print("IC: gamma=0, V=%.2f Vs, alpha=20 deg    motor tau_e=%.2f s" % (V0, TAU_M))
print("antes de detectar: de=%.1f deg (el piloto TIRA, no reacciono), palanca=0\n"
      % np.rad2deg(DE_NOREAC))
print("%-46s %9s %8s %10s %11s" % ("brazo", "h_min", "dur", "nose-down", "estado"))
print("-" * 90)
res = {}
for nom, t_det, rampa, cuando, col in BRAZOS:
    r = rodar(t_det, rampa, cuando); res[nom] = r
    print("%-46s %+9.3f %7.2fs %9.2fs %11s" % (
        nom, r["hmin"], r["t"], r["t_uns"] if r["t_uns"] else -1, r["est"]))

base = res[BRAZOS[0][0]]["hmin"]
print("\npenalizacion contra el sistema automatico:")
for nom, *_ in BRAZOS[1:]:
    print("  %-44s %+8.3f m" % (nom, res[nom]["hmin"] - base))

# ───── figura ─────
PAN = [("gamma", r"$\gamma$ (deg)", np.rad2deg), ("v_norm", r"$V/V_s$", lambda x: x),
       ("alpha", r"$\alpha$ (deg)", np.rad2deg), ("q", r"$q$ (deg/s)", np.rad2deg),
       ("de", r"$\delta_e$ (deg)", np.rad2deg), ("dt", r"$\delta_t$", lambda x: x),
       ("h", "altura (m)", lambda x: x)]
fig, axes = plt.subplots(len(PAN), 1, figsize=(7.4, 12.2), sharex=True)
for ax, (k, et, cv) in zip(axes, PAN):
    for nom, t_det, rampa, cuando, col in BRAZOS:
        H = res[nom]["H"]
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
    for tv, lab in ((TAU_DP, "detecta el sistema"), (TAU_H, "detecta el piloto")):
        ax.axvline(tv, color="0.65", lw=0.8, ls=":", zorder=1)
    if k == "alpha":
        ax.axhline(14.0, color="0.5", lw=0.8, ls="--", zorder=1)
    if k == "dt":
        ax.set_ylim(-0.02, 1.05)
        ax.annotate("punteado = palanca, lleno = motor", xy=(0.99, 0.28),
                    xycoords=("axes fraction", "data"), ha="right", fontsize=7.5,
                    color="0.35")
    if k == "gamma":
        ax.annotate("%.2f s\ndetecta\nel sistema" % TAU_DP, xy=(TAU_DP, 0.02),
                    xycoords=("data", "axes fraction"), fontsize=6.5, color="0.4",
                    ha="center", va="bottom")
        ax.annotate("%.2f s\ndetecta\nel piloto" % TAU_H, xy=(TAU_H, 0.02),
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
fig.suptitle(r"Deteccion: sistema (%.2f s) vs piloto (%.2f s), motor $\tau_e=%.2f$ s"
             "\n$V_0=%.2f\\,V_s$, sin reaccionar $\\delta_e=%.0f^\\circ$" % (TAU_DP, TAU_H, TAU_M, V0, np.rad2deg(DE_NOREAC)), fontsize=10, y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.94])
out = main.RESULTS_DIR / ("3_maniobras/deteccion_v%03d_de%+03d.png" % (round(V0*100), round(np.rad2deg(DE_NOREAC))))
fig.savefig(out, dpi=200)
print("\n-> %s" % out)
