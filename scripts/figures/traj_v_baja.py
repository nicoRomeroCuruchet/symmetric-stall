"""Entrada canonica a una velocidad arbitraria, con C_T instrumentado.

    python traj_v_baja.py 0.50        (por defecto 0.50 Vs)

A velocidades muy por debajo de la perdida la presion dinamica se derrumba y
C_T = T/(q_bar S) se dispara, asi que el recorte a 0.5 de _compute_ct pasa a
estar activo. Cuando eso ocurre el estado esta fuera de dominio por cuatro
lados a la vez, y el panel de C_T es justamente el que lo delata; por eso se
grafica junto a los estados en vez de quedar en el log.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from symmetric_stall import train as main
from symmetric_stall.policy_iteration import PolicyIterationStall
from symmetric_stall.aircraft.symmetric_stall import SymmetricStall

V0 = float(sys.argv[1]) if len(sys.argv) > 1 else 0.50
RAMAS = [("riley", "SymmetricStall_policy.npz", "-", "tab:blue"),
         ("paper1", "SymmetricStall_paper1_baseline.npz", "--", "tab:red")]

fig, axes = plt.subplots(2, 3, figsize=(13, 6))
print("entrada canonica del paper 1 pero a %.2f Vs  (grilla: V en [0.90, 2.00] Vs)" % V0)
for modo, npz, ls, color in RAMAS:
    os.environ["THRUST_MODEL"] = modo
    env = SymmetricStall()
    pi = PolicyIterationStall.load(main.RESULTS_DIR / npz, env=env)
    _, states, _, _ = main.setup_symmetric_stall_experiment()
    pi.states_space = states
    h = main.run_dp_simulation(pi, gamma_0_deg=0.0, v_norm_0=V0,
                               alpha_0_deg=20.0, q_0_deg=0.0)

    t = np.asarray(h["t"])
    vn = np.asarray(h["v_norm"])
    # C_T reconstruido a lo largo de la trayectoria, con el mismo codigo que
    # usa la planta, para ver cuando el recorte de 0.5 esta mordiendo
    ct = np.array([env.airplane._compute_ct(d, v * env.airplane.STALL_AIRSPEED)
                   for d, v in zip(h["dt_ctrl"], vn)])
    fuera = 100.0 * np.mean(ct > 0.5)
    print("  %-7s h_min %+8.2f m  t_fin %5.2f s  V_min %.3f Vs  "
          "C_T max %.4f  fuera de tabla el %.0f%% del tiempo"
          % (modo, np.min(h["h"]), t[-1], vn.min(), ct.max(), fuera))

    for ax, y, lab in ((axes[0, 0], np.rad2deg(h["gamma"]), r"$\gamma$ (deg)"),
                       (axes[0, 1], np.rad2deg(h["alpha"]), r"$\alpha$ (deg)"),
                       (axes[0, 2], vn, r"$V/V_s$ (--)"),
                       (axes[1, 0], np.rad2deg(h["de"]), r"$\delta_e$ (deg)"),
                       (axes[1, 1], ct, r"$C_T$ (--)"),
                       (axes[1, 2], np.asarray(h["h"]), r"$\Delta h$ (m)")):
        ax.plot(t, y, ls, color=color, lw=1.7, label=modo)
        ax.set_title(lab, fontsize=10)
        ax.set_xlabel("t (s)")
        ax.grid(alpha=0.3, ls=":")

axes[0, 2].axhline(0.9, color="k", lw=0.9, ls=":")
axes[0, 2].annotate("borde de la grilla", (0.02, 0.9), fontsize=7,
                    xycoords=("axes fraction", "data"), va="bottom")
axes[1, 1].axhline(0.5, color="k", lw=0.9, ls=":")
axes[1, 1].annotate("fin de tabla (dCD_T actua arriba)", (0.02, 0.5), fontsize=7,
                    xycoords=("axes fraction", "data"), va="top")
axes[0, 1].axhline(14.0, color="grey", lw=0.7, ls="--")
axes[0, 0].legend(fontsize=9)
fig.suptitle(r"IC canonica a $%.2f\,V_s$ ($\alpha_0=20^\circ$, $\gamma_0=0$)  "
             "-- FUERA DE LA GRILLA" % V0, fontsize=11)
fig.tight_layout()
out = main.RESULTS_DIR / ("traj_v%03d.png" % round(V0 * 100))
fig.savefig(out, dpi=140)
print("figura: %s" % out)
