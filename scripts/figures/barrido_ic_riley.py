"""Familia de trayectorias con la politica de empuje de Riley.

Corre un abanico de entradas y superpone, para cada una, la rama de Riley
(politica reentrenada sobre la planta de Riley) contra la del paper 1
(politica publicada sobre la planta con K_t lineal). Las dos ramas resuelven
el mismo problema en cada IC, asi que la comparacion es legitima; lo que
cambia es cuanto empuje tiene el avion y, a traves de C_T, que tablas
aerodinamicas ve.

Usage:  PYTHONPATH=. python barrido_ic_riley.py [prefijo]
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

PREF = sys.argv[1] if len(sys.argv) > 1 else "barrido_riley"

# (gamma0, V0/Vs, alpha0) -- la primera es la canonica del paper 1
ICS = [
    (0.0, 0.95, 20.0),
    (-10.0, 0.95, 20.0),
    (-20.0, 0.95, 20.0),
    (-30.0, 0.95, 20.0),
    (-10.0, 0.90, 25.0),
    (-20.0, 1.05, 16.0),
]

RAMAS = [
    ("riley", "SymmetricStall_policy.npz", "-", "tab:blue"),
    ("paper1", "SymmetricStall_paper1_baseline.npz", "--", "tab:red"),
]


def vuela(npz, g0, v0, a0):
    env = SymmetricStall()
    pi = PolicyIterationStall.load(main.RESULTS_DIR / npz, env=env)
    _, states, _, _ = main.setup_symmetric_stall_experiment()
    pi.states_space = states
    return main.run_dp_simulation(pi, gamma_0_deg=g0, v_norm_0=v0,
                                  alpha_0_deg=a0, q_0_deg=0.0)


fig, axes = plt.subplots(len(ICS), 4, figsize=(15, 2.5 * len(ICS)),
                         sharex="col")
print("  IC (gamma0, V0, alpha0)        modelo    h_min      t_fin   recupera?")
for fila, (g0, v0, a0) in enumerate(ICS):
    for modo, npz, ls, color in RAMAS:
        os.environ["THRUST_MODEL"] = modo
        h = vuela(npz, g0, v0, a0)
        t = np.asarray(h["t"])
        g = np.rad2deg(h["gamma"])
        al = np.rad2deg(h["alpha"])
        H = np.asarray(h["h"])
        dt_c = np.asarray(h["dt_ctrl"])
        # la corrida termina sola al recuperar; si llega al tope de 15 s es que
        # nunca pico por debajo de -2 deg, o sea que no hubo divergencia
        recupera = t[-1] < 14.9
        print("  %+6.1f deg  %.2f Vs  %4.1f deg    %-7s  %+8.3f m  %6.2f s   %s  (throttle medio %.3f)"
              % (g0, v0, a0, modo, H.min(), t[-1],
                 "si" if recupera else "NO PICA", dt_c.mean()))

        # el throttle no se grafica: las dos ramas lo dejan pegado en 1.000 en
        # todas las entradas, asi que el unico control que decide es el elevador
        de = np.rad2deg(h["de"])
        for col, (y, lab) in enumerate(((g, r"$\gamma$ (deg)"),
                                        (al, r"$\alpha$ (deg)"),
                                        (de, r"$\delta_e$ (deg)"),
                                        (H, r"$\Delta h$ (m)"))):
            ax = axes[fila, col]
            ax.plot(t, y, ls, color=color, lw=1.6,
                    label=modo if (fila == 0 and col == 0) else None)
            if fila == len(ICS) - 1:
                ax.set_xlabel("t (s)")
            if fila == 0:
                ax.set_title(lab, fontsize=10)
    axes[fila, 0].set_ylabel(r"$\gamma_0=%+.0f^\circ$" % g0 + "\n"
                             + r"$%.2f V_s,\ \alpha_0=%.0f^\circ$" % (v0, a0),
                             fontsize=8)
    for col in range(4):
        axes[fila, col].grid(alpha=0.3, ls=":")
    axes[fila, 0].axhline(0.0, color="grey", lw=0.7, ls="--")
    axes[fila, 1].axhline(14.0, color="grey", lw=0.7, ls="--")

axes[0, 0].legend(fontsize=9, loc="best")
fig.tight_layout()
out = main.RESULTS_DIR / (PREF + ".png")
fig.savefig(out, dpi=140)
print("\nfigura: %s" % out)
