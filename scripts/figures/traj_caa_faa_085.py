"""Optimo DP contra CAA y FAA, entrada canonica a 0.85 Vs, empuje de Riley.

Reusa las maniobras guionadas de paper_procedures.py -- no las reescribe --
para que la comparacion sea la misma del paper 1 y lo unico que cambie sea el
modelo de empuje y la grilla.

    CAA   morro abajo y potencia AL MISMO TIEMPO (rampa de 2 s desde t=0)
    FAA   morro abajo primero, potencia recien al des-entrar en perdida

Ambos brazos se acotan a la MISMA autoridad de tirada que alcanza el optimo en
esta entrada, como hace run_maneuvers: sin eso la comparacion mezclaria CUANDO
se aplica potencia con CUANTO tira el piloto.

    THRUST_MODEL=riley PYTHONPATH=. python traj_caa_faa_085.py [V0]
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from symmetric_stall import train as main
from symmetric_stall import procedures as pp
from symmetric_stall.aircraft.symmetric_stall import SymmetricStall
from symmetric_stall.policy_iteration import PolicyIterationStall

V0 = float(sys.argv[1]) if len(sys.argv) > 1 else 0.85
ALPHA0 = 20.0

env = SymmetricStall()
pi = PolicyIterationStall.load(main.RESULTS_DIR / "SymmetricStall_policy.npz",
                               env=env)
_, states, _, _ = main.setup_symmetric_stall_experiment()
pi.states_space = states

# autoridad de tirada del optimo en ESTA entrada, para acotar los brazos
r_opt = pp.rollout(env, pi, pp.ctrl_optimal, ALPHA0, V0, record=True)
tope = float(np.min(r_opt["hist"]["de"]))
print("entrada: alpha0 = %.0f deg, V0 = %.2f Vs, gamma0 = 0, q0 = 0" % (ALPHA0, V0))
print("tirada mas profunda del optimo: %.2f deg (se usa como tope de CAA y FAA)"
      % np.rad2deg(tope))

# Dos familias.
#
# Las guionadas son las de Gratton tal cual: empujan a +15 hasta cruzar 14 deg
# y despues sostienen alpha con una ley proporcional. Su elevador no se parece
# en nada al del optimo, asi que su diferencia contra el optimo mezcla DOS
# decisiones -- cuando entra la potencia y como se maneja el morro.
#
# Las de "delta_e optimo" toman el elevador del propio optimo y solo cambian
# el momento de la potencia. Aislan la decision que CAA y FAA realmente
# discuten, que es esa y no el pitcheo.
BRAZOS = [
    ("DP optimo", r_opt, "tab:blue", "-"),
    ("CAA de-opt", pp.rollout(env, pi, pp.make_power_delay(0.0, ramp=True),
                              ALPHA0, V0, record=True), "tab:green", "--"),
    ("FAA de-opt", pp.rollout(env, pi, pp.make_power_gated(ramp=True),
                              ALPHA0, V0, record=True), "tab:purple", "--"),
    ("CAA guionada", pp.rollout(env, pi, pp.make_maneuver("t0", "alpha_hold", tope),
                                ALPHA0, V0, record=True), "tab:orange", "-."),
    ("FAA guionada", pp.rollout(env, pi, pp.make_maneuver("unstall", "alpha_hold", tope),
                                ALPHA0, V0, record=True), "tab:red", "-."),
]

fig, axes = plt.subplots(2, 3, figsize=(13.5, 6.2))
print()
base = None
for nombre, r, color, ls in BRAZOS:
    h = r["hist"]
    t = np.asarray(h["t"])
    H = np.asarray(h["h"])
    if base is None:
        base = H.min()
    print("  %-13s h_min %+8.2f m   t %5.2f s   %-10s   gamma_min %+6.2f deg"
          "   %s" % (nombre, H.min(), r["t"], r["status"],
                     np.rad2deg(h["gamma"]).min(),
                     "" if nombre == "DP optimo"
                     else "x%.2f del optimo" % (H.min() / base)))
    for ax, y, lab in ((axes[0, 0], np.rad2deg(h["gamma"]), r"$\gamma$ (deg)"),
                       (axes[0, 1], np.rad2deg(h["alpha"]), r"$\alpha$ (deg)"),
                       (axes[0, 2], np.asarray(h["v_norm"]), r"$V/V_s$ (--)"),
                       (axes[1, 0], np.rad2deg(h["de"]), r"$\delta_e$ (deg)"),
                       (axes[1, 1], np.asarray(h["dt_ctrl"]), r"$\delta_t$ (--)"),
                       (axes[1, 2], H, r"$\Delta h$ (m)")):
        ax.plot(t, y, ls, color=color, lw=1.7, label=nombre)
        ax.set_title(lab, fontsize=10)
        ax.set_xlabel("t (s)")
        ax.grid(alpha=0.3, ls=":")

axes[0, 1].axhline(14.0, color="grey", lw=0.8, ls="--")
axes[1, 2].axhline(0.0, color="grey", lw=0.8, ls="--")
axes[0, 0].legend(fontsize=9)
fig.suptitle(r"Entrada canonica a $%.2f\,V_s$ ($\alpha_0=%.0f^\circ$, "
             r"$\gamma_0=0$): optimo DP vs CAA vs FAA, empuje de Riley"
             % (V0, ALPHA0), fontsize=11)
fig.tight_layout()
out = main.RESULTS_DIR / ("riley_caa_faa_v%03d.png" % round(V0 * 100))
fig.savefig(out, dpi=140)
print("\nfigura: %s" % out)
