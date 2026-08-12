"""Cuanto de la ventaja del optimo sobrevive si el motor no puede dar un escalon.

La politica manda potencia maxima desde t=0 y la planta se la da instantanea.
Un motor a piston con helice de paso fijo no hace eso, y Riley lo sabe: la
ecuacion (A4) del Apendice A mete un retardo de primer orden entre la palanca
y el empuje,

    delta_t = 1/(tau_e s + 1) * delta_t,comandado

Lo que NO da el informe es el valor de tau_e -- figura en el glosario como
"engine-response time constant, sec" y en ningun lado mas. Por eso se barre.

El retardo se aplica en la evaluacion, no en el DP: la politica se resolvio
sin el, asi que esto mide que le pasa a una politica optimizada para un motor
ideal cuando vuela uno real. Es la pregunta conservadora, y la que importa,
porque es el numero que uno reporta.

    THRUST_MODEL=riley PYTHONPATH=. python traj_tau_motor.py [V0]
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from symmetric_stall import train as main
import paper_procedures as pp
from symmetric_stall.aircraft.symmetric_stall import SymmetricStall
from symmetric_stall.policy_iteration import PolicyIterationStall

V0 = float(sys.argv[1]) if len(sys.argv) > 1 else 0.85
ALPHA0 = 20.0
TAUS = [0.0, 0.3, 0.6, 1.0]
TAU_FIG = 0.6            # el tau que se dibuja en los paneles de trayectoria


def con_retardo(ctrl, tau, dt):
    """Envuelve un controlador con el retardo de motor de la ec. (A4).

    El estado del filtro vive en ctx, que rollout crea nuevo por corrida, asi
    que dos brazos no se contaminan. Arranca en 0: la entrada en perdida es
    con motor en ralenti, que es lo que suponen las dos maniobras guionadas.
    """
    if tau <= 0.0:
        return ctrl

    def nuevo(obs, t, opt, ctx):
        de, thr_cmd = ctrl(obs, t, opt, ctx)
        thr = ctx.get("_thr_motor", 0.0)
        thr += (float(thr_cmd) - thr) * (dt / tau)
        ctx["_thr_motor"] = thr
        return (de, thr)
    return nuevo


env = SymmetricStall()
DT = env.airplane.TIME_STEP
pi = PolicyIterationStall.load(main.RESULTS_DIR / "SymmetricStall_policy.npz",
                               env=env)
_, states, _, _ = main.setup_symmetric_stall_experiment()
pi.states_space = states

BRAZOS = [
    ("DP optimo", lambda: pp.ctrl_optimal, "tab:blue", "-"),
    ("CAA de-opt", lambda: pp.make_power_delay(0.0, ramp=True), "tab:green", "--"),
    ("FAA de-opt", lambda: pp.make_power_gated(ramp=True), "tab:purple", "-."),
]

print("entrada: alpha0 = %.0f deg, V0 = %.2f Vs, gamma0 = 0, q0 = 0" % (ALPHA0, V0))
print("retardo de motor de la ec. (A4), tau_e en segundos\n")
print("  tau_e      DP optimo        CAA de-opt       FAA de-opt     brecha opt->CAA")
tabla = {}
for tau in TAUS:
    fila = []
    for nombre, mk, _, _ in BRAZOS:
        r = pp.rollout(env, pi, con_retardo(mk(), tau, DT), ALPHA0, V0,
                       record=True)
        fila.append(r)
        tabla[(tau, nombre)] = r
    hs = [np.asarray(r["hist"]["h"]).min() for r in fila]
    print("  %4.2f s   %+8.2f m       %+8.2f m       %+8.2f m       %6.2f m (x%.2f)"
          % (tau, hs[0], hs[1], hs[2], hs[1] - hs[0], hs[1] / hs[0]))

fig, axes = plt.subplots(2, 3, figsize=(13.5, 6.2))
for nombre, _, color, ls in BRAZOS:
    for tau, lw, alpha in ((0.0, 1.1, 0.45), (TAU_FIG, 2.0, 1.0)):
        r = tabla[(tau, nombre)]
        h = r["hist"]
        t = np.asarray(h["t"])
        etiq = ("%s, $\\tau_e$=%.1f s" % (nombre, tau)) if tau else None
        for ax, y in ((axes[0, 0], np.rad2deg(h["gamma"])),
                      (axes[0, 1], np.rad2deg(h["alpha"])),
                      (axes[0, 2], np.asarray(h["v_norm"])),
                      (axes[1, 0], np.rad2deg(h["de"])),
                      (axes[1, 1], np.asarray(h["dt_ctrl"])),
                      (axes[1, 2], np.asarray(h["h"]))):
            ax.plot(t, y, ls, color=color, lw=lw, alpha=alpha, label=etiq)
            etiq = None
for ax, lab in ((axes[0, 0], r"$\gamma$ (deg)"), (axes[0, 1], r"$\alpha$ (deg)"),
                (axes[0, 2], r"$V/V_s$ (--)"), (axes[1, 0], r"$\delta_e$ (deg)"),
                (axes[1, 1], r"$\delta_t$ entregado al motor (--)"),
                (axes[1, 2], r"$\Delta h$ (m)")):
    ax.set_title(lab, fontsize=10)
    ax.set_xlabel("t (s)")
    ax.grid(alpha=0.3, ls=":")
axes[0, 1].axhline(14.0, color="grey", lw=0.8, ls="--")
axes[1, 2].axhline(0.0, color="grey", lw=0.8, ls="--")
axes[0, 0].legend(fontsize=8, loc="lower right")
fig.suptitle(r"Retardo de motor (A4) a $%.2f\,V_s$: fino $\tau_e=0$ (ideal), "
             r"grueso $\tau_e=%.1f$ s" % (V0, TAU_FIG), fontsize=11)
fig.tight_layout()
out = main.RESULTS_DIR / ("riley_tau_motor_v%03d.png" % round(V0 * 100))
fig.savefig(out, dpi=140)
print("\nfigura: %s" % out)
