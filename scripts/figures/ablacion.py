"""Ablacion: cuanto de la penalizacion del procedimiento es el GATILLO tardio
y cuanto es la RAMPA de potencia.

El procedimiento de Gratton tiene dos diferencias con el optimo, y la tabla
CAA/FAA las mide juntas:

  1. el gatillo: suelta la picada cuando alpha < 14 (desestancado), mientras
     que el optimo la suelta con alpha ~ 16.7, o sea ANTICIPA;
  2. la potencia: rampa de 2 s, contra el escalon a full del optimo.

Se cruzan los dos factores. Uso: ablacion.py <policy.npz> <V0>
"""
import sys
import logging
from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from symmetric_stall import train as main

logging.disable(logging.INFO)

from symmetric_stall.policy_iteration import PolicyIterationStall
from symmetric_stall.aircraft.symmetric_stall import SymmetricStall
from paper_procedures import (rollout, ctrl_optimal, GRATTON_RAMP_S,
                              DE_DOWN, ALPHA_TARGET, K_ALPHA, K_Q)

POLICY, V0 = Path(sys.argv[1]), float(sys.argv[2])
A0 = float(sys.argv[3]) if len(sys.argv) > 3 else 20.0

env = SymmetricStall()
pi = PolicyIterationStall.load(POLICY, env=env)

r_opt = rollout(env, pi, ctrl_optimal, A0, V0, record=True)
CAP = float(np.min(r_opt["hist"]["de"]))


def piloto(alpha_gatillo_deg, potencia):
    """potencia: 'escalon' (full desde t=0) o 'rampa' (2 s desde t=0)."""
    a_gate = np.deg2rad(alpha_gatillo_deg)

    def ctrl(obs, t, opt, ctx):
        alpha, q = obs[2], obs[3]
        if "t_g" not in ctx and alpha < a_gate:
            ctx["t_g"] = t
        if "t_g" not in ctx:
            de = DE_DOWN
        else:
            de = float(np.clip(K_ALPHA * (alpha - ALPHA_TARGET) + K_Q * q,
                               CAP, DE_DOWN))
        thr = 1.0 if potencia == "escalon" else min(t / GRATTON_RAMP_S, 1.0)
        return (de, thr)
    return ctrl


print("IC: gamma=0, V=%.2f Vs, alpha=%.0f deg, q=0" % (V0, A0))
print("tirada del optimo: %.2f deg (acota todos los brazos)\n" % np.rad2deg(CAP))

brazos = [
    ("optimo (DP)",                    None),
    ("gatillo 14 + rampa   (= CAA)",   (14.0, "rampa")),
    ("gatillo 14 + escalon",           (14.0, "escalon")),
    ("gatillo 17 + rampa",             (17.0, "rampa")),
    ("gatillo 17 + escalon",           (17.0, "escalon")),
]

res = {}
print("%-30s %10s %8s %10s %11s" % ("brazo", "dh", "t_rec", "alpha_min", "estado"))
print("-" * 74)
for nom, cfg in brazos:
    r = r_opt if cfg is None else rollout(env, pi, piloto(*cfg), A0, V0, record=True)
    res[nom] = r
    print("%-30s %+10.3f %7.2fs %9.2f %11s" % (
        nom, r["h"], r["t"], np.rad2deg(np.min(r["hist"]["alpha"])), r["status"]))

base = res["optimo (DP)"]["h"]
caa = res["gatillo 14 + rampa   (= CAA)"]["h"]
print("\npenalizacion total del procedimiento: %+.3f m" % (caa - base))
print("\ndescomposicion (partiendo del CAA y arreglando un factor por vez):")
d_pot = res["gatillo 14 + escalon"]["h"] - caa
d_gat = res["gatillo 17 + rampa"]["h"] - caa
print("  arreglar SOLO la potencia (escalon en vez de rampa): %+8.3f m" % d_pot)
print("  arreglar SOLO el gatillo  (soltar en 17 en vez de 14): %+8.3f m" % d_gat)
print("  arreglar los DOS:                                     %+8.3f m"
      % (res["gatillo 17 + escalon"]["h"] - caa))
print("  (lo que falta para el optimo lo explica la forma del mando, no estos dos)")


# ───────────────────────── figura ─────────────────────────
ESTILO = {
    "optimo (DP)":                  ("#0072B2", "-",  1.9),
    "gatillo 14 + rampa   (= CAA)": ("#D55E00", "-",  1.4),
    "gatillo 17 + rampa":           ("#D55E00", ":",  1.4),
    "gatillo 14 + escalon":         ("#009E73", "-",  1.4),
    "gatillo 17 + escalon":         ("#009E73", ":",  1.4),
}
PAN = [("gamma", r"$\gamma$ (deg)", np.rad2deg), ("v_norm", r"$V/V_s$", lambda x: x),
       ("alpha", r"$\alpha$ (deg)", np.rad2deg), ("q", r"$q$ (deg/s)", np.rad2deg),
       ("de", r"$\delta_e$ (deg)", np.rad2deg), ("dt_ctrl", r"$\delta_t$", lambda x: x),
       ("h", "altura (m)", lambda x: x)]

fig, axes = plt.subplots(len(PAN), 1, figsize=(7.4, 12.2), sharex=True)
for ax, (k, et, cv) in zip(axes, PAN):
    for nom, (col, ls, lw) in ESTILO.items():
        h = res[nom]["hist"]
        ax.plot(h["t"], cv(np.asarray(h[k])), lw=lw, ls=ls, color=col,
                label=nom, zorder=3)
        if k == "h":
            y = cv(np.asarray(h[k])); i = int(np.argmin(y))
            ax.annotate("%.1f" % y[i], xy=(h["t"][i], y[i]), xytext=(6, 0),
                        textcoords="offset points", fontsize=7, color=col,
                        va="center")
    if k == "alpha":
        ax.axhline(14.0, color="0.5", lw=0.8, ls="--", zorder=1)
        ax.annotate(r"$\alpha_s=14^\circ$", xy=(0.99, 14.0),
                    xycoords=("axes fraction", "data"), ha="right", va="bottom",
                    fontsize=7, color="0.35")
    if k == "dt_ctrl":
        ax.set_ylim(-0.02, 1.05)
        ax.annotate("rampa 2 s vs escalon", xy=(0.99, 0.45),
                    xycoords=("axes fraction", "data"), ha="right",
                    fontsize=7.5, color="0.35")
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
fig.suptitle(r"Ablacion gatillo vs potencia — $V_0=%.2f\,V_s$, $\alpha_0=%.0f^\circ$"
             % (V0, A0), fontsize=10, y=0.988)
fig.tight_layout(rect=[0, 0, 1, 0.955])
out = main.RESULTS_DIR / ("ablacion_v%03d.png" % round(V0 * 100))
fig.savefig(out, dpi=200)
print("-> %s" % out)
