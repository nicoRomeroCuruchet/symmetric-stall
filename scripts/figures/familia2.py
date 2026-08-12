"""Familia de trayectorias, politica rellenada. Uso: familia2.py <npz> <salida> V0..."""
import sys, logging
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
logging.disable(logging.INFO)
from symmetric_stall import train as main
from symmetric_stall.policy_iteration import PolicyIterationStall
from symmetric_stall.aircraft.symmetric_stall import SymmetricStall

pi = PolicyIterationStall.load(Path(sys.argv[1]), env=SymmetricStall())
NOMBRE = sys.argv[2]
V0S = [float(v) for v in sys.argv[3:]]
RAMPA = ["#c6dbef", "#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#084594"]
COLS = [RAMPA[round(i * (len(RAMPA) - 1) / max(1, len(V0S) - 1))]
        for i in range(len(V0S))]

PAN = [("gamma", r"$\gamma$ (deg)", np.rad2deg), ("v_norm", r"$V/V_s$", lambda x: x),
       ("alpha", r"$\alpha$ (deg)", np.rad2deg), ("q", r"$q$ (deg/s)", np.rad2deg),
       ("de", r"$\delta_e$ (deg)", np.rad2deg), ("dt_ctrl", r"$\delta_t$", lambda x: x),
       ("h", "altura (m)", lambda x: x)]

hs = {}
print("%6s %8s %10s %10s %9s" % ("V0", "dur", "h_min", "gamma_min", "alpha_min"))
for v0 in V0S:
    h = main.run_dp_simulation(pi, 0.0, v0, 20.0, 0.0); hs[v0] = h
    hh = np.asarray(h["h"])
    print("%6.2f %8.2f %+10.3f %+10.2f %9.2f" % (
        v0, h["t"][-1], hh.min(), np.rad2deg(min(h["gamma"])),
        np.rad2deg(min(h["alpha"]))))

fig, axes = plt.subplots(len(PAN), 1, figsize=(7.2, 12.0), sharex=True)
for ax, (k, et, cv) in zip(axes, PAN):
    for v0, col in zip(V0S, COLS):
        h = hs[v0]; y = cv(np.asarray(h[k]))
        ax.plot(h["t"], y, lw=1.4, color=col, zorder=3, label=r"$%.2f$" % v0)
        if k == "h":
            i = int(np.argmin(y))
            ax.plot(h["t"][i], y[i], "o", ms=4.5, color=col, mec="white",
                    mew=1.0, zorder=4)
    if k == "h":                       # una sola etiqueta por curva, a la derecha
        for v0, col in zip(V0S, COLS):
            y = np.asarray(hs[v0]["h"]); i = int(np.argmin(y))
            ax.annotate("%.2f" % y[i], xy=(hs[v0]["t"][i], y[i]), xytext=(6, 0),
                        textcoords="offset points", fontsize=7, color=col,
                        va="center")
    if k == "alpha":
        ax.axhline(14.0, color="0.5", lw=0.8, ls="--", zorder=1)
        ax.annotate(r"$\alpha_s=14^\circ$", xy=(0.99, 14.0),
                    xycoords=("axes fraction", "data"), ha="right", va="bottom",
                    fontsize=7, color="0.35")
    if k == "dt_ctrl":
        ax.set_ylim(-0.02, 1.05)
    if k in ("gamma", "h"):
        ax.axhline(0.0, color="0.85", lw=0.6, zorder=0)
    ax.set_ylabel(et, fontsize=9); ax.grid(True, color="0.92", lw=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color("0.6")
    ax.tick_params(labelsize=8, color="0.6")
axes[-1].set_xlabel("tiempo (s)", fontsize=9)
axes[0].legend(loc="lower left", fontsize=8, frameon=False, ncol=len(V0S),
               bbox_to_anchor=(0.0, 1.02), title=r"$V_0/V_s$",
               title_fontsize=8)
fig.suptitle(r"$\alpha_0=20^\circ$, $\gamma_0=0$, politica rellenada",
             fontsize=10, y=0.985)
fig.tight_layout(rect=[0, 0, 1, 0.965])
out = main.RESULTS_DIR / NOMBRE
fig.savefig(out, dpi=200); print("-> %s" % out)
