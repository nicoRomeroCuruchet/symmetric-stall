"""Family of 0.91-0.94 Vs trajectories, filled policy, overlaid."""
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
V0S = [0.91, 0.92, 0.93, 0.94]
RAMPA = ["#9ecae1", "#6baed6", "#3182bd", "#08519c"]   # secuencial, L monotona

PAN = [("gamma", r"$\gamma$ (deg)", np.rad2deg), ("v_norm", r"$V/V_s$", lambda x: x),
       ("alpha", r"$\alpha$ (deg)", np.rad2deg), ("q", r"$q$ (deg/s)", np.rad2deg),
       ("de", r"$\delta_e$ (deg)", np.rad2deg), ("dt_ctrl", r"$\delta_t$", lambda x: x),
       ("h", "altitude (m)", lambda x: x)]

hs = {}
for v0 in V0S:
    hs[v0] = main.run_dp_simulation(pi, 0.0, v0, 20.0, 0.0)
    hh = np.asarray(hs[v0]["h"])
    print("  V0=%.2f  dur %6.2f s   h_min %+8.3f m" % (v0, hs[v0]["t"][-1], hh.min()))

fig, axes = plt.subplots(len(PAN), 1, figsize=(7.2, 12.0), sharex=True)
for ax, (k, et, cv) in zip(axes, PAN):
    for v0, col in zip(V0S, RAMPA):
        h = hs[v0]; y = cv(np.asarray(h[k]))
        ax.plot(h["t"], y, lw=1.4, color=col, zorder=3,
                label=r"$V_0=%.2f\,V_s$" % v0)
        if k == "h":                      # etiqueta directa en la excursion minima
            i = int(np.argmin(y))
            ax.plot(h["t"][i], y[i], "o", ms=4.5, color=col, mec="white", mew=1.0, zorder=4)
            ax.annotate("%.2f" % y[i], xy=(h["t"][i], y[i]), xytext=(5, -9 - 11 * V0S.index(v0)),
                        textcoords="offset points", fontsize=7.5, color=col)
    if k == "alpha":
        ax.axhline(14.0, color="0.5", lw=0.8, ls="--", zorder=1)
        ax.annotate(r"$\alpha_s=14^\circ$", xy=(0.99, 14.0),
                    xycoords=("axes fraction", "data"), ha="right", va="bottom",
                    fontsize=7, color="0.35")
    if k in ("gamma", "h"):
        ax.axhline(0.0, color="0.85", lw=0.6, zorder=0)
    if k == "dt_ctrl":
        ax.set_ylim(-0.02, 1.05)          # otherwise matplotlib zooms into the ULP
        ax.annotate("1.000 in all four", xy=(0.99, 1.0),
                    xycoords=("axes fraction", "data"), ha="right", va="top",
                    fontsize=7.5, color="0.35")
    ax.set_ylabel(et, fontsize=9); ax.grid(True, color="0.92", lw=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color("0.6")
    ax.tick_params(labelsize=8, color="0.6")
axes[-1].set_xlabel("tiempo (s)", fontsize=9)
axes[0].legend(loc="lower left", fontsize=8, frameon=False, ncol=4,
               bbox_to_anchor=(0.0, 1.02))
fig.suptitle(r"Marginal entries: $\alpha_0=20^\circ$, filled policy",
             fontsize=10, y=0.985)
fig.tight_layout(rect=[0, 0, 1, 0.965])
out = main.RESULTS_DIR / "familia_v091_v094.png"
fig.savefig(out, dpi=200); print("-> %s" % out)
