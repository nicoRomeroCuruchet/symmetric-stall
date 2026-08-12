"""Airspeed IC sweep: h_min(V0) for the filled and unfilled policies.

h_min = the minimum altitude excursion, which does NOT depend on the cut-off
criterion nor on the has_dived threshold. It also reports the minimum gamma, to
see where the threshold
-2 deg stops firing.

Uso: python barrido_v0.py <raw.npz> <filled.npz>
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

RAW, FILLED = Path(sys.argv[1]), Path(sys.argv[2])
V0S = np.round(np.arange(0.80, 0.9501, 0.01), 3)
GAMMA0, ALPHA0, Q0 = 0.0, 20.0, 0.0
COLOR = {"sin rellenar": "#D55E00", "rellenada": "#0072B2"}

env = SymmetricStall()
pis = {"sin rellenar": PolicyIterationStall.load(RAW, env=env),
       "rellenada": PolicyIterationStall.load(FILLED, env=env)}

res = {k: {"hmin": [], "gmin": [], "dur": [], "cerro": []} for k in pis}
print("%6s | %-32s | %-32s" % ("", "SIN RELLENAR", "RELLENADA"))
print("%6s | %9s %8s %7s %5s | %9s %8s %7s %5s" % (
    "V0/Vs", "h_min", "gamma_min", "dur", "cerro", "h_min", "gamma_min", "dur", "cerro"))
print("-" * 78)
for v0 in V0S:
    row = "%6.2f |" % v0
    for k, pi in pis.items():
        h = main.run_dp_simulation(pi, GAMMA0, float(v0), ALPHA0, Q0)
        hh = np.asarray(h["h"]); gg = np.rad2deg(np.asarray(h["gamma"]))
        cerro = h["t"][-1] < 14.9          # 14.99 s = agoto max_steps
        res[k]["hmin"].append(hh.min())
        res[k]["gmin"].append(gg.min())
        res[k]["dur"].append(h["t"][-1])
        res[k]["cerro"].append(cerro)
        row += " %9.3f %8.2f %7.2f %5s |" % (hh.min(), gg.min(), h["t"][-1],
                                              "si" if cerro else "NO")
    print(row)

# ---- figura ----
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.0, 6.4), sharex=True,
                               gridspec_kw={"height_ratios": [2, 1]})
for k in pis:
    ax1.plot(V0S, res[k]["hmin"], "o-", ms=4, lw=1.6, color=COLOR[k], label=k)
ax1.set_ylabel("excursion minima de altura  $h_{min}$  (m)", fontsize=9)
ax1.axvline(0.9124, color="0.5", lw=0.9, ls="--")
ax1.annotate(r"$V^*(20^\circ)=0.912$", xy=(0.9124, 0.02),
             xycoords=("data", "axes fraction"), rotation=90, fontsize=7.5,
             color="0.35", ha="right", va="bottom")
ax1.legend(fontsize=8, frameon=False, loc="lower right")

for k in pis:
    ax2.plot(V0S, res[k]["gmin"], "o-", ms=4, lw=1.6, color=COLOR[k])
ax2.axhline(-2.0, color="0.5", lw=0.9, ls="--")
ax2.annotate(r"umbral $has\_dived=-2^\circ$", xy=(0.801, -2.0), fontsize=7.5,
             color="0.35", va="bottom")
ax2.axvline(0.9124, color="0.5", lw=0.9, ls="--")
ax2.set_ylabel(r"$\gamma_{min}$ (deg)", fontsize=9)
ax2.set_xlabel(r"$V_0 / V_s$  de la condicion inicial", fontsize=9)

for ax in (ax1, ax2):
    ax.grid(True, color="0.92", lw=0.5); ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color("0.6")
    ax.tick_params(labelsize=8, color="0.6")

fig.suptitle(r"Sensibilidad a la IC: $\gamma_0=0$, $\alpha_0=20^\circ$, $q_0=0$",
             fontsize=10, y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = main.RESULTS_DIR / "barrido_v0.png"
fig.savefig(out, dpi=200)
print("\n-> %s" % out)

d = np.array(res["sin rellenar"]["hmin"]) - np.array(res["rellenada"]["hmin"])
print("the fill is worth: min %.3f m, max %.3f m, median %.3f m" % (
    d.min(), d.max(), float(np.median(d))))
