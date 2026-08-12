"""Trajectory from the canonical IC: FILLED policy against UNFILLED.

For each requested V0 it generates:
  - one PNG per policy, in the same format as main.plot_time_response
  - one comparison PNG with the two overlaid

Uso:  python canonica_filled_vs_raw.py <raw.npz> <filled.npz> [V0 ...]
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

RAW = Path(sys.argv[1])
FILLED = Path(sys.argv[2])
VS = [float(v) for v in sys.argv[3:]] or [0.95, 0.85]

GAMMA0, ALPHA0, Q0 = 0.0, 20.0, 0.0
COLOR = {"sin rellenar": "#D55E00", "rellenada": "#0072B2"}

env = SymmetricStall()
politicas = {}
for etiqueta, ruta in (("sin rellenar", RAW), ("rellenada", FILLED)):
    politicas[etiqueta] = PolicyIterationStall.load(ruta, env=env)
    print("cargada %-13s %s" % (etiqueta, ruta.name))

PANELES = [("gamma", r"$\gamma$ (deg)", np.rad2deg),
           ("v_norm", r"$V/V_s$", lambda x: x),
           ("alpha", r"$\alpha$ (deg)", np.rad2deg),
           ("q", r"$q$ (deg/s)", np.rad2deg),
           ("de", r"$\delta_e$ (deg)", np.rad2deg),
           ("dt_ctrl", r"$\delta_t$", lambda x: x),
           ("h", "altitude (m)", lambda x: x)]

for v0 in VS:
    hists = {}
    print("\n=== IC canonica: gamma=%.0f, V=%.2f Vs, alpha=%.0f, q=%.0f ==="
          % (GAMMA0, v0, ALPHA0, Q0))
    for etiqueta, pi in politicas.items():
        h = main.run_dp_simulation(pi, GAMMA0, v0, ALPHA0, Q0)
        hists[etiqueta] = h
        hh = np.asarray(h["h"])
        print("  %-13s dur %6.2f s   dh %+8.3f m   h_min %+8.3f m   "
              "alpha_end %6.2f deg   mean thr %.3f"
              % (etiqueta, h["t"][-1], h["h"][-1], hh.min(),
                 np.rad2deg(h["alpha"][-1]), np.mean(h["dt_ctrl"])))
        main.plot_time_response(h, "canonica_%s_v%03d"
                                % (etiqueta.replace(" ", ""), round(v0 * 100)))

    # --- comparativo superpuesto ---
    fig, axes = plt.subplots(len(PANELES), 1, figsize=(7.2, 12.0), sharex=True)
    for ax, (clave, etiq, conv) in zip(axes, PANELES):
        for etiqueta, h in hists.items():
            ax.plot(h["t"], conv(np.asarray(h[clave])), lw=1.3,
                    color=COLOR[etiqueta], label=etiqueta, zorder=3)
        if clave == "alpha":
            ax.axhline(14.0, color="0.5", lw=0.8, ls="--", zorder=1)
            ax.annotate(r"$\alpha_s=14^\circ$", xy=(0.99, 14.0),
                        xycoords=("axes fraction", "data"), ha="right",
                        va="bottom", fontsize=7, color="0.35")
        if clave in ("gamma", "h"):
            ax.axhline(0.0, color="0.85", lw=0.6, zorder=0)
        if clave == "h":
            for etiqueta, h in hists.items():
                hh = np.asarray(h["h"]); i = int(np.argmin(hh))
                ax.plot(h["t"][i], hh[i], "o", ms=5, color=COLOR[etiqueta],
                        mec="white", mew=1.0, zorder=4)
                ax.annotate("%.2f m" % hh[i], xy=(h["t"][i], hh[i]),
                            xytext=(4, -10), textcoords="offset points",
                            fontsize=7.5, color=COLOR[etiqueta])
        ax.set_ylabel(etiq, fontsize=9)
        ax.grid(True, color="0.92", lw=0.5)
        ax.set_axisbelow(True)
        for lado in ("top", "right"):
            ax.spines[lado].set_visible(False)
        for lado in ("left", "bottom"):
            ax.spines[lado].set_color("0.6")
        ax.tick_params(labelsize=8, color="0.6")
    axes[-1].set_xlabel("time (s)", fontsize=9)
    axes[0].legend(loc="lower left", fontsize=8, frameon=False, ncol=2,
                   bbox_to_anchor=(0.0, 1.02))
    fig.suptitle(r"IC canonica $V_0=%.2f\,V_s$, $\alpha_0=20^\circ$: "
                 "filled vs unfilled policy" % v0, fontsize=10,
                 y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    out = main.RESULTS_DIR / ("comparacion_fill_v%03d.png" % round(v0 * 100))
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("  -> %s" % out)
