"""The scenario figure, in the EXACT style of paper 1.

Copies rc, layout, colours, line styles and panel labels from
`procedures.make_trajectory_comparison_figure`, so that the figures
nuevas entren en el paper sin retoques.

Uso: fig_paper.py <npz> <V0> [t_dp] [t_pil] [tau_m] [de_noreac] [de_pull]
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
import escenario as E          # reuses run() and the already-loaded parameters

RUNS = [("DP optimum (automatic, %.2f s)" % E.T_DP,      "#2C4B9E", "-",  "dp"),
        ("CAA pilot (reacts %.2f s)" % E.T_PIL,          "#E8742A", "--", "caa"),
        ("FAA pilot (reacts %.2f s)" % E.T_PIL,          "#2CA02C", "-.", "faa")]

datos = [(lab, col, ls, E.rodar(modo)) for lab, col, ls, modo in RUNS]

rc = {"font.family": "serif", "mathtext.fontset": "stix",
      "font.size": 12, "axes.labelsize": 13,
      "xtick.labelsize": 11, "ytick.labelsize": 11,
      "axes.spines.top": False, "axes.spines.right": False}

SIG = [("gamma",  r"$\gamma$ (deg)",  np.rad2deg, "plot"),
       ("v_norm", r"$V/V_s$ (--)",    None,       "plot"),
       ("alpha",  r"$\alpha$ (deg)",  np.rad2deg, "plot"),
       ("q",      r"$q$ (deg/s)",     np.rad2deg, "plot"),
       ("de",     r"$\delta_e$ (deg)", np.rad2deg, "step"),
       ("dt_ef",  r"$\delta_t$ (--)",  None,       "step"),
       ("h",      r"$\Delta h$ (m)",   None,       "plot")]

with plt.rc_context(rc):
    fig = plt.figure(figsize=(10.0, 6.8))
    gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1, 1.4],
                          hspace=0.42, wspace=0.24)
    slots = [gs[0, 0], gs[0, 1], gs[1, 0], gs[1, 1], gs[2, 0], gs[2, 1], gs[3, :]]
    axs = []
    t_end = max(r["H"]["t"][-1] for _, _, _, r in datos)
    for k, (slot, (key, ylab, conv, estilo)) in enumerate(zip(slots, SIG)):
        ax = fig.add_subplot(slot)
        for lab, col, ls, r in datos:
            t = np.asarray(r["H"]["t"]); y = np.asarray(r["H"][key])
            if conv is not None:
                y = conv(y)
            if estilo == "step":
                ancho = 2.6 if lab.startswith("FAA") else 1.6
                ax.step(t, y, color=col, lw=ancho, ls=ls, where="post",
                        label=lab, alpha=0.9 if ancho > 2 else 1.0,
                        zorder=2 if ancho > 2 else 3)
            else:
                ax.plot(t, y, color=col, lw=1.6, ls=ls, label=lab)
        ax.set_ylabel(ylab)
        ax.set_xlim(0.0, t_end)
        ax.grid(True, linestyle=":", alpha=0.55)
        ax.text(0.0, 1.02, "(%s)" % "abcdefg"[k], transform=ax.transAxes,
                fontsize=11, va="bottom", ha="left")
        axs.append(ax)

    # the two instants that structure the scenario
    for ax in axs:
        ax.axvline(E.T_DP,  color="0.45", linestyle=":", linewidth=1.0)
        ax.axvline(E.T_PIL, color="0.45", linestyle=":", linewidth=1.0)
    axs[0].annotate(r"$t_{det}$", xy=(E.T_DP, 0.06), xycoords=("data", "axes fraction"),
                    fontsize=9, ha="right", va="bottom", color="0.35",
                    xytext=(-3, 0), textcoords="offset points")
    axs[0].annotate(r"$t_{pilot}$", xy=(E.T_PIL, 0.06), xycoords=("data", "axes fraction"),
                    fontsize=9, ha="left", va="bottom", color="0.35",
                    xytext=(3, 0), textcoords="offset points")
    axs[2].axhline(14.0, color="0.45", linestyle="--", linewidth=1.0)

    axs[6].set_xlabel("$t$ (s)")
    axs[6].legend(loc="lower left", fontsize=10, frameon=False, ncol=3)

    fig.savefig(main.RESULTS_DIR / ("3_maniobras/fig_escenario_v%03d.pdf"
                                    % round(E.V0 * 100)), bbox_inches="tight")
    fig.savefig(main.RESULTS_DIR / ("3_maniobras/fig_escenario_v%03d.png"
                                    % round(E.V0 * 100)), dpi=300,
                bbox_inches="tight")
print("-> 3_maniobras/fig_escenario_v%03d.{png,pdf}" % round(E.V0 * 100))
for lab, _, _, r in datos:
    print("   %-34s h_min %+8.3f m   dur %5.2f s" % (lab, r["hmin"], r["t"]))
