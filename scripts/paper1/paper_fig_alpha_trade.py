"""
paper_fig_alpha_trade.py — Fig: the marginal lift/drag trade vs alpha in
the Riley (1985) Table III data, at both propulsive conditions.

Each point is the finite difference between two adjacent table
breakpoints, plotted at the segment midpoint (the two-point slope
estimate belongs to the segment, not to either endpoint). Green = lift
gained per extra degree of alpha, red = drag paid per extra degree; the
curves cross at the stall angle. The figure behind the alpha-hold
result: past ~14 deg each degree costs more drag than the lift it buys,
with or without power, even though the power-on CL itself keeps rising.

Data identical to the CUDA kernel tables and to paper_fig_riley_coeffs.py;
alpha restricted to 0..25 deg (nothing new happens outside).

Output: fig_alpha_trade.{png,pdf} in paths.out_dir() (+ manuscript copy
if stall-paper/img is present, as the other paper figures do).
"""
import logging
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from symmetric_stall import paths

logger = logging.getLogger(__name__)

# Riley Table III breakpoints, both CT columns, restricted to 0..25 deg
ALPHA = np.array([0, 5, 10, 12, 14, 16, 18, 20, 25], dtype=float)
CL_O_CT05 = np.array([0.41, 0.97, 1.42, 1.54, 1.62, 1.67, 1.72, 1.76, 1.85])
CD_O_CT05 = np.array([-0.3474, -0.3139, -0.2483, -0.2057, -0.1435, -0.0709,
                      -0.0018, 0.0727, 0.2561])
CL_O_CT0 = np.array([0.41, 0.84, 1.16, 1.23, 1.26, 1.26, 1.26, 1.25, 1.22])
CD_O_CT0 = np.array([0.0526, 0.0846, 0.1456, 0.1856, 0.2446, 0.3136,
                     0.3786, 0.4486, 0.6186])

C_LIFT, C_DRAG = "#2E7D32", "#C62828"
ALPHA_STALL = 14.0

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "stix", "font.size": 10,
    "axes.labelsize": 11, "legend.fontsize": 9,
    "axes.grid": True, "grid.alpha": 0.35, "lines.linewidth": 1.7,
})


def main():
    mid = 0.5 * (ALPHA[1:] + ALPHA[:-1])
    dcl05 = np.diff(CL_O_CT05) / np.diff(ALPHA)
    dcd05 = np.diff(CD_O_CT05) / np.diff(ALPHA)
    dcl0 = np.diff(CL_O_CT0) / np.diff(ALPHA)
    dcd0 = np.diff(CD_O_CT0) / np.diff(ALPHA)

    fig, ax = plt.subplots(figsize=(6.4, 4.4))

    ax.plot(mid, dcl05, marker="s", ms=5, color=C_LIFT,
            label=r"lift gained / deg, $C_T = 0.5$")
    ax.plot(mid, dcd05, marker="^", ms=5, color=C_DRAG,
            label=r"drag paid / deg, $C_T = 0.5$")
    ax.plot(mid, dcl0, marker="s", ms=4, color=C_LIFT, ls="--", lw=1.2,
            alpha=0.55, label=r"lift gained / deg, $C_T = 0$")
    ax.plot(mid, dcd0, marker="^", ms=4, color=C_DRAG, ls="--", lw=1.2,
            alpha=0.55, label=r"drag paid / deg, $C_T = 0$")

    # The crossing of the two power-on piecewise-linear curves computes
    # to 13.9 deg, a hair below the 14-deg table breakpoint because the
    # finite differences live at segment midpoints; the label states the
    # table's stall angle, which is where the dotted line is drawn.
    ax.axvline(ALPHA_STALL, color="gray", lw=0.9, ls=":")
    ax.axvspan(ALPHA_STALL, 25.0, color=C_DRAG, alpha=0.06, zorder=0)
    ax.text(14.4, -0.017, rf"$\alpha_s = {ALPHA_STALL:.0f}°$", fontsize=9,
            color="gray")

    ax.set_xlabel(r"$\alpha$ (deg)")
    ax.set_ylabel(r"marginal change per degree of $\alpha$")
    ax.set_xlim(0, 25)
    ax.set_ylim(-0.025, 0.125)
    ax.legend(loc="upper right", fontsize=8, handlelength=1.8,
              borderpad=0.4, labelspacing=0.35)

    out_results = paths.out_dir()
    out_paper = Path("stall-paper/img")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_results / f"fig_alpha_trade.{ext}", dpi=300,
                    bbox_inches="tight")
        if out_paper.is_dir():
            fig.savefig(out_paper / f"alpha_trade.{ext}", dpi=300,
                        bbox_inches="tight")
        else:
            logger.info("[i] %s not present; manuscript copy skipped",
                        out_paper)
    plt.close(fig)
    logger.info("[+] fig_alpha_trade.{png,pdf} written")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
