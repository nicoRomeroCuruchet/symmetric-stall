"""
fig_delta_trade.py — delta CL vs delta CD, parametric in alpha, from the
Riley (1985) Table III data at both propulsive conditions.

Each marker is one table segment: x = drag paid per extra degree of
alpha, y = lift gained per extra degree, both finite differences taken
between adjacent breakpoints and attributed to the segment midpoint
(labeled beside each marker). The dashed diagonal is the break-even
line delta CL = delta CD: above it an extra degree of alpha buys more
lift than it costs in drag; below it the trade has turned against the
pullout. The path crosses the diagonal at the stall segment with or
without power.

Same source data as paper_fig_alpha_trade.py; alpha restricted to
0..25 deg.

Output: fig_delta_trade.{png,pdf} in paths.out_dir() (+ manuscript
copy if stall-paper/img is present, as the other paper figures do).
"""
import logging
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from symmetric_stall import paths

logger = logging.getLogger(__name__)

ALPHA = np.array([0, 5, 10, 12, 14, 16, 18, 20, 25], dtype=float)
CL_O_CT05 = np.array([0.41, 0.97, 1.42, 1.54, 1.62, 1.67, 1.72, 1.76, 1.85])
CD_O_CT05 = np.array([-0.3474, -0.3139, -0.2483, -0.2057, -0.1435, -0.0709,
                      -0.0018, 0.0727, 0.2561])
CL_O_CT0 = np.array([0.41, 0.84, 1.16, 1.23, 1.26, 1.26, 1.26, 1.25, 1.22])
CD_O_CT0 = np.array([0.0526, 0.0846, 0.1456, 0.1856, 0.2446, 0.3136,
                     0.3786, 0.4486, 0.6186])

C_PWROFF, C_PWRON = "#2C4B9E", "#E8742A"

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

    fig, ax = plt.subplots(figsize=(6.4, 4.8))

    ax.plot(dcd05, dcl05, marker="o", ms=5, color=C_PWRON,
            label=r"$C_T = 0.5$")
    ax.plot(dcd0, dcl0, marker="o", ms=4, color=C_PWROFF, ls="--", lw=1.2,
            alpha=0.7, label=r"$C_T = 0$")

    # Break-even diagonal: lift gained per degree equals drag paid.
    lim = np.array([-0.03, 0.14])
    ax.plot(lim, lim, color="gray", lw=0.9, ls=":", label="break-even")
    ax.fill_between(lim, lim, lim[0], color="#C62828", alpha=0.05, zorder=0)

    # Label only the alphas that do not collide: 15..22.5 sit almost on
    # top of each other past the stall, so the cluster gets its two ends.
    offsets = {2.5: (7, 0), 7.5: (7, 2), 11: (7, 2), 13: (7, 2),
               15: (-10, 7), 22.5: (2, -12)}
    for x, y, m in zip(dcd05, dcl05, mid):
        if m in offsets:
            ax.annotate(rf"${m:g}°$", (x, y), textcoords="offset points",
                        xytext=offsets[m], fontsize=8, color=C_PWRON)
    for x, y, m in zip(dcd0, dcl0, mid):
        if m in offsets:
            dx, dy = offsets[m]
            ax.annotate(rf"${m:g}°$", (x, y), textcoords="offset points",
                        xytext=(dx, dy - 12 if m < 15 else dy), fontsize=8,
                        color=C_PWROFF)

    ax.set_xlabel(r"$\Delta C_D$ per degree of $\alpha$")
    ax.set_ylabel(r"$\Delta C_L$ per degree of $\alpha$")
    ax.set_xlim(-0.005, 0.048)
    ax.set_ylim(-0.02, 0.125)
    ax.legend(loc="upper right", fontsize=8, handlelength=1.8,
              borderpad=0.4, labelspacing=0.35)

    out_results = paths.out_dir()
    out_paper = Path("stall-paper/img")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_results / f"fig_delta_trade.{ext}", dpi=300,
                    bbox_inches="tight")
        if out_paper.is_dir():
            fig.savefig(out_paper / f"delta_trade.{ext}", dpi=300,
                        bbox_inches="tight")
        else:
            logger.info("[i] %s not present; manuscript copy skipped",
                        out_paper)
    plt.close(fig)
    logger.info("[+] fig_delta_trade.{png,pdf} written")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
