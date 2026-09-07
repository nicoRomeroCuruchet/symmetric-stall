"""
paper_fig_drag_polar.py — Fig: drag polar (CL_o vs CD_o) of the Riley
(1985) Table III data at both propulsive conditions, alpha annotated
along each curve.

The derived view behind the alpha-hold result: past the stall angle the
power-off polar goes flat-right (drag for zero lift), the power-on polar
keeps a quarter of its pre-stall slope, and the power-on net axial force
(thrust embedded in CD_o) crosses zero near 18 deg — beyond that the
lift gained is paid with the entire engine. The optimal pullout rides
the knee at ~13–14 deg.

Data identical to the CUDA kernel tables and to paper_fig_riley_coeffs.py;
alpha restricted to 0–25 deg (nothing new happens outside).

Output: fig_drag_polar.{png,pdf} in paths.out_dir() (+ manuscript copy if
stall-paper/img is present, as the other paper figures do).
"""
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from symmetric_stall import paths

logger = logging.getLogger(__name__)

# Riley Table III breakpoints, restricted to 0..25 deg
ALPHA = [0, 5, 10, 12, 14, 16, 18, 20, 25]
CL_O_CT0 = [0.41, 0.84, 1.16, 1.23, 1.26, 1.26, 1.26, 1.25, 1.22]
CL_O_CT05 = [0.41, 0.97, 1.42, 1.54, 1.62, 1.67, 1.72, 1.76, 1.85]
CD_O_CT0 = [0.0526, 0.0846, 0.1456, 0.1856, 0.2446, 0.3136, 0.3786,
            0.4486, 0.6186]
CD_O_CT05 = [-0.3474, -0.3139, -0.2483, -0.2057, -0.1435, -0.0709,
             -0.0018, 0.0727, 0.2561]

C_PWROFF, C_PWRON = "#2C4B9E", "#E8742A"
ALPHA_STALL = 14.0

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "stix", "font.size": 10,
    "axes.labelsize": 11, "legend.fontsize": 9,
    "axes.grid": True, "grid.alpha": 0.35, "lines.linewidth": 1.7,
})


def main():
    fig, ax = plt.subplots(figsize=(6.4, 4.8))

    ax.plot(CD_O_CT0, CL_O_CT0, marker="o", ms=4, color=C_PWROFF,
            label=r"$C_T = 0$ (power-off)")
    ax.plot(CD_O_CT05, CL_O_CT05, marker="s", ms=4, color=C_PWRON,
            label=r"$C_T = 0.5$ (power-on)")

    # Alpha labels along each curve. Power-off runs right-and-flat, so its
    # labels go below; power-on runs up-and-right, labels above-left.
    for a, x, y in zip(ALPHA, CD_O_CT0, CL_O_CT0):
        ax.annotate(f"{a:.0f}°", xy=(x, y), xytext=(2, -12),
                    textcoords="offset points", fontsize=8, color=C_PWROFF)
    for a, x, y in zip(ALPHA, CD_O_CT05, CL_O_CT05):
        ax.annotate(f"{a:.0f}°", xy=(x, y), xytext=(-15, 5),
                    textcoords="offset points", fontsize=8, color=C_PWRON)

    # Stall angle: filled, larger marker on both curves
    i_s = ALPHA.index(14)
    ax.plot(CD_O_CT0[i_s], CL_O_CT0[i_s], marker="o", ms=9, mfc=C_PWROFF,
            mec="black", zorder=5)
    ax.plot(CD_O_CT05[i_s], CL_O_CT05[i_s], marker="s", ms=9, mfc=C_PWRON,
            mec="black", zorder=5)

    # Net-axial-force sign: thrust is embedded in the power-on CD_o, so the
    # zero line splits net acceleration from net deceleration for that curve.
    ax.axvline(0.0, color="black", lw=0.9)
    ax.axvspan(0.0, 0.7, color="gray", alpha=0.08, zorder=0)
    ax.annotate("net axial force crosses\nzero at $\\alpha \\approx 18°$",
                xy=(CD_O_CT05[6], CL_O_CT05[6]), xytext=(0.06, 1.45),
                fontsize=8.5,
                arrowprops=dict(arrowstyle="->", lw=0.8, color="black"))

    # Power-off flat top
    ax.annotate("power-off flat top:\n$\\Delta C_L = 0$, 14°–18°",
                xy=(0.31, 1.26), xytext=(0.30, 0.95), fontsize=8.5,
                arrowprops=dict(arrowstyle="->", lw=0.8, color="black"))

    ax.set_xlabel(r"$C_{D_o}$ (net axial force, thrust embedded)")
    ax.set_ylabel(r"$C_{L_o}$")
    ax.set_xlim(-0.42, 0.68)
    ax.set_ylim(0.3, 2.0)
    ax.legend(loc="upper left")

    out_results = paths.out_dir()
    out_paper = Path("stall-paper/img")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_results / f"fig_drag_polar.{ext}", dpi=300,
                    bbox_inches="tight")
        if out_paper.is_dir():
            fig.savefig(out_paper / f"drag_polar.{ext}", dpi=300,
                        bbox_inches="tight")
        else:
            logger.info("[i] %s not present; manuscript copy skipped",
                        out_paper)
    plt.close(fig)
    logger.info("[+] fig_drag_polar.{png,pdf} written")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
