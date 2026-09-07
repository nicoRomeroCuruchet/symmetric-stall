"""
paper_fig_riley_coeff_row.py — Fig: the four Riley (1985) coefficients
that drive the recovery physics, in a single row, at both propulsive
conditions (CT=0 power-off, CT=0.5 power-on).

Replaces the seven-panel table dump in the manuscript body (the
complete tables stay in the appendix for reproducibility). Layout per
the thesis directors: (a) CL and CD together, four lines, with and
without power; (b) Cm_o beside them, showing its thrust independence;
(c) pitch-rate damping Cm_q, doubling across the stall; (d) elevator
moment effectiveness Cm_de.

Color encodes the propulsive condition everywhere (blue power-off,
orange power-on); in panel (a) the linestyle separates the quantities
(solid lift, dashed axial force).

Data identical to the CUDA kernel tables and to
paper_fig_riley_coeffs.py.

Output: fig_riley_coeff_row.{png,pdf} in paths.out_dir() (+ manuscript
copy riley_coeff_row.{png,pdf} if stall-paper/img is present).
"""
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from symmetric_stall import paths

logger = logging.getLogger(__name__)

ALPHA = [-10, -5, 0, 5, 10, 12, 14, 16, 18, 20, 25, 30, 35, 40]

CL_O_CT0 = [-0.41, -0.01, 0.41, 0.84, 1.16, 1.23, 1.26, 1.26, 1.26, 1.25,
            1.22, 1.17, 1.13, 1.08]
CL_O_CT05 = [-0.67, -0.14, 0.41, 0.97, 1.42, 1.54, 1.62, 1.67, 1.72, 1.76,
             1.85, 1.92, 1.99, 2.05]
CD_O_CT0 = [0.0666, 0.0486, 0.0526, 0.0846, 0.1456, 0.1856, 0.2446, 0.3136,
            0.3786, 0.4486, 0.6186, 0.7786, 0.9255, 1.0636]
CD_O_CT05 = [-0.3273, -0.3494, -0.3474, -0.3139, -0.2483, -0.2057, -0.1435,
             -0.0709, -0.0018, 0.0727, 0.2561, 0.4322, 0.5979, 0.7572]
CM_O = [0.270, 0.158, 0.076, 0.002, -0.080, -0.118, -0.167, -0.225,
        -0.277, -0.316, -0.408, -0.480, -0.556, -0.606]
CM_Q_CT0 = [-7.00, -7.00, -7.04, -7.15, -7.52, -8.62, -10.80, -13.73,
            -15.38, -15.00, -14.66, -14.71, -14.77, -14.77]
CM_Q_CT05 = [-8.75, -8.75, -8.80, -9.36, -10.44, -12.64, -17.64, -18.54,
             -20.30, -19.85, -19.06, -17.80, -17.33, -16.88]
CM_DE_CT0 = [-1.105, -1.105, -1.105, -1.031, -0.945, -0.939, -0.933,
             -0.928, -0.928, -0.928, -0.928, -0.859, -0.745, -0.573]
CM_DE_CT05 = [-2.142, -2.250, -2.256, -2.262, -2.199, -2.062, -1.912,
              -1.781, -1.650, -1.541, -1.294, -1.220, -1.088, -0.859]

C_PWROFF, C_PWRON = "#2C4B9E", "#E8742A"
ALPHA_STALL = 14.0

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "stix", "font.size": 9,
    "axes.labelsize": 10, "legend.fontsize": 7,
    "axes.grid": True, "grid.alpha": 0.35, "lines.linewidth": 1.5,
})


def main():
    fig, axes = plt.subplots(1, 4, figsize=(13.0, 3.1))
    ax_a, ax_b, ax_c, ax_d = axes

    # (a) lift and net axial force, four lines
    ax_a.plot(ALPHA, CL_O_CT05, color=C_PWRON, ls="-", marker="s", ms=3,
              label=r"$C_{L_o}$, $C_T = 0.5$")
    ax_a.plot(ALPHA, CL_O_CT0, color=C_PWROFF, ls="-", marker="o", ms=3,
              label=r"$C_{L_o}$, $C_T = 0$")
    ax_a.plot(ALPHA, CD_O_CT05, color=C_PWRON, ls="--", marker="s", ms=3,
              label=r"$C_{D_o}$, $C_T = 0.5$")
    ax_a.plot(ALPHA, CD_O_CT0, color=C_PWROFF, ls="--", marker="o", ms=3,
              label=r"$C_{D_o}$, $C_T = 0$")
    ax_a.axhline(0.0, color="black", lw=0.7, alpha=0.6)
    ax_a.set_ylabel(r"$C_{L_o}$ (solid), $C_{D_o}$ (dashed)")
    ax_a.set_title("(a) Lift and net axial force", fontsize=9)
    ax_a.legend(loc="upper left", handlelength=1.6, borderpad=0.3,
                labelspacing=0.25)

    # (b) static pitching moment: identical columns at both conditions
    ax_b.plot(ALPHA, CM_O, color="#444444", marker="o", ms=3,
              label=r"$C_T = 0$ and $0.5$ (identical)")
    ax_b.set_ylabel(r"$C_{m_o}$")
    ax_b.set_title("(b) Static pitching moment", fontsize=9)
    ax_b.legend(loc="upper right", handlelength=1.6, borderpad=0.3)

    # (c) pitch-rate damping
    ax_c.plot(ALPHA, CM_Q_CT0, color=C_PWROFF, marker="o", ms=3,
              label=r"$C_T = 0$ (power-off)")
    ax_c.plot(ALPHA, CM_Q_CT05, color=C_PWRON, marker="s", ms=3,
              label=r"$C_T = 0.5$ (power-on)")
    ax_c.set_ylabel(r"$C_{m_{\hat{q}}}$")
    ax_c.set_title("(c) Pitch-rate damping", fontsize=9)
    ax_c.legend(loc="lower left", handlelength=1.6, borderpad=0.3)

    # (d) elevator moment effectiveness
    ax_d.plot(ALPHA, CM_DE_CT0, color=C_PWROFF, marker="o", ms=3,
              label=r"$C_T = 0$ (power-off)")
    ax_d.plot(ALPHA, CM_DE_CT05, color=C_PWRON, marker="s", ms=3,
              label=r"$C_T = 0.5$ (power-on)")
    ax_d.set_ylabel(r"$C_{m_{\delta_e}}$ (rad$^{-1}$)")
    ax_d.set_title("(d) Elevator effectiveness", fontsize=9)
    ax_d.legend(loc="lower right", handlelength=1.6, borderpad=0.3)

    for ax in axes:
        ax.axvline(ALPHA_STALL, color="gray", lw=0.9, ls=":")
        ax.set_xlabel(r"$\alpha$ (deg)")
        ax.set_xlim(-10, 40)

    fig.tight_layout()
    out_results = paths.out_dir()
    out_paper = Path("stall-paper/img")
    for ext in ("png", "pdf"):
        fig.savefig(out_results / f"fig_riley_coeff_row.{ext}", dpi=300,
                    bbox_inches="tight")
        if out_paper.is_dir():
            fig.savefig(out_paper / f"riley_coeff_row.{ext}", dpi=300,
                        bbox_inches="tight")
        else:
            logger.info("[i] %s not present; manuscript copy skipped",
                        out_paper)
    plt.close(fig)
    logger.info("[+] fig_riley_coeff_row.{png,pdf} written")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
