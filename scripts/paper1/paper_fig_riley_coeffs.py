"""
paper_fig_riley_coeffs.py — Fig: ALL seven Riley (1985) Table III
coefficients vs alpha at both propulsive conditions (CT=0 power-off,
CT=0.5 power-on) — one panel per appendix table.

Layout 4+3: top row lift family + axial force (CL_o, CL_q, CL_de, CD_o),
bottom row moment family (Cm_o, Cm_q, Cm_de). Cm_o is CT-independent
(identical columns in Riley Table IIIc) — single curve, annotated.

Key physics highlighted in the caption: (a) power-off flat-top plateau vs
power-on monotone growth (slipstream lift), (d) negative power-on values =
net axial force (embedded thrust), (f) pitch damping doubling across stall.

Data identical to the CUDA kernel tables (PolicyIteration.py) and the
transcription in the paper appendix.

Output: stall-paper/img/riley_coefficients.{png,pdf} (+ results/paper copy)
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
CM_Q_CT0 = [-7.00, -7.00, -7.04, -7.15, -7.52, -8.62, -10.80, -13.73,
            -15.38, -15.00, -14.66, -14.71, -14.77, -14.77]
CM_Q_CT05 = [-8.75, -8.75, -8.80, -9.36, -10.44, -12.64, -17.64, -18.54,
             -20.30, -19.85, -19.06, -17.80, -17.33, -16.88]
CL_Q_CT0 = [2.41, 2.41, 2.42, 2.46, 2.59, 2.96, 3.72, 4.73, 5.29, 5.16,
            5.05, 5.06, 5.98, 5.08]
CL_Q_CT05 = [3.01, 3.01, 3.03, 3.22, 3.59, 4.35, 6.07, 6.38, 6.99, 6.83,
             6.56, 6.13, 5.97, 5.81]
CL_DE_CT0 = [0.355, 0.361, 0.355, 0.332, 0.304, 0.292, 0.286, 0.281,
             0.275, 0.269, 0.252, 0.241, 0.223, 0.212]
CL_DE_CT05 = [0.796, 0.779, 0.750, 0.705, 0.624, 0.595, 0.578, 0.561,
              0.538, 0.515, 0.458, 0.418, 0.349, 0.315]
CM_O = [0.270, 0.158, 0.076, 0.002, -0.080, -0.118, -0.167, -0.225,
        -0.277, -0.316, -0.408, -0.480, -0.556, -0.606]
CM_DE_CT0 = [-1.105, -1.105, -1.105, -1.031, -0.945, -0.939, -0.933,
             -0.928, -0.928, -0.928, -0.928, -0.859, -0.745, -0.573]
CM_DE_CT05 = [-2.142, -2.250, -2.256, -2.262, -2.199, -2.062, -1.912,
              -1.781, -1.650, -1.541, -1.294, -1.220, -1.088, -0.859]

C_PWROFF, C_PWRON = "#2C4B9E", "#E8742A"
ALPHA_STALL = 14.0

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "stix", "font.size": 10,
    "axes.labelsize": 11, "legend.fontsize": 9,
    "axes.grid": True, "grid.alpha": 0.35, "lines.linewidth": 1.7,
})


def main():
    fig, axes = plt.subplots(2, 4, figsize=(13.0, 6.2))

    panels = [
        (axes[0, 0], CL_O_CT0, CL_O_CT05, r"$C_{L_o}$", "(a) Baseline lift"),
        (axes[0, 1], CL_Q_CT0, CL_Q_CT05, r"$C_{L_{\hat{q}}}$", "(b) Pitch-rate lift"),
        (axes[0, 2], CL_DE_CT0, CL_DE_CT05, r"$C_{L_{\delta_e}}$ (rad$^{-1}$)",
         "(c) Elevator lift effectiveness"),
        (axes[0, 3], CD_O_CT0, CD_O_CT05, r"$C_{D_o}$", "(d) Net axial force"),
        (axes[1, 0], CM_O, None, r"$C_{m_o}$", "(e) Static pitching moment"),
        (axes[1, 1], CM_Q_CT0, CM_Q_CT05, r"$C_{m_{\hat{q}}}$",
         "(f) Pitch-rate damping"),
        (axes[1, 2], CM_DE_CT0, CM_DE_CT05, r"$C_{m_{\delta_e}}$ (rad$^{-1}$)",
         "(g) Elevator moment effectiveness"),
    ]
    for ax, y0, y05, ylab, title in panels:
        if y05 is None:
            # Cm_o is CT-independent in Riley Table IIIc: single curve.
            ax.plot(ALPHA, y0, marker="o", ms=3.5, color="#444444",
                    label=r"$C_T$-independent")
            ax.legend(loc="upper right", fontsize=8)
        else:
            ax.plot(ALPHA, y0, marker="o", ms=3.5, color=C_PWROFF,
                    label=r"$C_T = 0$ (power-off)")
            ax.plot(ALPHA, y05, marker="s", ms=3.5, color=C_PWRON,
                    label=r"$C_T = 0.5$ (power-on)")
        ax.axvline(ALPHA_STALL, color="gray", lw=0.9, ls=":")
        ax.set_xlabel(r"$\alpha$ (deg)")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=10)

    axes[0, 3].axhline(0.0, color="black", lw=0.8, alpha=0.6)
    axes[0, 0].annotate(r"$\alpha_s$",
                        xy=(ALPHA_STALL, axes[0, 0].get_ylim()[0]),
                        xytext=(2, 6), textcoords="offset points",
                        fontsize=9, color="gray")
    axes[0, 0].legend(loc="lower right", fontsize=8)
    axes[1, 3].axis("off")

    fig.tight_layout()
    # The results copy follows --out like every other figure; it used to be
    # pinned to results/paper, which silently wrote outside the run's own
    # directory. The manuscript copy is written only if the manuscript tree is
    # actually there -- it is a separate repo and is usually not, and a missing
    # sibling directory should not abort a suite on its last experiment.
    out_results = paths.out_dir()
    out_paper = Path("stall-paper/img")
    for ext in ("png", "pdf"):
        fig.savefig(out_results / f"fig_riley_coefficients.{ext}", dpi=300,
                    bbox_inches="tight")
        if out_paper.is_dir():
            fig.savefig(out_paper / f"riley_coefficients.{ext}", dpi=300,
                        bbox_inches="tight")
        else:
            logger.info("[i] %s not present; manuscript copy skipped",
                        out_paper)
    plt.close(fig)
    logger.info("[+] riley_coefficients.{png,pdf} written")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
