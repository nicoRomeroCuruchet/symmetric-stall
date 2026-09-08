"""
paper_fig_policy_q_slices.py — Fig: the elevator switching surface
across pitch-rate slices of the converged policy.

Two rows of three panels: commanded elevator over the (alpha, gamma)
plane at V/Vs = 0.9, nose-down slices q in {-10,-20,-30} deg/s on
top and nose-up slices {+10,+20,+30} below (the q = 0 slice lives in
the main policy figure). A white dotted line travels with each
slice's median crossing; a faint fixed one marks alpha_s = 14. The
boundary shifts ~2 deg per 10 deg/s of pitch rate: the momentum
anticipation of the reversal, reconciling the q = 0 maps (crossing
13.4 deg) with the closed-loop reversal statistics (median 17.2 deg,
flown at q ~ -20 deg/s). Median crossing per the criterion of
paper_switching_stats.py (-40 <= gamma <= 0, V <= 1.2 Vs, elevator
zero crossing below alpha = 30 deg).

Output: fig_policy_q_slices.{png,pdf} in paths.out_dir()
(+ manuscript copy img/policy_q_slices.{png,pdf} if present).
"""
import logging
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

POLICY = ("data/policies/"
          "SymmetricStall_riley_56x81x80x41_thrust-riley.npz")
Q_ROWS_DEG = ((-10.0, -20.0, -30.0), (10.0, 20.0, 30.0))
V_TARGET = 0.9
ALPHA_STALL = 14.0

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "stix", "font.size": 9,
    "axes.labelsize": 10,
})


def main():
    d = np.load(os.environ.get("STALL_POLICY", POLICY))
    P = d["policy"].reshape(d["grid_shape"])
    A = d["action_space"]
    lo, hi, shape = d["bounds_low"], d["bounds_high"], np.array(d["grid_shape"])
    gamma = np.linspace(lo[0], hi[0], shape[0])
    vn = np.linspace(lo[1], hi[1], shape[1])
    alpha = np.rad2deg(np.linspace(lo[2], hi[2], shape[2]))
    qb = np.linspace(lo[3], hi[3], shape[3])
    de_of = np.rad2deg(A[:, 0])

    vi = int(np.argmin(np.abs(vn - V_TARGET)))
    gmask = (gamma >= np.deg2rad(-90.1)) & (gamma <= 0.01)
    amask_plot = (alpha >= -5.1) & (alpha <= 25.1)
    gdeg = np.rad2deg(gamma[gmask])
    adeg = alpha[amask_plot]

    def median_crossing(qi):
        gm = (gamma >= np.deg2rad(-40)) & (gamma <= 0.0)
        vm = vn <= 1.2
        am = alpha <= 30.0
        sw = []
        for g_i in np.where(gm)[0]:
            for v_i in np.where(vm)[0]:
                col = de_of[P[g_i, v_i, :, qi]]
                neg = np.where((col < 0) & am)[0]
                if len(neg) == 0 or neg[-1] + 1 >= len(alpha):
                    continue
                k = neg[-1]
                d0, d1 = col[k], col[k + 1]
                t = d0 / (d0 - d1) if d1 != d0 else 0.5
                sw.append(alpha[k] + t * (alpha[k + 1] - alpha[k]))
        return float(np.median(sw))

    fig, axes = plt.subplots(2, 3, figsize=(11.0, 5.6),
                             sharey=True, sharex=True)
    for row, qrow in zip(axes, Q_ROWS_DEG):
        for ax, qt in zip(row, qrow):
            qi = int(np.argmin(np.abs(qb - np.deg2rad(qt))))
            de_slice = de_of[P[:, vi, :, qi]][gmask][:, amask_plot]
            ax.pcolormesh(adeg, gdeg, de_slice, cmap="plasma",
                          vmin=-25, vmax=15, shading="gouraud")
            # Fixed faint reference at alpha_s; the white dotted line
            # travels with the slice's own median crossing.
            ax.axvline(ALPHA_STALL, color="0.6", ls=":", lw=0.8)
            med = median_crossing(qi)
            ax.axvline(med, color="white", ls=":", lw=1.4)
            ax.set_title(rf"$q = {qt:+.0f}$ deg/s", fontsize=9)
            ax.text(0.03, 0.06, rf"median crossing ${med:.1f}°$",
                    transform=ax.transAxes, fontsize=8, color="white")
    for ax in axes[1]:
        ax.set_xlabel(r"$\alpha$ (deg)")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\gamma$ (deg)")
    fig.text(0.995, -0.03, "solver fields, ideal engine; thrust riley",
             ha="right", va="bottom", fontsize=7, color="0.55")

    fig.tight_layout()
    from symmetric_stall import paths
    out_results = paths.out_dir()
    out_paper = Path("stall-paper/img")
    for ext in ("png", "pdf"):
        fig.savefig(out_results / f"fig_policy_q_slices.{ext}", dpi=300,
                    bbox_inches="tight")
        if out_paper.is_dir():
            fig.savefig(out_paper / f"policy_q_slices.{ext}", dpi=300,
                        bbox_inches="tight")
    plt.close(fig)
    logger.info("[+] fig_policy_q_slices.{png,pdf} written")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
