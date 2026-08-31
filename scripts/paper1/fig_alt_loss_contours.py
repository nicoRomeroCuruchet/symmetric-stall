"""Regenerate Fig. 1 of Case I: minimal altitude loss over the entry envelope.

The figure the manuscript calls `combined_alt_loss_contours.png` existed --
it is on page 9 of the 2026-08-17 build -- but neither the file nor the
script that drew it survived, in either repository, on any branch, or on any
of the three GPU nodes. Only the rendered page remained.

It is recoverable because it is not a rollout: it is the converged value
function itself. On the shortest-path formulation the minimal altitude loss
from an entry is the negative of V* at that state,

    dh_min(V_0, gamma_0, mu_0)  ~=  -V*(gamma_0, V_0/V_s, mu_0)

up to the marginal contribution of the bank-rate regularisation, so the four
panels are four slices of `policy_L1_53280.npz` -- 430 KB, in the repository,
no GPU and no solve involved.

The four airspeeds the panels ask for, V_0/V_s in {1.2, 2.0, 3.0, 4.0}, land
exactly on grid nodes of the baseline discretisation (0.9 to 4.0 in steps of
0.1), so no interpolation enters the figure and what is plotted is the
solver's own output.

Grid, from the archive: state order (gamma, V/V_s, mu), shape (37, 32, 45),
gamma over [-180, 0] deg, mu over [-20, 200] deg. The published panels crop
to the physically meaningful quadrant, gamma in [-90, 0] and mu in [0, 180].

Usage:
    python3 scripts/paper1/fig_alt_loss_contours.py
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
CASE1 = REPO / "results" / "6_riley_engine" / "case1_3dof"
POLICY = CASE1 / "policy_L1_53280.npz"

#: The panels of the published figure, in reading order.
SPEEDS = (1.2, 2.0, 3.0, 4.0)
#: The quadrant the published figure crops to.
GAMMA_WINDOW = (-90.0, 0.0)
MU_WINDOW = (0.0, 180.0)
#: Contour levels, 30 m apart, as read off the archived page: the labelled
#: lines there are 30 apart AND each one separates two filled bands. Filling
#: on a different set from the lines -- which the first version of this script
#: did -- leaves bands and lines crossing each other at nothing in particular,
#: and the figure reads as noise even though the field is identical.
LEVELS = tuple(range(0, 271, 30))
#: Tolerance for deciding whether a grid node is inside the plotted window.
#:
#: It is not cosmetic. The archive stores the bounds in float32, so
#: bounds_low[0] is -3.1415927, which widens to -180.0000005 deg and shifts
#: every node by -2.5e-6 deg. A bare `gamma >= -90.0` then rejects the node at
#: -90.00000250 and `mu <= 180.0` rejects the one at 180.00000096, dropping a
#: whole row and a whole column -- 5 deg each -- of the solver's own output.
#: It shows up as a white strip along the bottom and right of every panel,
#: which is how it was caught.
EDGE_TOL_DEG = 1e-3


def load():
    d = np.load(POLICY)
    shape = tuple(int(x) for x in d["grid_shape"])
    lo, hi = d["bounds_low"], d["bounds_high"]
    axes = [np.linspace(float(lo[i]), float(hi[i]), shape[i])
            for i in range(len(shape))]
    # Altitude LOST, positive, which is what the axis labels of the published
    # figure report. V* is negative on this formulation.
    dh = -d["value_function"].reshape(shape)
    return dh, np.rad2deg(axes[0]), axes[1], np.rad2deg(axes[2])


def main() -> None:
    dh, gamma, vnorm, mu = load()

    gsel = ((gamma >= GAMMA_WINDOW[0] - EDGE_TOL_DEG)
            & (gamma <= GAMMA_WINDOW[1] + EDGE_TOL_DEG))
    msel = ((mu >= MU_WINDOW[0] - EDGE_TOL_DEG)
            & (mu <= MU_WINDOW[1] + EDGE_TOL_DEG))
    g, m = gamma[gsel], mu[msel]
    # The window is a whole number of grid steps on both axes, so anything
    # else means the tolerance is wrong or the grid is not the one assumed.
    assert len(g) == 19 and len(m) == 37, (
        f"expected a 19 x 37 window, got {len(g)} x {len(m)}: "
        f"gamma {g[0]:.4f}..{g[-1]:.4f}, mu {m[0]:.4f}..{m[-1]:.4f}")

    # One shared colour scale across the four panels: the point of the figure
    # is that the loss grows with entry speed, and per-panel autoscaling would
    # hide exactly that by giving each panel its own red.
    slices = []
    for v in SPEEDS:
        k = int(np.argmin(np.abs(vnorm - v)))
        assert abs(vnorm[k] - v) < 1e-6, (
            f"V/Vs = {v} is not a grid node; the nearest is {vnorm[k]:.3f}. "
            "Interpolating here would put something in the figure that the "
            "solver did not compute.")
        slices.append(dh[np.ix_(np.where(gsel)[0], [k], np.where(msel)[0])]
                      .squeeze(axis=1))
    vmax = max(s.max() for s in slices)
    assert vmax < LEVELS[-1], (
        f"the deepest loss is {vmax:.1f} m but the levels stop at "
        f"{LEVELS[-1]}; the top band would be clipped")

    rc = {"font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9}
    with plt.rc_context(rc):
        fig, axs = plt.subplots(2, 2, figsize=(8.0, 5.2), sharex=True,
                                sharey=True)
        for ax, v, s in zip(axs.ravel(), SPEEDS, slices):
            # Same levels for the fill and the lines, and a fixed set rather
            # than one derived from the data, so the four panels share a scale
            # without a colourbar having to assert it.
            ax.contourf(m, g, s, levels=LEVELS, cmap="jet")
            cs = ax.contour(m, g, s, levels=LEVELS[1:], colors="k",
                            linewidths=0.5)
            ax.clabel(cs, inline=True, fontsize=6, fmt="%d")
            ax.set_title(rf"$V/V_s = {v:.1f}$")
            ax.set_xticks([0, 45, 90, 135, 180])
            ax.set_yticks([-90, -60, -30, 0])
            # Matplotlib pads the data range by 5% by default, which leaves a
            # white strip along the top and bottom of every panel. The
            # archived figure has none: the field is defined on the whole box.
            ax.set_xlim(MU_WINDOW)
            ax.set_ylim(GAMMA_WINDOW)
        for ax in axs[-1, :]:
            ax.set_xlabel("Bank angle (deg)")
        for ax in axs[:, 0]:
            ax.set_ylabel("Flight path angle (deg)")
        # No colourbar: the archived figure has none, and every band boundary
        # is already a labelled line, so one would only repeat the labels.
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(CASE1 / f"combined_alt_loss_contours.{ext}",
                        dpi=300, bbox_inches="tight")
        plt.close(fig)

    print(f"[+] combined_alt_loss_contours.{{png,pdf}} written to {CASE1}")
    print(f"    dh_min over the four panels: 0.0 to {vmax:.1f} m")
    for v, s in zip(SPEEDS, slices):
        print(f"    V/Vs = {v:.1f}:  max {s.max():6.1f} m "
              f"at mu = {m[np.unravel_index(s.argmax(), s.shape)[1]]:.0f} deg, "
              f"gamma = {g[np.unravel_index(s.argmax(), s.shape)[0]]:.0f} deg")


if __name__ == "__main__":
    main()
