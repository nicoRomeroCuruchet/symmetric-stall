"""
paper_fig_cg_sweep.py — Publication-quality Fig. F: CG sensitivity of the
exact DP solution for the 4-DOF symmetric stall recovery (Riley model).

Reads the artifacts produced by paper_cg_sweep_solve.py:
  (left)  altitude loss from the canonical deep-stall IC vs. CG location;
  (right) elevator-policy switching surface in the (alpha, gamma) plane
          (slice V/Vs ~ 0.95, q = 0) for three CG locations.

Output: results/paper/fig_cg_sweep.{pdf,png}
"""
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from symmetric_stall.aircraft.symmetric_stall import SymmetricStall
from symmetric_stall.policy_iteration import PolicyIterationStall
from symmetric_stall.utils.utils import get_optimal_action

logger = logging.getLogger(__name__)

SWEEP_DIR = Path("results/cg_sweep")
XCG_REF = 0.25
SURFACE_XCGS = [0.20, 0.25, 0.32]          # CGs shown in the right panel
SURFACE_COLORS = ["#D62728", "#2C4B9E", "#2CA02C"]
V_SLICE, Q_SLICE = 0.95, 0.0

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "lines.linewidth": 1.6,
})


def rollout_h(npz_path: Path, dxcg: float, max_time: float = 12.0):
    """Canonical-IC rollout (Approach A) returning t and cumulative h(t).
    Stops at the first upward gamma crossing after the dive (same rule as
    paper_cg_sweep_solve.rollout)."""
    env = SymmetricStall()
    env.airplane.DXCG_OVER_CHORD = dxcg
    pi = PolicyIterationStall.load(npz_path, env=env)

    v_stall = env.airplane.STALL_AIRSPEED
    step_dt = env.airplane.TIME_STEP
    obs, _ = env.specific_reset(0.0, 0.95, np.deg2rad(20.0), 0.0)

    t, h = 0.0, 0.0
    ts, hs = [0.0], [0.0]
    has_dived = False
    while t < max_time:
        action, _, _ = get_optimal_action(obs, pi)
        obs, _, _, _, _ = env.step(action)
        h += obs[1] * v_stall * np.sin(obs[0]) * step_dt
        t += step_dt
        ts.append(t)
        hs.append(h)
        gamma_deg = np.rad2deg(obs[0])
        if gamma_deg < -0.5:
            has_dived = True
        if has_dived and gamma_deg >= 0.0:
            break
    return np.array(ts), np.array(hs)


def main():
    summary = json.loads((SWEEP_DIR / "summary.json").read_text())
    xcgs = sorted(float(k) for k in summary)
    dh = [summary[f"{x:.2f}"]["h"] for x in xcgs]
    status = [summary[f"{x:.2f}"]["status"] for x in xcgs]

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(10.0, 3.8))

    # ── Left: altitude loss vs CG ──────────────────────────────────────
    ax_l.plot(xcgs, dh, color="#2C4B9E", marker="o", markersize=5)
    for x, h, s in zip(xcgs, dh, status):
        if s != "recovered":
            ax_l.annotate(s, xy=(x, h), xytext=(0, 8),
                          textcoords="offset points", ha="center",
                          fontsize=8, color="#D62728")
    ax_l.axvline(XCG_REF, color="gray", linewidth=0.9, linestyle=":")
    ax_l.annotate("Riley ref.\n(0.25 $\\bar{c}$)", xy=(XCG_REF, ax_l.get_ylim()[0]),
                  xytext=(6, 10), textcoords="offset points", fontsize=8.5,
                  color="gray")
    ax_l.set_xlabel(r"CG location $x_{cg}/\bar{c}$")
    ax_l.set_ylabel(r"$\Delta h$ from canonical IC (m)")
    ax_l.set_title("(a) Recovery altitude loss vs. CG", fontsize=10)

    # ── Right: recovery trajectories h(t) for three CG locations ──────
    for xcg, color in zip(SURFACE_XCGS, SURFACE_COLORS):
        path = SWEEP_DIR / f"xcg_{xcg:.2f}.npz"
        if not path.exists():
            logger.warning(f"missing {path}, skipping")
            continue
        ts, hs = rollout_h(path, xcg - XCG_REF)
        ax_r.plot(ts, hs, color=color,
                  label=rf"$x_{{cg}} = {xcg:.2f}\,\bar{{c}}$")
        ax_r.annotate(f"{hs[-1]:.2f} m", xy=(ts[-1], hs[-1]),
                      xytext=(4, -2), textcoords="offset points",
                      fontsize=8.5, color=color, va="top")

    ax_r.set_xlabel("Time (s)")
    ax_r.set_ylabel(r"$\Delta h$ (m)")
    ax_r.set_title("(b) Optimal recovery from the canonical IC", fontsize=10)
    ax_r.legend(loc="lower left")
    ax_r.set_xlim(right=ax_r.get_xlim()[1] * 1.12)

    fig.tight_layout()
    out_dir = Path("results/paper")
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"fig_cg_sweep.{ext}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)
    logger.info("[+] Saved results/paper/fig_cg_sweep.{pdf,png}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
