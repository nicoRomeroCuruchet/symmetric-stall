"""How far aft the CG can go before the recovery, not the airframe, breaks.

The robustness matrix sweeps +-7.5 % of chord and finds that an aft CG helps,
monotonically. That reads wrong against the obvious intuition -- move the CG
far enough aft and the aeroplane becomes uncontrollable -- so this pushes the
axis until something actually breaks, and measures two things at each station:

  open loop   the short-period eigenvalues with the controls frozen, obtained
              by numerically linearising d(alpha, q)/dt. Negative static
              margin has to show up here as a real positive root.

  closed loop the altitude loss of the DP optimum, which is what the matrix
              reports.

They part company, and that is the point: the airframe goes divergent near
0.46 c-bar while the closed-loop recovery keeps improving straight through it.
A policy stepping at 100 Hz with full elevator authority holds a mildly
unstable airframe without difficulty -- relaxed static stability, arrived at by
accident. What the robustness matrix measures is a closed-loop quantity, and
closed loop the instability is not merely survivable, it is useful: less
stability means the nose comes down sooner.

    python scripts/paper1/cg_reach.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import paper_robustness as R
from symmetric_stall import paths
from symmetric_stall import procedures as P

logger = logging.getLogger(__name__)

#: Reference CG of Riley's tables, in chord fractions.
X_REF = 0.25

#: Where the sweep goes. Well past the divergence boundary on the aft side.
DXCG = [round(-0.100 + 0.025 * i, 4) for i in range(17)]      # -0.10 .. +0.30

#: Linearisation point: unstalled, in the linear range, so the eigenvalues mean
#: what the textbook says they mean.
LIN_ALPHA_DEG, LIN_VNORM, LIN_DE_DEG = 5.0, 1.5, -2.0

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "stix", "font.size": 10,
    "axes.labelsize": 11, "legend.fontsize": 9,
})


def short_period_eigs(env, alpha_deg=LIN_ALPHA_DEG, vnorm=LIN_VNORM,
                      de_deg=LIN_DE_DEG, h=1e-5):
    """Eigenvalues of the (alpha, q) Jacobian with the controls frozen."""
    def rate(a, q):
        env.specific_reset(0.0, vnorm, a, q)
        act = np.array([np.deg2rad(de_deg), 1.0], dtype=np.float32)
        o1, *_ = env.step(act)
        return (np.array([o1[2], o1[3]]) - np.array([a, q])) / env.airplane.TIME_STEP

    a0, q0 = np.deg2rad(alpha_deg), 0.0
    J = np.empty((2, 2))
    for i, (da, dq) in enumerate(((h, 0.0), (0.0, h))):
        J[:, i] = (rate(a0 + da, q0 + dq) - rate(a0 - da, q0 - dq)) / (2 * h)
    return np.linalg.eigvals(J)


def sweep():
    pi = paths.load_policy(env=P.SymmetricStall())
    a0, v0 = P.CANONICAL
    rows = []
    for dx in DXCG:
        env = R.perturbed_env(1.0, dx)
        ev = short_period_eigs(env)
        lam = float(np.max(ev.real))
        r = P.rollout(env, pi, P.ctrl_optimal, a0, v0, record=True)
        amax = float(np.rad2deg(np.max(r["hist"]["alpha"])))
        de = np.rad2deg(np.array(r["hist"]["de"]))
        rows.append({
            "dxcg": dx, "x_cg": X_REF + dx, "lambda_max": lam,
            "h": r["h"], "t": r["t"], "status": r["status"],
            "alpha_max_deg": amax,
            # How hard the policy has to work to hold it: total elevator travel.
            "de_tv_deg": float(np.abs(np.diff(de)).sum()),
            "de_min_deg": float(de.min()), "de_max_deg": float(de.max()),
        })
        logger.info(f"x_cg={X_REF+dx:.3f}c  lambda_max={lam:+.3f}/s  "
                    f"h={r['h']:7.2f} m  de travel={rows[-1]['de_tv_deg']:6.1f} deg"
                    f"  {r['status']}")
    return rows


def figure(rows, out_dir):
    x = np.array([r["x_cg"] for r in rows])
    lam = np.array([r["lambda_max"] for r in rows])
    h = np.array([r["h"] for r in rows])
    tv = np.array([r["de_tv_deg"] for r in rows])

    # Where the airframe goes divergent, by interpolation on lambda_max.
    x_div = None
    for i in range(len(x) - 1):
        if lam[i] <= 0.0 < lam[i + 1]:
            x_div = x[i] + (x[i + 1] - x[i]) * (-lam[i]) / (lam[i + 1] - lam[i])
            break

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.2, 5.6), sharex=True)

    ax1.axhline(0.0, color="0.6", lw=0.9, ls=":")
    ax1.plot(x, lam, marker="o", ms=4, lw=1.8, color="#B22222")
    ax1.set_ylabel(r"open-loop $\max\,\mathrm{Re}\,\lambda$  (1/s)")
    ax1.set_title("(a) The airframe, controls frozen", fontsize=10)
    ax1.grid(True, ls=":", lw=0.6, alpha=0.6)

    ax2.plot(x, h, marker="o", ms=4, lw=1.8, color="#2C4B9E",
             label=r"$\Delta h$, DP optimum")
    ax2.set_ylabel(r"$\Delta h$ (m)")
    ax2.set_xlabel(r"$x_{cg}/\bar{c}$")
    ax2.set_title("(b) The recovery the policy flies", fontsize=10)
    ax2.grid(True, ls=":", lw=0.6, alpha=0.6)

    axr = ax2.twinx()
    axr.plot(x, tv, marker="s", ms=3.5, lw=1.4, color="#888888", ls="--",
             label="elevator travel")
    axr.set_ylabel("total elevator travel (deg)", color="#666666", fontsize=9)
    axr.tick_params(axis="y", labelcolor="#666666")

    for ax in (ax1, ax2):
        ax.axvline(X_REF, color="0.35", lw=0.9, ls="-.")
        if x_div is not None:
            ax.axvspan(x_div, x.max() + 0.01, color="#B22222", alpha=0.08,
                       zorder=0)
            ax.axvline(x_div, color="#B22222", lw=1.1, ls="--")
    ax1.annotate("Riley reference", xy=(X_REF, 0), xytext=(3, -6),
                 textcoords="offset points", fontsize=8, color="0.35")
    if x_div is not None:
        ax1.annotate(f"divergent beyond\n{x_div:.3f}" + r"$\,\bar{c}$",
                     xy=(x_div, lam.max() * 0.55), xytext=(6, 0),
                     textcoords="offset points", fontsize=8.5, color="#B22222")

    lines = ax2.get_lines()[:1] + axr.get_lines()[:1]
    ax2.legend(lines, [ln.get_label() for ln in lines], loc="lower left",
               fontsize=8.5)
    fig.tight_layout()
    P.stamp_engine(fig)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig_cg_reach.{ext}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)
    logger.info("[+] fig_cg_reach.{png,pdf} written")
    return x_div


def main():
    import json
    out_dir = paths.out_dir()
    rows = sweep()
    x_div = figure(rows, out_dir)
    P.dump_json(out_dir / "cg_reach.json",
                {"x_ref": X_REF, "x_divergent": x_div,
                 "linearisation": {"alpha_deg": LIN_ALPHA_DEG,
                                   "vnorm": LIN_VNORM, "de_deg": LIN_DE_DEG},
                 "canonical_ic": {"alpha0_deg": P.CANONICAL[0],
                                  "vnorm0": P.CANONICAL[1]},
                 "rows": rows}, indent=1)
    logger.info("[+] cg_reach.json written")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
