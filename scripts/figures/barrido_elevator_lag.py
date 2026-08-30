"""Sweep the elevator time constant at Riley's engine lag, canonical entry.

The engine is held at RILEY_TAU_E = 0.85 s -- the constant recovered from
Riley's own throttle chops -- and only the elevator channel is swept, so every
metre of difference between rows is attributable to elevator bandwidth alone.

WHY A SWEEP AND NOT A VALUE. Riley models no elevator dynamics, and neither
Gratton's flight tests nor the theses in refs/ report one, so unlike tau_e
there is no number to recover from the source. The AA-1 has a reversible,
directly linked control system -- no servo -- so what limits the elevator is
the pilot's neuromuscular response plus control-system compliance, which the
quasi-linear pilot models put near 0.1 s. That is an order of magnitude taken
from a literature this repository does not hold a copy of, so the defensible
result is the SHAPE of the curve rather than one point on it: if the loss is
flat across the plausible band, the choice of tau stops mattering and the
caveat becomes trivial to defend.

The sweep therefore runs well past any plausible pilot, out to 1 s, to find
where the recovery does start to degrade.

The policy is never re-solved. It is the one trained against an instantaneous
elevator, flown here on a lagged one, which is the honest evaluation of that
policy and biases every number in the SAFE direction: a policy optimal for the
ideal plant can only do worse on the lagged plant than the policy optimal for
the lagged plant, so these losses are upper bounds.

Usage:
    STALL_POLICY=data/policies/SymmetricStall_riley_56x81x80x41_thrust-riley.npz \
    python3 scripts/figures/barrido_elevator_lag.py
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from symmetric_stall import runconfig  # noqa: E402

ENGINE_TAU = 0.85
#: Starts at the nominal pilot bandwidth and runs an order of magnitude past
#: it. Anything beyond ~0.3 s is no longer a pilot but a control system in
#: trouble; it is included to locate the knee, not to claim it is realistic.
ELEVATOR_TAUS = [0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.75, 1.00]

OUT = REPO / "results" / "6_riley_engine" / "elevator-lag"
OUT.mkdir(parents=True, exist_ok=True)
os.environ["STALL_OUT"] = str(OUT)

runconfig.apply(thrust="riley", engine_tau=ENGINE_TAU)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import symmetric_stall.procedures as P  # noqa: E402


def main() -> None:
    a0, v0 = P.CANONICAL
    print(f"canonical entry: alpha0 = {a0} deg, V0/Vs = {v0}")
    print(f"engine held at tau_e = {ENGINE_TAU} s\n")

    rows = []
    # tau = 0 is the reference the sweep is measured against: same engine, no
    # elevator lag. It is not part of the sweep the user asked for, but without
    # it every number below would be an absolute with nothing to subtract.
    for tau in [0.0] + ELEVATOR_TAUS:
        out = P.make_trajectory_comparison_figure(
            engine_tau=ENGINE_TAU, elevator_tau=tau)
        row = {"elevator_tau_s": tau}
        for label, res in out.items():
            arm = ("DP" if "optimum" in label
                   else "CAA" if "CAA" in label else "FAA")
            row[arm] = float(res["h"])
            row[f"{arm}_t"] = float(res["t"])
            row[f"{arm}_status"] = res["status"]
        rows.append(row)
        print(f"  tau_de = {tau:4.2f} s   "
              + "   ".join(f"{a}={row[a]:8.2f} m" for a in ("DP", "CAA", "FAA"))
              + f"   [{row['DP_status']}]")

    P.dump_json(OUT / "elevator_lag_sweep.json",
                {"engine_tau_s": ENGINE_TAU,
                 "canonical_ic": {"alpha0_deg": a0, "v0_over_vs": v0},
                 "rows": rows})

    summary_figure(rows)


def summary_figure(rows) -> None:
    """Loss vs elevator time constant, one curve per arm.

    Deliberately the same shape as the power-delay figure of
    Sec. pilot_sensitivity, so the two effects can be read on a common scale:
    that one costs ~7.7 m per second of delay, and the question this figure
    answers is how many metres per second of elevator LAG -- a different
    quantity, phase rather than transport, and the paper should not let a
    reader conflate them.
    """
    taus = np.array([r["elevator_tau_s"] for r in rows])
    colors = {"DP": "#1f77b4", "CAA": "#ff7f0e", "FAA": "#2ca02c"}
    names = {"DP": "DP optimum", "CAA": r"CAA ($\alpha$-hold)",
             "FAA": r"FAA ($\alpha$-hold)"}

    rc = {"font.size": 12, "axes.labelsize": 12, "legend.fontsize": 10,
          "axes.spines.top": False, "axes.spines.right": False}
    with plt.rc_context(rc):
        fig, axs = plt.subplots(1, 2, figsize=(10.0, 4.0))

        for arm in ("DP", "CAA", "FAA"):
            h = np.array([r[arm] for r in rows])
            axs[0].plot(taus, h, "o-", color=colors[arm], label=names[arm], lw=1.6)
            # Excess over the same arm with an instantaneous elevator, which is
            # what isolates the elevator from everything else in the plant.
            axs[1].plot(taus, h[0] - h, "o-", color=colors[arm],
                        label=names[arm], lw=1.6)

        axs[0].set_ylabel(r"$\Delta h$ (m)")
        axs[0].set_title("Altitude loss")
        axs[1].set_ylabel(r"excess over $\tau_{\delta_e}=0$ (m)")
        axs[1].set_title("Cost of elevator lag alone")
        for ax in axs:
            ax.set_xlabel(r"elevator time constant $\tau_{\delta_e}$ (s)")
            ax.grid(alpha=0.3)
            ax.legend()
        fig.suptitle(
            rf"Elevator bandwidth at Riley's engine ($\tau_e={ENGINE_TAU}$ s), "
            "canonical entry", fontsize=12)
        fig.tight_layout()
        P.stamp_engine(fig)
        for ext in ("png", "pdf"):
            fig.savefig(OUT / f"fig_elevator_lag_sweep.{ext}",
                        dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(f"\n[+] fig_elevator_lag_sweep.{{png,pdf}} written to {OUT}")


if __name__ == "__main__":
    main()
