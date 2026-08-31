"""Does the optimum only hold at the point it was solved for?

The reviewer's objection to a globally optimal policy computed for one
aeroplane is that it is optimal for THAT aeroplane and nothing else. The
robustness matrix answers half of it -- the nominal policy does not depart
controlled flight anywhere in the loading envelope -- but it cannot say how
much of the remaining loss is avoidable, because it never computes the policy
that aeroplane deserves. This script does: pi* is re-solved at the perturbed
mass and the two are flown against each other on the same aircraft.

THREE ARMS, NOT TWO. Comparing only pi_M0 against pi_M1 conflates two
different failures, and the paper's own argument depends on separating them:

  pi_M0 raw        the aircraft is heavier and nobody told the policy. Vs is
                   the nominal one, so the vnorm axis the policy indexes no
                   longer means what it meant. This is a NORMALISATION error.
  pi_M0 rescaled   the same policy, with Vs recomputed for the real mass --
                   one scalar, known from the loading sheet before take-off.
  pi_M1            the policy re-solved for that aircraft. The floor.

If arm 2 nearly reaches arm 3, the answer to the objection is much stronger
than robustness: retraining is not merely survivable to skip, it is close to
unnecessary, because the whole recoverable gap is one pre-flight scalar. If
arm 2 stays far from arm 3, the gap is real policy error and the paper must
say so.

All three arms start at the SAME PHYSICAL AIRSPEED -- v0 as a fraction of the
aircraft's REAL stall speed. That is why the raw arm's initial condition is
scaled by sqrt(mass_factor): its model carries the nominal Vs, so reaching the
same true speed takes a different vnorm. Getting this wrong makes the raw arm
start somewhere else entirely and the comparison meaningless.

Engine and elevator both lagged, as everywhere else in the current results.

Usage:
    python3 scripts/paper1/retrain_mass_study.py
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from symmetric_stall import runconfig  # noqa: E402

ENGINE_TAU = 0.85
ELEVATOR_TAU = 0.10

runconfig.apply(thrust="riley", engine_tau=ENGINE_TAU, elevator_tau=ELEVATOR_TAU)

import numpy as np  # noqa: E402

from symmetric_stall import paths  # noqa: E402
from symmetric_stall.aircraft.symmetric_stall import SymmetricStall  # noqa: E402
from symmetric_stall.procedures import (  # noqa: E402
    ALPHA_GRID_DEG, VNORM_GRID, CANONICAL, ctrl_optimal, rollout,
)

POLICY_DIR = REPO / "data" / "policies"
STEM = "SymmetricStall_riley_56x81x80x41_thrust-riley"
NOMINAL = POLICY_DIR / f"{STEM}.npz"

#: Retrained policies, by mass factor. Add a row as each solve lands.
RETRAINED = {
    0.90: POLICY_DIR / f"{STEM}_mass090.npz",
    1.10: POLICY_DIR / f"{STEM}_mass110.npz",
}

OUT = REPO / "results" / "6_riley_engine" / "retrain-mass"


def perturbed_plant(mass_factor: float, rescale_vs: bool) -> SymmetricStall:
    """The aeroplane at `mass_factor`, with or without a corrected Vs.

    MASS is scaled AFTER construction on purpose: STALL_AIRSPEED is derived
    from the mass in __init__, so assigning here leaves it at the nominal
    value, which is precisely the un-rescaled case. `rescale_vs` then puts it
    right, and sqrt is exact because Vs goes as the square root of weight.

    Do NOT reach for the MASS_FACTOR environment variable here. That rebuilds
    the whole aeroplane consistently, which is what TRAINING wants and what
    makes the un-rescaled arm impossible to express.
    """
    env = SymmetricStall()
    env.airplane.MASS = env.airplane.MASS * mass_factor
    if rescale_vs:
        env.airplane.STALL_AIRSPEED *= float(np.sqrt(mass_factor))
    return env


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pi_nom = paths.load_policy(NOMINAL, env=SymmetricStall())

    rows = []
    for mf in sorted(RETRAINED):
        path = RETRAINED[mf]
        if not path.exists():
            print(f"[!] no retrained policy for mass {mf}: {path.name} -- skipped")
            continue
        pi_ret = paths.load_policy(path, env=SymmetricStall())
        meta = json.loads(str(np.load(path)["run_metadata"]))
        assert float(meta["mass_factor"]) == mf, (
            f"{path.name} says mass_factor={meta['mass_factor']}, expected {mf}")

        print(f"\n=== mass factor {mf}  ({715.3152 * mf:.1f} kg) ===")
        print(f"{'alpha0':>7} {'v0/Vs':>6} | {'pi_M0 raw':>10} {'pi_M0 resc':>11} "
              f"{'pi_M1':>9} | {'raw vs M1':>10} {'resc vs M1':>11}")
        for a0 in ALPHA_GRID_DEG:
            for v0 in VNORM_GRID:
                # Same physical entry speed in all three arms.
                h_raw = rollout(perturbed_plant(mf, False), pi_nom, ctrl_optimal,
                                a0, v0 * float(np.sqrt(mf)))["h"]
                h_res = rollout(perturbed_plant(mf, True), pi_nom, ctrl_optimal,
                                a0, v0)["h"]
                h_ret = rollout(perturbed_plant(mf, True), pi_ret, ctrl_optimal,
                                a0, v0)["h"]
                # Excess over the retrained floor, per cent of the floor's own
                # loss. Losses are negative, so the ratio is written on
                # magnitudes to keep "more loss" positive.
                exc_raw = 100.0 * (abs(h_raw) - abs(h_ret)) / abs(h_ret)
                exc_res = 100.0 * (abs(h_res) - abs(h_ret)) / abs(h_ret)
                rows.append({"mass_factor": mf, "alpha0_deg": a0, "vnorm0": v0,
                             "h_pi_M0_raw": h_raw, "h_pi_M0_rescaled": h_res,
                             "h_pi_M1": h_ret,
                             "excess_raw_pct": exc_raw,
                             "excess_rescaled_pct": exc_res,
                             "canonical": (a0, v0) == CANONICAL})
                print(f"{a0:7.0f} {v0:6.2f} | {h_raw:10.2f} {h_res:11.2f} "
                      f"{h_ret:9.2f} | {exc_raw:9.1f}% {exc_res:10.1f}%")

    (OUT / "retrain_mass_study.json").write_text(json.dumps(
        {"run_config": runconfig.describe(),
         "engine_tau_s": ENGINE_TAU, "elevator_tau_s": ELEVATOR_TAU,
         "rows": rows}, indent=1))
    print(f"\n[+] retrain_mass_study.json written to {OUT}")

    if rows:
        summarise(rows)
        make_table(rows, OUT / "table_retrain_mass.tex")
        make_dumbbell(rows, OUT / "fig_retrain_mass")


def summarise(rows) -> None:
    """The two sentences the paper needs, computed rather than eyeballed."""
    for mf in sorted({r["mass_factor"] for r in rows}):
        sub = [r for r in rows if r["mass_factor"] == mf]
        raw = np.array([r["excess_raw_pct"] for r in sub])
        res = np.array([r["excess_rescaled_pct"] for r in sub])
        print(f"\nmass {mf}: over {len(sub)} entries")
        print(f"  pi_M0 raw       excess {raw.min():6.1f} .. {raw.max():6.1f} %"
              f"   median {np.median(raw):6.1f} %")
        print(f"  pi_M0 rescaled  excess {res.min():6.1f} .. {res.max():6.1f} %"
              f"   median {np.median(res):6.1f} %")
        recovered = 100.0 * (1.0 - np.median(res) / np.median(raw)) \
            if abs(np.median(raw)) > 1e-9 else float("nan")
        print(f"  -> rescaling one scalar removes {recovered:.0f} % of the "
              f"median gap; retraining is what remains")




def make_table(rows, out: Path) -> None:
    """LaTeX table: the deliverable. One row per (mass, entry).

    Numbers rather than a figure because the claim is per-cell -- "in none of
    the eighteen entries does retraining recover more than X" is something a
    reader checks, and a plot cannot be checked.
    """
    lines = [
        r"\begin{table}[H]",
        r"\caption{Retraining the optimum for a perturbed mass. $\pi_{M_0}$ is",
        r"the nominal policy (715.3~kg); $\pi_{M_1}$ is re-solved for the mass",
        r"in the first column. All three arms fly the SAME aircraft and enter",
        r"at the same true airspeed. \emph{raw} carries the nominal stall-speed",
        r"normalisation; \emph{rescaled} corrects that single scalar from the",
        r"pre-flight weight. Excess is over $\pi_{M_1}$, the floor. Negative",
        r"entries beat that floor and so measure the resolution of the",
        r"comparison itself (discretisation plus barycentric execution),",
        r"about 3\%: a gap smaller than that is not resolved.}",
        r"\label{tab:retrain_mass}",
        r"\centering",
        r"\begin{tabular}{ccrrrrr}",
        r"\hline",
        r"$m/m_0$ & $(\alpha_0,\,V_0/V_s)$ & $\pi_{M_0}$ raw & "
        r"$\pi_{M_0}$ resc. & $\pi_{M_1}$ & raw & resc. \\",
        r" & (deg, --) & (m) & (m) & (m) & (\%) & (\%) \\",
        r"\hline",
    ]
    for mf in sorted({r["mass_factor"] for r in rows}):
        for r in [x for x in rows if x["mass_factor"] == mf]:
            star = r"$^\ast$" if r["canonical"] else ""
            lines.append(
                f"{mf:.2f} & $({r['alpha0_deg']:.0f},\\,{r['vnorm0']:.2f})${star} & "
                f"{r['h_pi_M0_raw']:.2f} & {r['h_pi_M0_rescaled']:.2f} & "
                f"{r['h_pi_M1']:.2f} & {r['excess_raw_pct']:+.1f} & "
                f"{r['excess_rescaled_pct']:+.1f} \\\\")
        lines.append(r"\hline")
    lines += [r"\end{tabular}", r"\end{table}",
              r"% $^\ast$ canonical entry."]
    out.write_text("\n".join(lines) + "\n")
    print(f"[+] {out.name} written")


def make_dumbbell(rows, out_stem: Path, noise_pct: float = 3.0) -> None:
    """Excess over the retrained floor, one row per entry.

    The first version of this plotted absolute Delta h and was useless: the
    spread BETWEEN entries is 15 m while the gap WITHIN an entry is under 1 m,
    so all three marks landed on top of each other and the figure showed the
    one thing already known. Plotting the excess puts every entry on a common
    scale and makes the gap the only thing on the page.

    The shaded band is the resolution of the comparison itself. At m/m0 = 0.90
    the rescaled policy BEATS the retrained one by up to 3.4 %, which is
    impossible by construction -- pi_M1 is optimal for that aircraft -- so that
    excursion measures discretisation plus barycentric execution rather than
    any real advantage. Nothing inside the band is resolved, and saying so is
    what stops a reader from reading the small positive numbers as findings.
    """
    import matplotlib.pyplot as plt

    rc = {"font.size": 11, "axes.labelsize": 11, "legend.fontsize": 9,
          "axes.spines.top": False, "axes.spines.right": False}
    masses = sorted({r["mass_factor"] for r in rows})
    with plt.rc_context(rc):
        fig, axs = plt.subplots(1, len(masses), figsize=(5.4 * len(masses), 4.2))
        axs = np.atleast_1d(axs)
        for ax, mf in zip(axs, masses):
            sub = [r for r in rows if r["mass_factor"] == mf]
            y = np.arange(len(sub))
            ax.axvspan(-noise_pct, noise_pct, color="0.88", zorder=0,
                       label=f"$\pm${noise_pct:g}%")
            ax.axvline(0.0, color="0.4", lw=1.0, zorder=1)
            for k, r in enumerate(sub):
                ax.plot([r["excess_rescaled_pct"], r["excess_raw_pct"]], [k, k],
                        color="0.6", lw=2.0, zorder=2, solid_capstyle="round")
            ax.scatter([r["excess_raw_pct"] for r in sub], y, s=46, zorder=3,
                       color="#c44e52", label=r"$\pi_{M_0}$ raw")
            ax.scatter([r["excess_rescaled_pct"] for r in sub], y, s=46, zorder=4,
                       color="#dd8452", marker="D", label=r"$\pi_{M_0}$ rescaled")
            ax.set_yticks(y, [f"{r['alpha0_deg']:.0f}$^\\circ$, "
                              f"{r['vnorm0']:.2f}" for r in sub], fontsize=9)
            ax.set_xlabel(r"excess altitude loss over $\pi_{M_1}$ (%)")
            ax.set_title(rf"$m/m_0 = {mf:.2f}$  (${715.3152 * mf:.0f}$ kg)")
            ax.grid(alpha=0.3, axis="x")
            ax.invert_yaxis()
        axs[0].set_ylabel(r"entry $(\alpha_0,\ V_0/V_s)$")
        axs[-1].legend(loc="lower right", framealpha=0.95)
        fig.suptitle(r"Retraining buys what rescaling one scalar does not",
                     fontsize=12)
        fig.tight_layout()
        from symmetric_stall.procedures import stamp_engine
        stamp_engine(fig, engine_tau=ENGINE_TAU, elevator_tau=ELEVATOR_TAU)
        for ext in ("png", "pdf"):
            fig.savefig(f"{out_stem}.{ext}", dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(f"[+] {out_stem.name}.{{png,pdf}} written")


if __name__ == "__main__":
    main()
