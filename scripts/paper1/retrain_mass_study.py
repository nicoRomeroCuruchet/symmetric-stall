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

FOUR PERTURBATIONS, NOT ONE. Mass alone is the weakest form of the
objection, because it is the one parameter the policy already sees through
Vs. The centre of gravity is the harder case: nothing in the observation
carries it, so a CG-shifted aeroplane is invisible to the nominal policy and
there is no scalar to correct. The cases are ordered so the two effects can
be separated -- mass alone, CG alone, then both at once -- and the combined
case is the loading the paper should be judged on: heaviest permitted weight
with the CG at the forward limit.

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

#: Riley's power-off stall boundary is 14 deg, so the paper's default entry
#: grid starts 2 deg past it. That is a mild upset, and on a 10% lighter
#: aeroplane the shallowest cells stop being recoveries at all: the value
#: function puts the optimum at 1.04 m there, and on an ideal engine the
#: aircraft climbs away instead of descending. The comparison then has almost
#: nothing to resolve, and the per-cent axis divides the residue by a very
#: small number -- which is where the impossible negative cells came from.
#:
#: This study therefore enters deep, 6 to 18 deg past the boundary. Every
#: entry recovers at every mass, and the sign anomaly disappears above
#: 24 deg. Set before procedures is imported, since it reads the grid at
#: import time.
os.environ["STALL_ALPHA_GRID"] = "20,26,32"

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

#: Chord of the AA-1 (m). The CG offsets below are quoted in metres because
#: that is what the plant and the .npz metadata carry; this converts them to
#: the per-cent-of-chord the paper and the figures speak in.
CHORD_M = 1.2192

#: One entry per retrained policy: (mass factor, CG offset in metres AFT of
#: Riley's reference, path). Negative CG is FORWARD, towards the nose --
#: the plant variable is aft-positive, the figures are not, and the sign is
#: converted at the presentation layer only.
#:
#: Ordered mass -> CG -> both, which is the order the argument is made in.
CASES = [
    (0.90, 0.0, POLICY_DIR / f"{STEM}_mass090.npz"),
    (1.10, 0.0, POLICY_DIR / f"{STEM}_mass110.npz"),
    (1.00, -0.12192, POLICY_DIR / f"{STEM}_cg--0.12192_0_0.npz"),
    (1.15, -0.12192, POLICY_DIR / f"{STEM}_cg--0.12192_0_0_mass115.npz"),
]


def case_label(mf: float, cg_aft: float) -> str:
    """Panel and table label for one perturbation, CG forward-positive."""
    cg_pct = -100.0 * cg_aft / CHORD_M
    if abs(cg_pct) < 1e-9:
        return rf"$m/m_0={mf:.2f}$, CG nominal"
    where = "fwd" if cg_pct > 0 else "aft"
    return rf"$m/m_0={mf:.2f}$, CG {abs(cg_pct):.0f}% {where}"

OUT = REPO / "results" / "6_riley_engine" / "retrain-mass"


def perturbed_plant(mass_factor: float, cg_aft: float,
                    rescale_vs: bool) -> SymmetricStall:
    """The aeroplane at `mass_factor` and `cg_aft`, with or without a corrected Vs.

    MASS is scaled AFTER construction on purpose: STALL_AIRSPEED is derived
    from the mass in __init__, so assigning here leaves it at the nominal
    value, which is precisely the un-rescaled case. `rescale_vs` then puts it
    right, and sqrt is exact because Vs goes as the square root of weight.

    Do NOT reach for the MASS_FACTOR environment variable here. That rebuilds
    the whole aeroplane consistently, which is what TRAINING wants and what
    makes the un-rescaled arm impossible to express.

    CG_AFT is assigned on the INSTANCE, shadowing the class attribute that
    grumman.py reads from the environment at import time. That is the only way
    to fly several centres of gravity in one process; setting CG_AFT_M in the
    environment here would be silently ignored, the plant having been imported
    already.

    There is no rescaling counterpart for the CG. Vs is a function of weight,
    so a mass change leaves a correctable trace in the observation; a CG
    change leaves none, which is exactly why it is the harder case.
    """
    env = SymmetricStall()
    env.airplane.MASS = env.airplane.MASS * mass_factor
    env.airplane.CG_AFT = float(cg_aft)
    if rescale_vs:
        env.airplane.STALL_AIRSPEED *= float(np.sqrt(mass_factor))
    return env


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pi_nom = paths.load_policy(NOMINAL, env=SymmetricStall())

    rows = []
    for mf, cg_aft, path in CASES:
        if not path.exists():
            print(f"[!] no retrained policy for {case_label(mf, cg_aft)}: "
                  f"{path.name} -- skipped")
            continue
        pi_ret = paths.load_policy(path, env=SymmetricStall())
        # The .npz records what it was SOLVED for. Checking both parameters
        # here is what stops a mislabelled file from being read as the floor
        # for an aeroplane it was never trained on -- the failure this whole
        # study would report as a result rather than a bug.
        meta = json.loads(str(np.load(path)["run_metadata"].ravel()[0]))
        assert float(meta.get("mass_factor", 1.0)) == mf, (
            f"{path.name} says mass_factor={meta.get('mass_factor')}, expected {mf}")
        assert abs(float(meta.get("cg_aft_m", 0.0)) - cg_aft) < 1e-9, (
            f"{path.name} says cg_aft_m={meta.get('cg_aft_m')}, expected {cg_aft}")

        print(f"\n=== {case_label(mf, cg_aft)}  ({715.3152 * mf:.1f} kg, "
              f"cg_aft {cg_aft:+.5f} m) ===")
        print(f"{'alpha0':>7} {'v0/Vs':>6} | {'pi_M0 raw':>10} {'pi_M0 resc':>11} "
              f"{'pi_M1':>9} | {'raw vs M1':>10} {'resc vs M1':>11}")
        for a0 in ALPHA_GRID_DEG:
            for v0 in VNORM_GRID:
                # Same physical entry speed in all three arms.
                h_raw = rollout(perturbed_plant(mf, cg_aft, False), pi_nom,
                                ctrl_optimal, a0, v0 * float(np.sqrt(mf)))["h"]
                h_res = rollout(perturbed_plant(mf, cg_aft, True), pi_nom,
                                ctrl_optimal, a0, v0)["h"]
                h_ret = rollout(perturbed_plant(mf, cg_aft, True), pi_ret,
                                ctrl_optimal, a0, v0)["h"]
                # Excess over the retrained floor, per cent of the floor's own
                # loss. Losses are negative, so the ratio is written on
                # magnitudes to keep "more loss" positive.
                exc_raw = 100.0 * (abs(h_raw) - abs(h_ret)) / abs(h_ret)
                exc_res = 100.0 * (abs(h_res) - abs(h_ret)) / abs(h_ret)
                rows.append({"mass_factor": mf, "cg_aft_m": cg_aft,
                             "alpha0_deg": a0, "vnorm0": v0,
                             "h_pi_M0_raw": h_raw, "h_pi_M0_rescaled": h_res,
                             "h_pi_M1": h_ret,
                             "excess_raw_pct": exc_raw,
                             "excess_rescaled_pct": exc_res,
                             # Metres, which is what the figure plots. The
                             # per-cent form divides by the retrained loss and
                             # so inflates wherever that loss is small; the
                             # metres are the same number without a
                             # denominator that varies by a factor of ten
                             # across the panel.
                             "excess_raw_m": abs(h_raw) - abs(h_ret),
                             "excess_rescaled_m": abs(h_res) - abs(h_ret),
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
    """The sentences the paper needs, computed rather than eyeballed.

    Reported in metres first. The per-cent form is kept because a reader will
    ask for it, but it is the derived quantity: its denominator is the
    retrained loss, which varies by a factor of four across this study, so the
    same centimetres read as very different percentages depending on which
    cell they land in.
    """
    for mf, cg_aft, _ in CASES:
        sub = [r for r in rows
               if r["mass_factor"] == mf and r["cg_aft_m"] == cg_aft]
        if not sub:
            continue
        raw_m = np.array([r["excess_raw_m"] for r in sub])
        res_m = np.array([r["excess_rescaled_m"] for r in sub])
        raw_p = np.array([r["excess_raw_pct"] for r in sub])
        res_p = np.array([r["excess_rescaled_pct"] for r in sub])
        floor = np.array([abs(r["h_pi_M1"]) for r in sub])
        print(f"\n{case_label(mf, cg_aft)}: over {len(sub)} entries, "
              f"retrained loss {floor.min():.1f}-{floor.max():.1f} m")
        print(f"  pi_M0 raw       {raw_m.min():+6.2f} .. {raw_m.max():+6.2f} m"
              f"   ({raw_p.min():+5.1f} .. {raw_p.max():+5.1f} %)")
        print(f"  pi_M0 rescaled  {res_m.min():+6.2f} .. {res_m.max():+6.2f} m"
              f"   ({res_p.min():+5.1f} .. {res_p.max():+5.1f} %)")
        print(f"  -> rescaling recovers {np.median(raw_m) - np.median(res_m):.2f} m "
              f"of the median gap; retraining is worth the "
              f"{np.median(res_m):.2f} m that remain")


def make_table(rows, out: Path) -> None:
    """LaTeX table: the deliverable. One row per (mass, entry).

    Numbers rather than a figure because the claim is per-cell -- "in none of
    the eighteen entries does retraining recover more than X" is something a
    reader checks, and a plot cannot be checked.
    """
    lines = [
        r"\begin{table}[H]",
        r"\caption{Retraining the optimum for a perturbed loading. $\pi_{M_0}$",
        r"is the nominal policy (715.3~kg, reference CG); $\pi_{M_1}$ is",
        r"re-solved for the mass and centre of gravity in the first two",
        r"columns, $\Delta x_{cg}$ positive towards the nose. All three arms",
        r"fly the SAME aircraft and enter",
        r"at the same true airspeed. \emph{raw} carries the nominal stall-speed",
        r"normalisation; \emph{rescaled} corrects that single scalar from the",
        r"pre-flight weight. Excess is over $\pi_{M_1}$, the floor, and is",
        r"quoted in metres: the retrained loss varies by a factor of four",
        r"across the study, so the same disagreement reads as very different",
        r"percentages depending on the cell. The entries are deep, $6$ to",
        r"$18^\circ$ past the $14^\circ$ stall boundary, so that every cell",
        r"is a recovery the policy has to work for. A handful of cells come",
        r"out a few centimetres below zero, which is impossible against a",
        r"policy optimal for that aeroplane and is the integration step of",
        r"the rollout.}",
        r"\label{tab:retrain_mass}",
        r"\centering",
        r"\begin{tabular}{cccrrrrr}",
        r"\hline",
        r"$m/m_0$ & $\Delta x_{cg}$ & $(\alpha_0,\,V_0/V_s)$ & "
        r"$\pi_{M_0}$ raw & $\pi_{M_0}$ resc. & $\pi_{M_1}$ & raw & resc. \\",
        r" & (\% $\bar{c}$) & (deg, --) & (m) & (m) & (m) & (\%) & (\%) \\",
        r"\hline",
    ]
    # Grouped in the order CASES declares -- mass, CG, both -- not sorted, so
    # the table reads as the argument is made rather than as the floats sort.
    for mf, cg_aft, _ in CASES:
        block = [x for x in rows
                 if x["mass_factor"] == mf and x["cg_aft_m"] == cg_aft]
        if not block:
            continue
        # -0.0 formats as "-0", which reads as a CG offset that is not there.
        cg_pct = -100.0 * cg_aft / CHORD_M
        cg_pct = 0.0 if cg_pct == 0.0 else cg_pct
        for r in block:
            star = r"$^\ast$" if r["canonical"] else ""
            lines.append(
                f"{mf:.2f} & {cg_pct:+.0f} & "
                f"$({r['alpha0_deg']:.0f},\\,{r['vnorm0']:.2f})${star} & "
                f"{r['h_pi_M0_raw']:.2f} & {r['h_pi_M0_rescaled']:.2f} & "
                f"{r['h_pi_M1']:.2f} & {r['excess_raw_pct']:+.1f} & "
                f"{r['excess_rescaled_pct']:+.1f} \\\\")
        lines.append(r"\hline")
    lines += [r"\end{tabular}", r"\end{table}",
              r"% $^\ast$ canonical entry."]
    out.write_text("\n".join(lines) + "\n")
    print(f"[+] {out.name} written")


def make_dumbbell(rows, out_stem: Path) -> None:
    """Excess over the retrained floor IN METRES, one row per entry.

    The first version of this plotted per cent of the retrained loss, and the
    axis was doing damage. That loss varies by a factor of ten across the
    study, from 9 m at the lightest and mildest entry to 87 m at the heaviest
    and deepest, so a fixed disagreement of a few centimetres reads as 0.2%
    at one end of a panel and 3.4% at the other. Two cells even came out
    NEGATIVE, which is impossible against a policy that is optimal for that
    aeroplane, and the figure had to carry a shaded band and a paragraph
    explaining that the impossibility was a resolution artefact.

    In metres there is nothing to explain. The whole effect fits inside half
    a metre, the sign anomaly shrinks to 2 cm, and the claim the figure makes
    stops depending on what it is divided by: retraining buys centimetres.

    ONE SERIES, not two. The raw arm is not plotted: it answers a different
    question -- what happens if nobody updates Vs -- and the robustness matrix
    already covers that case, since perturbed_env leaves the nominal Vs in
    place. Here the question is whether the optimum holds away from the point
    it was solved at, and for that the honest pi_M0 is the rescaled one: no
    aeroplane flies without a weight and balance, so the raw arm charges the
    policy for an error the procedure it is compared against would not make.
    It stays in the table, where a reader who asks "and if you did not
    rescale?" finds the answer without the figure having to carry it.

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
    KEY = "excess_rescaled_m"
    # One panel per perturbation, in the order CASES declares. Four panels in
    # a single row would be 21 inches wide and unreadable at journal column
    # width, so anything past two wraps into a grid.
    cases = [(mf, cg, [r for r in rows
                       if r["mass_factor"] == mf and r["cg_aft_m"] == cg])
             for mf, cg, _ in CASES]
    cases = [c for c in cases if c[2]]
    ncols = 2 if len(cases) > 2 else max(len(cases), 1)
    nrows = int(np.ceil(len(cases) / ncols))
    # Shared limits, wide enough that the band reads as a band rather than as
    # the background, and never narrower than the band itself.
    spread = max(abs(r[KEY]) for _, _, sub in cases for r in sub)
    xmax = spread * 1.30
    with plt.rc_context(rc):
        fig, axs = plt.subplots(nrows, ncols, squeeze=False,
                                figsize=(5.4 * ncols, 4.2 * nrows))
        flat = axs.ravel()
        for ax, (mf, cg_aft, sub) in zip(flat, cases):
            y = np.arange(len(sub))
            ax.axvline(0.0, color="0.4", lw=1.0, zorder=1)
            for k, r in enumerate(sub):
                ax.plot([0.0, r[KEY]], [k, k],
                        color="0.6", lw=2.0, zorder=2, solid_capstyle="round")
            ax.scatter([r[KEY] for r in sub], y, s=52, zorder=4,
                       color="#dd8452", marker="D",
                       label=r"$\pi_{M_0}$, $V_s$ rescaled")
            # The floor goes in the tick label. Plotting the excess in metres
            # removes the denominator the per-cent axis carried implicitly, and
            # without it 0.4 m is uncalibrated: a reader cannot tell whether it
            # is a fifth of the manoeuvre or a hundredth. Each row now shows
            # what its own excess is an excess OVER.
            ax.set_yticks(y, [f"{r['alpha0_deg']:.0f}$^\\circ$, "
                              f"{r['vnorm0']:.2f}   ({abs(r['h_pi_M1']):.0f} m)"
                              for r in sub], fontsize=8)
            ax.set_xlabel(r"excess altitude loss over $\pi_{M_1}$ (m)")
            floor = [abs(r["h_pi_M1"]) for r in sub]
            ax.set_title(f"{case_label(mf, cg_aft)}  ({715.3152 * mf:.0f} kg)\n"
                         f"retrained policy loses {min(floor):.0f} to "
                         f"{max(floor):.0f} m", fontsize=10)
            ax.grid(alpha=0.3, axis="x")
            ax.invert_yaxis()
            ax.set_xlim(-xmax, xmax)
            # the retrained policy IS the origin, so say so
            ax.annotate(r"$\pi_{M_1}$", xy=(0.0, 1.0),
                        xycoords=("data", "axes fraction"),
                        xytext=(4, -10), textcoords="offset points",
                        ha="left", va="top", fontsize=9, color="0.35")
        for ax in flat[len(cases):]:     # an odd case count leaves a hole
            ax.set_visible(False)
        for row in range(nrows):
            axs[row, 0].set_ylabel(r"entry $(\alpha_0,\ V_0/V_s)$, "
                                   r"and its $\pi_{M_1}$ loss", fontsize=10)
        flat[len(cases) - 1].legend(loc="lower right", framealpha=0.95)
        fig.suptitle(r"Excess of the nominal policy over one retrained "
                     r"for the same aircraft", fontsize=12)
        fig.tight_layout()
        from symmetric_stall.procedures import stamp_engine
        stamp_engine(fig, engine_tau=ENGINE_TAU, elevator_tau=ELEVATOR_TAU)
        for ext in ("png", "pdf"):
            fig.savefig(f"{out_stem}.{ext}", dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(f"[+] {out_stem.name}.{{png,pdf}} written")


if __name__ == "__main__":
    main()
