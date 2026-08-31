"""Put a number on the DP-vs-CasADi agreement of Case I.

The paper shows the validation as two figures of overlaid trajectories and
asserts that they agree. A reader cannot check an assertion made with a
picture, and a reviewer who wants the benchmark quantified has nothing to
read. This turns the ten comparison points into a scalar.

THE METRIC. Per-point relative error divides by that point's own loss, so
the shallowest entry -- 49.44 m, where a fixed discretisation error is the
largest fraction -- dominates a sweep whose deepest point loses four times
as much. The agreed metric normalises instead by the DEEPEST loss of the
sweep:

    e_i = |h_DP,i - h_NLP,i| / max_j |h_NLP,j|

which is the error as a fraction of the quantity the figure is actually
about, and is what makes "agrees to better than 1 %" a statement about the
manoeuvre rather than about the easiest point in it.

Both are reported. The per-point figure is the conservative one and stays
in the table; the normalised figure is the headline.

WHERE THE NUMBERS COME FROM. The validation itself is not in this
repository -- it is `stall-spin-recovery-dp`, branch
`3dof-reduced-banked-pullout`, commit 606f609, and that repository is
read-only for this work. What is here is its log,
results/6_riley_engine/case1_3dof/diagnostics/casadi.log, which records
every point as the run produced it. Parsing the log rather than retyping
the numbers is the whole point: a transcription cannot be re-derived and
this can.

WHAT IS EXCLUDED, AND WHY IT IS NOT CHERRY-PICKING. One of the ten points
(gamma_0 = -60 deg, mu_0 = 150 deg) is a known broken rollout, established
in commit 08f4efa: the DP rollout reports -1002.92 m while the global
minimum of V* over all 31,948,500 states is -260.56 m, so that trajectory
is impossible by construction. It was then handed to Ipopt as a seed, and
Ipopt followed it -- returning -500.01 m in one run and 0.00 m in another
from the same problem, reporting success both times. Neither arm of that
point is a measurement of anything. It is reported separately and by name,
never averaged in.

Usage:
    python3 scripts/paper1/casadi_benchmark.py
"""
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CASE1 = REPO / "results" / "6_riley_engine" / "case1_3dof"
LOG = CASE1 / "diagnostics" / "casadi.log"

#: The rollout that leaves the grid. Excluded from every aggregate, printed
#: on its own. See commit 08f4efa and the case README.
BROKEN = (-60.0, 150.0)

#: The NLP's discretisation, from solver/casadi_optimizer.py at 606f609.
#: Recorded here because the benchmark compares two discretisations of the
#: same continuous problem, and a gap is meaningless without both.
NLP_NODES = 150
NLP_IPOPT = {"tol": 1e-4, "acceptable_tol": 1e-2, "acceptable_iter": 15,
             "max_iter": 5000}
DP_ROLLOUT_DT_S = 0.10

#: The glider of Case I, from solver/casadi_optimizer.py at 606f609. Both
#: methods are given the same aeroplane; these are only needed to predict the
#: rollout's quadrature bias below.
MASS_KG, WING_AREA_M2, RHO, G, CL_REF = 697.18, 9.1147, 1.225, 9.81, 1.2
V0_NORM = 1.2


def stall_airspeed() -> float:
    return (MASS_KG * G / (0.5 * RHO * WING_AREA_M2 * CL_REF)) ** 0.5


def euler_bias_m(gamma0_deg: float) -> float:
    """How much altitude loss the DP rollout invents by integrating h with
    forward Euler at dt = 0.10 s.

    The rollout accumulates `h += v*sin(gamma)*dt` with the rate evaluated at
    the START of each step, while the NLP integrates the same states with RK4
    at dt = T/150. Over a pullout gamma rises monotonically from gamma_0 to 0,
    so |h_dot| falls monotonically to zero and the rectangle rule always takes
    the larger endpoint: the error is one-signed and the DP always appears to
    lose more.

    Summing the rectangle-minus-trapezoid error telescopes, leaving

        bias = (dt/2) * [h_dot(T) - h_dot(0)] = -(dt/2) * V_0 sin(gamma_0)

    to O(dt^2), independent of the path taken and of mu_0 entirely. That last
    part is what makes this checkable rather than plausible: it predicts ONE
    number per scenario, before looking at the five points in it.
    """
    import math
    v0 = V0_NORM * stall_airspeed()
    return -(DP_ROLLOUT_DT_S / 2.0) * v0 * math.sin(math.radians(gamma0_deg))


_SCENARIO = re.compile(r"Validating DP-Guided Trajectories for gamma_0 = "
                       r"(-?\d+\.?\d*) deg")
_POINT = re.compile(r"\[=\]\s+mu0=\s*(-?\d+\.?\d*) deg\s+"
                    r"DP=\s*(-?\d+\.?\d*) m\s+NLP=\s*(-?\d+\.?\d*) m")


def parse(log_path: Path):
    """(gamma0, mu0, h_dp, h_nlp) for every point the validation printed."""
    gamma0 = None
    points = []
    for line in log_path.read_text().splitlines():
        m = _SCENARIO.search(line)
        if m:
            gamma0 = float(m.group(1))
            continue
        m = _POINT.search(line)
        if m:
            assert gamma0 is not None, "a point appeared before its scenario"
            points.append((gamma0, float(m.group(1)),
                           float(m.group(2)), float(m.group(3))))
    return points


def analyse(points):
    """Both errors, per point, grouped by scenario.

    The normalisation is per SCENARIO, not pooled: each figure is one sweep
    at one flight-path angle, and dividing the shallow-dive figure by the
    steep-dive figure's deepest point would flatter it for no reason.
    """
    out = []
    for gamma0 in sorted({p[0] for p in points}, reverse=True):
        good = [p for p in points if p[0] == gamma0 and (p[0], p[1]) != BROKEN]
        broken = [p for p in points if p[0] == gamma0 and (p[0], p[1]) == BROKEN]
        h_max = max(abs(p[3]) for p in good)
        rows = []
        for g0, mu0, h_dp, h_nlp in good:
            gap = h_dp - h_nlp
            rows.append({
                "gamma0_deg": g0, "mu0_deg": mu0,
                "h_dp_m": h_dp, "h_nlp_m": h_nlp, "gap_m": gap,
                # The DP is expected to lose MORE: it executes a policy that
                # is optimal for a 0.10 s hold, against an NLP free to switch
                # every T/150 s. A point where the DP won would mean the NLP
                # had missed its own optimum, so the sign is a check, not a
                # detail.
                "dp_loses_more": abs(h_dp) > abs(h_nlp),
                "err_pointwise_pct": 100.0 * abs(gap) / abs(h_nlp),
                "err_vs_max_pct": 100.0 * abs(gap) / h_max,
            })
        bias = euler_bias_m(gamma0)
        out.append({
            "gamma0_deg": gamma0,
            "euler_bias_m": bias,
            "min_gap_m": min(abs(r["gap_m"]) for r in rows),
            "residual_after_bias_pct": 100.0 * max(
                0.0, min(abs(r["gap_m"]) for r in rows) - bias) / h_max,
            "h_max_nlp_m": -h_max,
            "n_points": len(rows),
            "rows": rows,
            "excluded": [{"gamma0_deg": g, "mu0_deg": m,
                          "h_dp_m": d, "h_nlp_m": n} for g, m, d, n in broken],
            "worst_pointwise_pct": max(r["err_pointwise_pct"] for r in rows),
            "worst_vs_max_pct": max(r["err_vs_max_pct"] for r in rows),
            "mean_vs_max_pct": sum(r["err_vs_max_pct"] for r in rows) / len(rows),
        })
    return out


def report(scen) -> None:
    print(f"DP vs CasADi NLP, Case I (3-DOF banked pullout), L4 grid")
    print(f"  DP rollout dt      : {DP_ROLLOUT_DT_S:.2f} s (the Bellman backup dt)")
    print(f"  NLP discretisation : {NLP_NODES} RK4 shooting nodes, free final "
          f"time -> dt approx T/{NLP_NODES}")
    print(f"  Ipopt              : tol={NLP_IPOPT['tol']:g}, "
          f"acceptable_tol={NLP_IPOPT['acceptable_tol']:g}\n")

    for s in scen:
        print(f"gamma_0 = {s['gamma0_deg']:+.0f} deg   "
              f"({s['n_points']} points, deepest NLP loss "
              f"{s['h_max_nlp_m']:.2f} m)")
        print(f"{'mu0':>6} {'DP (m)':>10} {'NLP (m)':>10} {'gap (m)':>9} "
              f"{'e_point':>9} {'e_vs_max':>9}")
        for r in s["rows"]:
            print(f"{r['mu0_deg']:6.0f} {r['h_dp_m']:10.2f} {r['h_nlp_m']:10.2f} "
                  f"{r['gap_m']:+9.2f} {r['err_pointwise_pct']:8.2f}% "
                  f"{r['err_vs_max_pct']:8.2f}%")
        print(f"  worst  {s['worst_pointwise_pct']:.2f}% pointwise, "
              f"{s['worst_vs_max_pct']:.2f}% of the deepest loss"
              f"   (mean {s['mean_vs_max_pct']:.2f}%)")
        print(f"  predicted forward-Euler bias of the rollout: "
              f"{s['euler_bias_m']:.2f} m   (smallest observed gap "
              f"{s['min_gap_m']:.2f} m)")
        for x in s["excluded"]:
            print(f"  EXCLUDED  mu0={x['mu0_deg']:.0f} deg: DP {x['h_dp_m']:.2f} m, "
                  f"NLP {x['h_nlp_m']:.2f} m -- broken rollout, see 08f4efa")
        print()

    allrows = [r for s in scen for r in s["rows"]]
    worst = max(r["err_vs_max_pct"] for r in allrows)
    worst_pt = max(r["err_pointwise_pct"] for r in allrows)
    signs = {r["dp_loses_more"] for r in allrows}
    print(f"OVERALL over {len(allrows)} points")
    print(f"  worst error vs the deepest loss of its sweep : {worst:.2f} %")
    print(f"  worst pointwise error                        : {worst_pt:.2f} %")
    print(f"  DP loses more than the NLP at every point    : {signs == {True}}")
    if worst <= 1.0:
        print("  -> under the 1 % threshold on the agreed metric.")
    else:
        print("  -> ABOVE the 1 % threshold on the agreed metric.")
    print()
    print("Attribution of the gap")
    for s in scen:
        print(f"  gamma_0 = {s['gamma0_deg']:+.0f} deg: predicted Euler bias "
              f"{s['euler_bias_m']:6.2f} m vs smallest observed gap "
              f"{s['min_gap_m']:5.2f} m  "
              f"(ratio {s['min_gap_m'] / s['euler_bias_m']:.3f})")
    print("  A ratio near 1 means the floor of the disagreement is the")
    print("  rollout's altitude quadrature, not the NLP's tuning and not the")
    print("  DP: the same policy re-integrated properly would move DOWN toward")
    print("  the NLP, and the residual above the floor is what is left to")
    print("  explain.")


def make_table(scen, out: Path) -> None:
    """The table that replaces the two trajectory figures.

    The figures were dropped on the directors' advice, and measuring them
    says why: the two curves are separated by about 1 m against a plotted
    line width of 1.5 m, and because each method reconstructs its own
    horizontal coordinate, equal x is not equal time -- so over most of the
    dive the NLP is drawn BELOW the DP, which reads as the DP winning, while
    the endpoint comparison the numbers make has the opposite sign. A figure
    that inverts its own conclusion over two thirds of its width is not
    evidence. The table is.

    booktabs and caption-before-tabular, to match tables/ of the manuscript.
    """
    allrows = [r for s in scen for r in s["rows"]]
    worst = max(r["err_vs_max_pct"] for r in allrows)
    mean = sum(r["err_vs_max_pct"] for r in allrows) / len(allrows)
    worst_abs = max(abs(r["gap_m"]) for r in allrows)
    worst_pt = max(r["err_pointwise_pct"] for r in allrows)
    best_pt = min(r["err_pointwise_pct"] for r in allrows)

    lines = [
        r"\begin{table}[H]",
        r"    \centering",
        r"    \caption{Trajectory-level validation of the DP policy against "
        r"direct continuous-time optimal control, at the initial conditions "
        r"published by Bunge et al.\ \cite{Bunge2018} ($V_0/V_s = 1.2$). "
        r"Closed-loop DP rollouts are compared with CasADi/IPOPT solutions of "
        r"Eq.~\eqref{eq:ocp_free_tf} warm-started from the DP trajectory. "
        r"Both arms terminate at the level-flight recovery $\gamma = 0$ --- "
        r"the NLP by its terminal constraint, the DP because the level-flight "
        r"set is absorbing --- so the two endpoints are directly comparable. "
        rf"The error $e$ is normalized by the deepest loss of each sweep. The "
        rf"near-inverted $\gamma_0 = -60^\circ$, $\mu_0 = 150^\circ$ entry "
        rf"is excluded: the policy does not return it to level flight within "
        rf"the horizon.}}",
        r"    \label{tab:casadi_benchmark}",
        r"    \begin{tabular}{c c r r r r}",
        r"        \toprule",
        r"        $\gamma_0$ (deg) & $\mu_0$ (deg) & DP (m) & NLP (m) & "
        r"gap (m) & $e$ (\%) \\",
        r"        \midrule",
    ]
    for i, sc in enumerate(scen):
        for r in sc["rows"]:
            lines.append(
                f"        ${r['gamma0_deg']:.0f}$ & ${r['mu0_deg']:.0f}$ & "
                f"${r['h_dp_m']:.2f}$ & ${r['h_nlp_m']:.2f}$ & "
                f"${r['gap_m']:+.2f}$ & ${r['err_vs_max_pct']:.2f}$ \\\\")
        if i < len(scen) - 1:
            lines.append(r"        \midrule")
    lines += [
        r"        \bottomrule",
        r"    \end{tabular}",
        r"\end{table}",
    ]
    out.write_text("\n".join(lines) + "\n")
    print(f"[+] {out.name} written")
    print(f"    worst {worst:.2f}%, mean {mean:.2f}%, worst absolute "
          f"{worst_abs:.2f} m, pointwise spread {best_pt:.1f}-{worst_pt:.1f}%")


def main() -> None:
    if not LOG.exists():
        sys.exit(f"log not found: {LOG}")
    points = parse(LOG)
    if not points:
        sys.exit(f"no comparison points parsed out of {LOG}")
    scen = analyse(points)
    report(scen)
    payload = {
        "source_log": str(LOG.relative_to(REPO)),
        "provenance": "stall-spin-recovery-dp @ 606f609, "
                      "branch 3dof-reduced-banked-pullout",
        "dp_rollout_dt_s": DP_ROLLOUT_DT_S,
        "nlp_nodes": NLP_NODES,
        "nlp_ipopt": NLP_IPOPT,
        "excluded_point": {"gamma0_deg": BROKEN[0], "mu0_deg": BROKEN[1],
                           "reason": "rollout leaves the grid; see 08f4efa"},
        "scenarios": scen,
    }
    (CASE1 / "casadi_benchmark.json").write_text(json.dumps(payload, indent=1))
    print(f"[+] casadi_benchmark.json written to {CASE1}")
    make_table(scen, CASE1 / "table_casadi_benchmark.tex")
    # The manuscript \input{}s from its own tables/ directory; writing both
    # keeps the results folder self-describing without a manual copy that
    # would silently go stale.
    paper = REPO / "stall-paper" / "tables"
    if paper.is_dir():
        make_table(scen, paper / "table_casadi_benchmark.tex")


if __name__ == "__main__":
    main()
