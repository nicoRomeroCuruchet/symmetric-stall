"""Every number Sec. `time_domain` asserts, recomputed from the rollouts.

The section is a prose description of one trajectory, and its sentences carry
about fifteen figures: where the angle of attack settles, how deep each arm
dives, what the mean elevator deflection is over the sliding arc. Those came
from a run under an instantaneous engine and on an initial-condition grid
that Riley's thrust model has since invalidated -- at V_0 = 0.95 V_s the
aeroplane climbs away instead of stalling -- so none of them survive, and
reading the replacements off a rendered figure would only start the same
decay again.

This computes them. One rollout per arm at the canonical entry, on the
plant the paper is now defined against (Riley's engine lag, tau_e = 0.85 s,
instantaneous elevator), printing each quantity beside the claim it has to
replace so the two can be compared line by line.

The elevator lag is deliberately absent. It enters the paper as a
sensitivity study, not as the baseline: every table, heat map and robustness
result is computed at tau_de = 0, and mixing one figure from a different
plant into that set is what produced the present situation.

Usage:
    python3 scripts/paper1/audit_time_domain.py
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

runconfig.apply(thrust="riley", engine_tau=ENGINE_TAU)

import numpy as np  # noqa: E402

from symmetric_stall import paths  # noqa: E402
from symmetric_stall.aircraft.symmetric_stall import SymmetricStall  # noqa: E402
from symmetric_stall.procedures import (  # noqa: E402
    CANONICAL, ctrl_optimal, make_maneuver, rollout,
)

#: The power-off stall boundary of the Riley tables. The section calls it
#: alpha_s and leans on it twice, so it is named once here.
ALPHA_S_DEG = 14.0

OUT = REPO / "results" / "6_riley_engine"


def summarise(res, name):
    """The quantities the prose talks about, from one recorded rollout."""
    h = res["hist"]
    t = np.array(h["t"])
    alpha = np.rad2deg(np.array(h["alpha"]))
    gamma = np.rad2deg(np.array(h["gamma"]))
    de = np.rad2deg(np.array(h["de"]))
    dt_eff = np.array(h["dt_eff"])
    dt_cmd = np.array(h["dt_ctrl"])

    # The nose-down phase ends when alpha first reaches the stall boundary;
    # the sliding arc is everything after it. Defining the split from the
    # trajectory rather than from a hand-picked time is what lets the mean
    # deflection below be quoted without an arbitrary window.
    below = np.where(alpha <= ALPHA_S_DEG)[0]
    i_s = int(below[0]) if below.size else 0
    arc = slice(i_s, None)

    return {
        "arm": name,
        "h_m": float(res["h"]),
        "t_s": float(res["t"]),
        "status": res["status"],
        "t_reach_alpha_s": float(t[i_s]),
        "alpha_min_deg": float(alpha.min()),
        "alpha_max_deg": float(alpha.max()),
        "gamma_min_deg": float(gamma.min()),
        "alpha_mean_arc_deg": float(alpha[arc].mean()),
        "alpha_max_arc_deg": float(alpha[arc].max()),
        "de_mean_arc_deg": float(de[arc].mean()),
        "de_min_deg": float(de.min()),
        "de_max_deg": float(de.max()),
        # How long the engine takes to deliver what was asked for: the lag is
        # the whole reason the numbers moved, so it is reported, not implied.
        "t_cmd_full_s": float(t[np.argmax(dt_cmd >= 0.999)])
        if (dt_cmd >= 0.999).any() else float("nan"),
        "t_eff_90pct_s": float(t[np.argmax(dt_eff >= 0.9)])
        if (dt_eff >= 0.9).any() else float("nan"),
    }


def main() -> None:
    a0, v0 = CANONICAL
    env = SymmetricStall()
    pi = paths.load_policy(env=env)

    print(f"canonical entry: gamma_0 = 0, alpha_0 = {a0:.0f} deg, "
          f"V_0 = {v0:.2f} V_s, q_0 = 0")
    print(f"plant: Riley thrust, engine tau_e = {ENGINE_TAU} s, "
          f"instantaneous elevator\n")

    # The optimum flies first, because the scripted pilots are capped at ITS
    # own deepest pull. Without that cap the comparison confounds WHEN power
    # is applied with HOW HARD the pilot hauls back, and the figure's caption
    # already claims the cap, so it has to be the one actually flown.
    dp_res = rollout(env, pi, ctrl_optimal, a0, v0, record=True,
                     engine_tau=ENGINE_TAU)
    de_cap = float(np.min(dp_res["hist"]["de"]))
    print(f"pull authority of the optimum at this entry: "
          f"{np.rad2deg(de_cap):.2f} deg (the scripted pilots are capped there)\n")

    arms = {
        "CAA (alpha-hold)": make_maneuver(power_start="t0", pull="alpha_hold",
                                          de_pull_limit=de_cap),
        "FAA (alpha-hold)": make_maneuver(power_start="unstall", pull="alpha_hold",
                                          de_pull_limit=de_cap),
        # The full-pull arms are the degenerate technique, so they go to the
        # actuator stop on purpose: capping them would remove the very thing
        # they measure.
        "CAA (full pull)": make_maneuver(power_start="t0", pull="full"),
        "FAA (full pull)": make_maneuver(power_start="unstall", pull="full"),
    }

    rows = [summarise(dp_res, "DP optimum")]
    for name, ctrl in arms.items():
        res = rollout(env, pi, ctrl, a0, v0, record=True, engine_tau=ENGINE_TAU)
        rows.append(summarise(res, name))

    w = 20
    keys = [("h_m", "dh (m)", "%8.2f"), ("t_s", "t (s)", "%8.2f"),
            ("status", "status", "%9s"),
            ("gamma_min_deg", "gamma_min (deg)", "%8.2f"),
            ("alpha_min_deg", "alpha_min (deg)", "%8.2f"),
            ("alpha_max_deg", "alpha_max (deg)", "%8.2f"),
            ("t_reach_alpha_s", "t to alpha_s (s)", "%8.2f"),
            ("alpha_mean_arc_deg", "mean alpha, arc", "%8.2f"),
            ("alpha_max_arc_deg", "max alpha, arc", "%8.2f"),
            ("de_mean_arc_deg", "mean de, arc (deg)", "%8.2f"),
            ("t_cmd_full_s", "t cmd full thr (s)", "%8.2f"),
            ("t_eff_90pct_s", "t engine 90% (s)", "%8.2f")]
    print(" " * w + "".join(f"{r['arm']:>20s}" for r in rows))
    for k, label, fmt in keys:
        line = f"{label:<{w}}"
        for r in rows:
            v = r[k]
            line += (f"{v:>20s}" if isinstance(v, str)
                     else f"{(fmt % v):>20s}")
        print(line)

    dp, caa, faa = rows[0], rows[1], rows[2]
    print("\nderived, as the section states them")
    print(f"  optimum over CAA           : {abs(caa['h_m']) - abs(dp['h_m']):6.2f} m")
    print(f"  optimum over FAA           : {abs(faa['h_m']) - abs(dp['h_m']):6.2f} m")
    print(f"  CAA over FAA               : {abs(faa['h_m']) - abs(caa['h_m']):6.2f} m")
    print(f"  CAA as a fraction of FAA   : {abs(caa['h_m']) / abs(faa['h_m']):6.3f}")
    print(f"  procedure dive vs optimum  : "
          f"{caa['gamma_min_deg'] / dp['gamma_min_deg']:.2f}x (CAA), "
          f"{faa['gamma_min_deg'] / dp['gamma_min_deg']:.2f}x (FAA)")
    print(f"  FAA power gating delay     : "
          f"{faa['t_cmd_full_s'] - caa['t_cmd_full_s']:6.2f} s")

    (OUT / "time_domain_audit.json").write_text(json.dumps(
        {"run_config": runconfig.describe(), "canonical": {
            "alpha0_deg": a0, "vnorm0": v0}, "arms": rows}, indent=1))
    print(f"\n[+] time_domain_audit.json written to {OUT}")


if __name__ == "__main__":
    main()
