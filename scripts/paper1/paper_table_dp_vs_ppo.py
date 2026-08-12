"""
paper_table_dp_vs_ppo.py — Altitude-loss table: DP (Policy Iteration) vs PPO
over a grid of initial conditions (4-DOF symmetric stall, Riley model).

Grid: alpha_0 in {16, 18, 20} deg  x  V_0/Vs in {0.90, 0.95, 1.00},
gamma_0 = 0 deg, q_0 = 0 deg/s (same convention as the canonical IC).

Output: results/paper/table_dp_vs_ppo.tex (LaTeX booktabs-style tabular)
and a plain-text summary on stdout.
"""
import logging
from pathlib import Path
from typing import Callable

import numpy as np

from symmetric_stall.aircraft.symmetric_stall import SymmetricStall
from symmetric_stall.utils.recovery import RecoveryMonitor
from symmetric_stall.utils.utils import get_optimal_action

logger = logging.getLogger(__name__)

ALPHA_GRID_DEG = [16.0, 18.0, 20.0]
VNORM_GRID = [0.90, 0.95, 1.00]
GAMMA_0_DEG = 0.0
Q_0_DEG = 0.0
MAX_TIME = 15.0


def simulate(env: SymmetricStall, get_action: Callable,
             alpha0_deg: float, vnorm0: float) -> dict:
    """Altitude loss until first return to level flight (gamma = 0).

    Consistent with the infinite-horizon formulation: the cost accumulates
    only until the trajectory enters the absorbing set {gamma >= 0}, so the
    episode stops at the first upward gamma crossing after a dive deeper
    than DIVE_THRESHOLD_DEG (the threshold filters the initial powered
    transient where gamma hovers around zero before the dive develops).
    Same rule as PPO_vs_PI.run_simulation, with the threshold relaxed from
    -2 deg so shallow upsets (V0/Vs = 1.0) also register their dive.
    The rule itself lives in utils.recovery, shared with paper_procedures."""
    v_stall = env.airplane.STALL_AIRSPEED
    step_dt = env.airplane.TIME_STEP

    obs, _ = env.specific_reset(
        np.deg2rad(GAMMA_0_DEG), vnorm0,
        np.deg2rad(alpha0_deg), np.deg2rad(Q_0_DEG),
    )

    t, h = 0.0, 0.0
    stop = RecoveryMonitor(step_dt)

    while t < MAX_TIME:
        action = get_action(obs)
        obs, _, _, _, _ = env.step(action)
        h += obs[1] * v_stall * np.sin(obs[0]) * step_dt
        t += step_dt

        if stop.update(np.rad2deg(obs[0])):
            return {"h": h, "t": t, "status": "recovered"}

        if (obs[2] >= np.deg2rad(40) or obs[2] <= np.deg2rad(-40)
                or obs[0] <= -np.pi + 0.05):
            return {"h": h, "t": t, "status": "crash"}

    return {"h": h, "t": t, "status": "timeout"}


def main():
    # Imported lazily: torch/SB3 are needed only by this PPO comparison, so
    # the other paper scripts that reuse the constants above stay dependency-free.
    from PPO_vs_PI import load_pi
    from stable_baselines3 import PPO

    pi = load_pi(Path("results/SymmetricStall_policy.npz"))
    ppo_model = PPO.load(Path("policy_symmetric_stall/models/best_model.zip"),
                         device="cpu")
    ppo_env = SymmetricStall()

    def pi_action(obs):
        return get_optimal_action(obs, pi)[0]

    def ppo_action(obs):
        return ppo_model.predict(obs, deterministic=True)[0]

    rows = []
    for alpha0 in ALPHA_GRID_DEG:
        for vnorm0 in VNORM_GRID:
            r_pi = simulate(pi.env, pi_action, alpha0, vnorm0)
            r_ppo = simulate(ppo_env, ppo_action, alpha0, vnorm0)
            # The reduction is only meaningful when both policies complete
            # the recovery; at timeout h(15 s) is an arbitrary snapshot.
            both_ok = (r_pi["status"] == r_ppo["status"] == "recovered")
            gap = (100.0 * (r_ppo["h"] - r_pi["h"]) / r_ppo["h"]
                   if both_ok else float("nan"))
            rows.append((alpha0, vnorm0, r_pi, r_ppo, gap))
            logger.info(
                f"a0={alpha0:4.0f}  V0={vnorm0:.2f}  "
                f"PI: {r_pi['h']:6.2f} m ({r_pi['status']:9s})  "
                f"PPO: {r_ppo['h']:6.2f} m ({r_ppo['status']:9s})  "
                f"gap: {gap:5.1f}%")

    out_dir = Path("results/paper")
    out_dir.mkdir(parents=True, exist_ok=True)
    tex_path = out_dir / "table_dp_vs_ppo.tex"

    def fmt(r):
        mark = {"recovered": "", "crash": r"\textsuperscript{c}",
                "timeout": r"\textsuperscript{t}"}[r["status"]]
        return f"{r['h']:.2f}{mark}"

    lines = [
        r"\begin{table}[hbt!]",
        r"    \centering",
    r"    \caption{Altitude loss $\Delta h$ (m) until first return to "
        r"level flight ($\gamma = 0$) for the 4-DOF symmetric stall recovery "
        r"with the Riley aerodynamic model: exact DP (Policy Iteration) "
        r"vs.\ PPO, from initial conditions $\gamma_0 = 0^\circ$, $q_0 = 0$.}",
        r"    \label{tab:dp_vs_ppo}",
        r"    \begin{tabular}{ccccc}",
        r"        \hline",
        r"        $\alpha_0$ (deg) & $V_0/V_s$ & DP $\Delta h$ (m) & "
        r"PPO $\Delta h$ (m) & Reduction (\%) \\",
        r"        \hline",
    ]
    for alpha0, vnorm0, r_pi, r_ppo, gap in rows:
        gap_str = "--" if np.isnan(gap) else f"{gap:.1f}"
        lines.append(
            f"        {alpha0:.0f} & {vnorm0:.2f} & {fmt(r_pi)} & "
            f"{fmt(r_ppo)} & {gap_str} \\\\")
    lines += [
        r"        \hline",
        r"        \multicolumn{5}{l}{\footnotesize\textsuperscript{t}\,No "
        r"return to level flight within 15 s: both policies settle into a"
        r" shallow}\\",
        r"        \multicolumn{5}{l}{\footnotesize powered descent "
        r"($|\gamma| < 1^\circ$, $V \approx 0.98\,V_s$); the value is "
        r"$h(15\,\mathrm{s})$.}\\",
        r"    \end{tabular}",
        r"\end{table}",
    ]
    tex_path.write_text("\n".join(lines) + "\n")
    logger.info(f"[+] LaTeX table written to {tex_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
