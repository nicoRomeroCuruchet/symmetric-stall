"""
paper_mca_comparison.py — Fixed-dt vs MCA-endogenous-timestep comparison for
the 4-DOF symmetric stall recovery (Riley model).

Compares the two trained policies (same 56x41x60x41 grid, same reward):
  results/SymmetricStall_policy.npz       fixed macro dt = 0.01 s
  results/SymmetricStall_policy_mca.npz   MCA dt_h = 1/(sum |f_i|/h_i), <= 0.05 s

Three comparisons:
  1. Discretization-artifact metrics (islands / TV per control component)
     computed on the raw discrete policies (analysis.artifacts).
  2. Closed-loop altitude loss over the standard IC grid
     (alpha_0 x V_0/Vs, same grid and stopping rule as paper_table_dp_vs_ppo).
  3. Full-throttle fraction of the non-terminal state space (evidence for the
     CAA-structure claim: the DP argmax selects delta_t = 1 essentially
     everywhere).

Outputs:
  results/paper/table_mca_comparison.tex
  results/paper/mca_comparison.json      (all numbers, for downstream scripts)
  stdout summary

CPU-only (rollouts + numpy); does not touch the GPU.
"""
import json
import logging
from pathlib import Path

import numpy as np

from symmetric_stall.analysis.artifacts import policy_metrics
from symmetric_stall.aircraft.symmetric_stall import SymmetricStall
from paper_table_dp_vs_ppo import ALPHA_GRID_DEG, VNORM_GRID, simulate
from symmetric_stall.policy_iteration import PolicyIterationStall
from symmetric_stall.utils.utils import get_optimal_action

logger = logging.getLogger(__name__)

POLICIES = {
    "fixed": Path("results/SymmetricStall_policy.npz"),
    "mca": Path("results/SymmetricStall_policy_mca.npz"),
}
CANONICAL = (20.0, 0.95)          # (alpha0_deg, v0/Vs) — same as PPO_vs_PI
CANONICAL_H_REF = -9.09           # historical fixed-policy altitude loss (m)
OUT_DIR = Path("results/paper")


def full_throttle_fraction(pi) -> float:
    """Fraction of NON-terminal grid states whose argmax action commands
    delta_t = 1.0 (max throttle). Terminal cells keep policy index 0 from
    initialization, so they must be excluded."""
    actions = np.asarray(pi.action_space)
    thr_levels = np.unique(actions[:, 1])
    thr_of_idx = actions[np.asarray(pi.policy), 1]

    env = SymmetricStall()
    # Rebuild the grid axes from the persisted bounds/shape (states_space is
    # not stored in the npz).
    axes = [
        np.linspace(pi.bounds_low[d], pi.bounds_high[d], int(pi.grid_shape[d]))
        for d in range(len(pi.grid_shape))
    ]
    mesh = np.meshgrid(*axes, indexing="ij")
    states = np.vstack([m.ravel() for m in mesh]).astype(np.float32).T
    terminal_mask, _ = env.terminal(states)

    non_term = ~terminal_mask
    frac = float(np.mean(np.isclose(thr_of_idx[non_term], thr_levels[-1])))
    return frac


def rollout_grid(pi) -> dict:
    """Altitude loss over the standard IC grid, canonical stopping rule."""
    def action_fn(obs):
        return get_optimal_action(obs, pi)[0]

    out = {}
    for alpha0 in ALPHA_GRID_DEG:
        for v0 in VNORM_GRID:
            r = simulate(pi.env, action_fn, alpha0, v0)
            out[f"a{alpha0:.0f}_v{v0:.2f}"] = {
                "h": float(r["h"]), "t": float(r["t"]), "status": r["status"],
            }
            logger.info(f"    a0={alpha0:4.0f} V0={v0:.2f}: "
                        f"{r['h']:7.2f} m ({r['status']})")
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {}

    for name, path in POLICIES.items():
        logger.info(f"[*] {name}: loading {path}")
        pi = PolicyIterationStall.load(path, env=SymmetricStall())

        entry = {
            "path": str(path),
            "dt_scheme": {
                "use_mca_timestep": pi.config.use_mca_timestep,
                "dt_fixed": pi.config.dt_fixed,
                "dt_max": pi.config.dt_max,
                "n_micro": pi.config.n_micro,
            },
        }

        logger.info(f"[*] {name}: artifact metrics...")
        entry["artifacts"] = {
            k: (float(v) if isinstance(v, float) else int(v))
            for k, v in policy_metrics(pi).items()
        }

        logger.info(f"[*] {name}: full-throttle fraction...")
        entry["full_throttle_frac"] = full_throttle_fraction(pi)

        logger.info(f"[*] {name}: IC-grid rollouts...")
        entry["rollouts"] = rollout_grid(pi)

        report[name] = entry

    # ── Sanity: canonical IC of the fixed policy vs historical value ──────
    a0, v0 = CANONICAL
    h_canon = report["fixed"]["rollouts"][f"a{a0:.0f}_v{v0:.2f}"]["h"]
    drift = h_canon - CANONICAL_H_REF
    report["sanity"] = {
        "canonical_fixed_h": h_canon,
        "historical_ref": CANONICAL_H_REF,
        "drift_m": drift,
    }
    flag = "OK" if abs(drift) < 0.25 else "DRIFT — investigate"
    logger.info(f"[=] Sanity canonical fixed: {h_canon:.2f} m "
                f"(hist {CANONICAL_H_REF:.2f}, Δ={drift:+.2f} m) → {flag}")

    # ── Summary table (stdout + LaTeX) ────────────────────────────────────
    fx, mc = report["fixed"], report["mca"]
    hs_f = [v["h"] for v in fx["rollouts"].values()]
    hs_m = [v["h"] for v in mc["rollouts"].values()]
    mean_gain = float(np.mean(np.array(hs_m) - np.array(hs_f)))
    report["mean_mca_gain_m"] = mean_gain

    print("\n──── fixed vs MCA (4-DOF Riley, same grid/reward) ────")
    print(f"{'metric':<32}{'fixed':>12}{'mca':>12}")
    rows = [
        ("islands delta_e (interior)", fx["artifacts"]["island_de"], mc["artifacts"]["island_de"]),
        ("islands delta_t (interior)", fx["artifacts"]["island_thr"], mc["artifacts"]["island_thr"]),
        ("TV delta_e (deg/neighbor)", f"{fx['artifacts']['TV_de']:.4f}", f"{mc['artifacts']['TV_de']:.4f}"),
        ("TV delta_t (/neighbor)", f"{fx['artifacts']['TV_thr']:.4f}", f"{mc['artifacts']['TV_thr']:.4f}"),
        ("full-throttle fraction", f"{fx['full_throttle_frac']:.4f}", f"{mc['full_throttle_frac']:.4f}"),
        ("mean dh over IC grid (m)", f"{np.mean(hs_f):.3f}", f"{np.mean(hs_m):.3f}"),
    ]
    for label, a, b in rows:
        print(f"{label:<32}{a!s:>12}{b!s:>12}")
    print(f"\nmean MCA gain over IC grid: {mean_gain:+.3f} m "
          f"({'MCA better' if mean_gain > 0 else 'fixed better'})")
    print(f"canonical fixed sanity: {h_canon:.2f} m vs hist {CANONICAL_H_REF} → {flag}\n")

    # LaTeX per-IC table
    lines = [
        r"\begin{table}[hbt!]",
        r"    \centering",
        r"    \caption{Altitude loss $\Delta h$ (m) of the DP policy trained "
        r"with the fixed macro timestep ($\Delta t = 0.01$\,s) vs.\ the "
        r"MCA endogenous timestep, from initial conditions "
        r"$\gamma_0 = 0^\circ$, $q_0 = 0$.}",
        r"    \label{tab:mca_comparison}",
        r"    \begin{tabular}{cccc}",
        r"        \hline",
        r"        $\alpha_0$ (deg) & $V_0/V_s$ & Fixed $\Delta t$ & MCA \\",
        r"        \hline",
    ]
    for alpha0 in ALPHA_GRID_DEG:
        for v0 in VNORM_GRID:
            key = f"a{alpha0:.0f}_v{v0:.2f}"
            lines.append(
                f"        {alpha0:.0f} & {v0:.2f} & "
                f"{fx['rollouts'][key]['h']:.2f} & "
                f"{mc['rollouts'][key]['h']:.2f} \\\\")
    lines += [r"        \hline", r"    \end{tabular}", r"\end{table}"]
    (OUT_DIR / "table_mca_comparison.tex").write_text("\n".join(lines) + "\n")

    (OUT_DIR / "mca_comparison.json").write_text(json.dumps(report, indent=2))
    logger.info(f"[+] Wrote {OUT_DIR}/table_mca_comparison.tex and "
                f"mca_comparison.json")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
