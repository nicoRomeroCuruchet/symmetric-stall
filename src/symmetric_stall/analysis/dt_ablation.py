"""
dt_ablation.py
--------------
A/B study of the timestep scheme for the 4-DOF Symmetric Stall Policy
Iteration, ported from the 3DOF branch (commit 015fcf9).

Trains the SAME grid (main.py's setup_symmetric_stall_experiment) under
several schemes and compares the discretization-artifact fingerprint of the
resulting policies:

    fixed_0.01 : original hardcoded behavior (macro dt = 0.01 s, 10×1ms RK4)
    fixed_0.02 : coarser fixed macro step
    mca_state  : state-driven MCA interval dt_h = 1/(Σ|f_i|/h_i), clamped to
                 dt_max, evaluated at the current policy's action and shared
                 across the improvement argmax

The per-action variant (config.mca_per_action=True) is implemented in the
kernel as the biased control arm but intentionally NOT scheduled here —
3DOF showed it fragments the policy (μ̇ islands 5 → 174).

NOTE (3DOF precedent): mca_state failed to converge at fine grids under
gamma=1 (policy oscillation). This grid is 5.65M states — watch `iters`.

Run (needs CuPy / GPU; each scheme is a full training on 5.65M × 451):
    python -m analysis.dt_ablation
"""

import argparse
import logging
import time
from dataclasses import replace
from pathlib import Path

from symmetric_stall.analysis.artifacts import policy_metrics
from symmetric_stall.policy_iteration import PolicyIterationStall
from symmetric_stall.train import setup_symmetric_stall_experiment

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

OUT_DIR = Path("results/ablation/stall")

# PI-step cap for ablation runs: bounds the wall-clock of a non-converging
# scheme (3DOF precedent: mca_state oscillated forever at fine grids). A
# scheme that hits the cap is reported as non-converged — that IS a result.
ABLATION_N_STEPS = 150

# (name, overrides applied to the base config)
SCHEMES = [
    ("fixed_0.01", dict(use_mca_timestep=False, dt_fixed=0.01)),
    ("fixed_0.02", dict(use_mca_timestep=False, dt_fixed=0.02)),
    ("mca_state",  dict(use_mca_timestep=True,  dt_max=0.05)),
]


def run_ablation(only: list[str] | None = None) -> list[dict]:
    """Train each (selected) scheme on the symmetric-stall grid."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    env, states, actions, base_config = setup_symmetric_stall_experiment()
    base_config = replace(base_config, n_steps=ABLATION_N_STEPS)
    rows = []

    schemes = [s for s in SCHEMES if only is None or s[0] in only]
    for name, overrides in schemes:
        logger.info("=" * 60)
        logger.info(f"  Scheme: {name}  ({overrides})")
        logger.info("=" * 60)

        config = replace(base_config, **overrides)
        pi = PolicyIterationStall(env, states, actions, config)

        t0 = time.perf_counter()
        pi.run(save_path=OUT_DIR / f"{name}.npz")
        wall = time.perf_counter() - t0

        row = {
            "scheme": name,
            "iters": getattr(pi, "n_policy_steps", -1),
            "residual": getattr(pi, "final_residual", float("nan")),
            "wall_s": wall,
        }
        row.update(policy_metrics(pi))
        rows.append(row)
        logger.info(f"  {name}: {row['iters']} PI steps in {wall:.0f}s")

    return rows


def format_table(rows: list[dict]) -> str:
    """Compact comparison table across schemes."""
    cols = [
        ("scheme", "{:<12s}"),
        ("iters", "{:>6}"),
        ("residual", "{:>11.3e}"),
        ("wall_s", "{:>8.0f}"),
        ("TV_de", "{:>9.4f}"),
        ("TV_thr", "{:>8.4f}"),
        ("island_de", "{:>10}"),
        ("island_thr", "{:>11}"),
    ]
    header = "  ".join(f"{name:>{max(len(name), 6)}s}" for name, _ in cols)
    lines = [header, "-" * len(header)]
    for r in rows:
        cells = []
        for name, fmt in cols:
            val = r.get(name, "")
            try:
                cells.append(fmt.format(val))
            except (ValueError, TypeError):
                cells.append(str(val))
        lines.append("  ".join(cells))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="4DOF timestep-scheme ablation")
    parser.add_argument(
        "--schemes", nargs="+", default=None,
        choices=[name for name, _ in SCHEMES],
        help="Subset of schemes to run (default: all). Useful to run the "
             "baseline first and gauge wall-clock before committing to the rest.",
    )
    args = parser.parse_args()

    rows = run_ablation(only=args.schemes)
    table = format_table(rows)
    print("\n" + table + "\n")

    # Append (don't clobber) so schemes run in separate invocations accumulate.
    report = OUT_DIR / "metrics.txt"
    with open(report, "a", encoding="utf-8") as f:
        f.write(table + "\n\n")
        for r in rows:
            f.write(f"{r['scheme']}: {r}\n")
    logger.info(f"Metrics appended to {report}")


if __name__ == "__main__":
    main()
