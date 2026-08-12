"""
diag_eval_6dof.py — Phase-1 diagnostic for 6-DOF policy-evaluation convergence.

Builds a COARSE banked-spin grid (the terminal-reachability problem is structural,
not resolution-dependent), then runs a few PI iterations manually with per-sweep
Δ logging enabled. Goal: distinguish divergence (Δ grows / V → -∞ on a recurrent
set) from slow convergence (Δ decreases monotonically but stays above θ).
"""
import logging

import cupy as cp
import numpy as np

from main import setup_banked_spin_experiment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("diag")

# Full grid (the user's actual case): confirm bounded-but-slow vs divergence.
env, states, actions, config = setup_banked_spin_experiment()

config.log = True               # enable the Δ / V-range logging we just added
config.maximum_iterations = 1500  # cap per eval so we see the Δ trajectory
config.theta = 1e-12             # force the eval to run to the cap

from PolicyIterationBankedSpin import PolicyIterationBankedSpin

pi = PolicyIterationBankedSpin(env, states, actions, config)

# How many grid nodes are terminal at all?
term_mask, _ = env.terminal(states)
logger.info(f"Terminal nodes on grid: {int(term_mask.sum()):,} / {len(states):,} "
            f"({100.0 * term_mask.mean():.2f}%)")

for it in range(1):
    logger.info(f"================ PI iteration {it + 1} ================")
    pi.policy_evaluation()
    # Inspect where the value is most negative (suspected spin / recurrent region)
    v = pi.d_value_function.get()
    worst = int(np.argmin(v))
    s = states[worst]
    logger.info(
        f"  after eval {it + 1}: V min={v.min():.3e}, max={v.max():.3e}, "
        f"mean={v.mean():.3e}")
    logger.info(
        f"  most-negative state: γ={np.rad2deg(s[0]):.1f}° V/Vs={s[1]:.2f} "
        f"α={np.rad2deg(s[2]):.1f}° μ={np.rad2deg(s[3]):.1f}° "
        f"p={s[4]:.2f} q={np.rad2deg(s[5]):.1f}°/s")
    pi.policy_improvement()

cp.get_default_memory_pool().free_all_blocks()
logger.info("Diagnostic complete.")
