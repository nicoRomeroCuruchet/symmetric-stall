"""
paper_cg_sweep_solve.py — CG-sensitivity sweep for the 4-DOF symmetric stall
recovery (Riley model): re-solves the exact DP problem for several CG
locations and records the recovery performance from the canonical IC.

Moment transfer about the Riley reference (0.25 c-bar):
    Cm_cg = Cm_ref + CL * (x_cg - 0.25)
injected into the CUDA kernel (DXCG_OVER_CHORD) and mirrored in the CPU
RK4 model so the trajectory rollouts are consistent with the solver.

Each run is warm-started from the nominal (0.25 c-bar) converged policy.
Outputs:
    results/cg_sweep/xcg_<value>.npz   (converged V*, policy per CG)
    results/cg_sweep/summary.json      (altitude loss per CG from canonical IC)
"""
import json
import logging
from pathlib import Path

import numpy as np

from symmetric_stall.train import setup_symmetric_stall_experiment
from symmetric_stall.policy_iteration import PolicyIterationStall
from symmetric_stall.utils.utils import get_optimal_action

logger = logging.getLogger(__name__)

XCG_VALUES = [0.20, 0.22, 0.25, 0.28, 0.30, 0.32]   # fraction of chord
XCG_REF = 0.25                                       # Riley Table I reference
BASELINE = Path("results/SymmetricStall_policy.npz")
OUT_DIR = Path("results/cg_sweep")

# Canonical IC (same as PPO_vs_PI.py / Fig. D)
GAMMA_0, V_NORM_0, ALPHA_0_DEG, Q_0 = 0.0, 0.95, 20.0, 0.0
MAX_TIME = 15.0
DIVE_THRESHOLD_DEG = -0.5


def rollout(pi, dxcg: float) -> dict:
    """Canonical-IC rollout under the Approach-A policy with the CPU model
    set to the same CG shift. Same stopping rule as paper_table_dp_vs_ppo."""
    env = pi.env
    env.airplane.DXCG_OVER_CHORD = dxcg
    v_stall = env.airplane.STALL_AIRSPEED
    step_dt = env.airplane.TIME_STEP

    obs, _ = env.specific_reset(GAMMA_0, V_NORM_0, np.deg2rad(ALPHA_0_DEG), Q_0)
    t, h = 0.0, 0.0
    has_dived = False
    while t < MAX_TIME:
        action, _, _ = get_optimal_action(obs, pi)
        obs, _, _, _, _ = env.step(action)
        h += obs[1] * v_stall * np.sin(obs[0]) * step_dt
        t += step_dt
        gamma_deg = np.rad2deg(obs[0])
        if gamma_deg < DIVE_THRESHOLD_DEG:
            has_dived = True
        if has_dived and gamma_deg >= 0.0:
            return {"h": float(h), "t": float(t), "status": "recovered"}
        if (obs[2] >= np.deg2rad(40) or obs[2] <= np.deg2rad(-40)
                or obs[0] <= -np.pi + 0.05):
            return {"h": float(h), "t": float(t), "status": "crash"}
    return {"h": float(h), "t": float(t), "status": "timeout"}


def main():
    import cupy as cp

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    baseline_policy = np.load(BASELINE)["policy"]
    summary = {}

    for xcg in XCG_VALUES:
        dxcg = xcg - XCG_REF
        out_path = OUT_DIR / f"xcg_{xcg:.2f}.npz"

        env, states, actions, config = setup_symmetric_stall_experiment()

        if out_path.exists():
            logger.info(f"[=] xcg={xcg:.2f}: cached, loading {out_path}")
            pi = PolicyIterationStall.load(out_path, env=env)
            pi.states_space = states
        elif xcg == XCG_REF:
            logger.info(f"[=] xcg={xcg:.2f}: nominal — reusing baseline npz")
            pi = PolicyIterationStall.load(BASELINE, env=env)
            pi.states_space = states
            pi.save(out_path)
        else:
            logger.info(f"[*] xcg={xcg:.2f} (dx/c={dxcg:+.2f}): solving PI...")
            config.dxcg_over_chord = dxcg
            pi = PolicyIterationStall(env, states, actions, config)
            # Warm start from the nominal converged policy
            pi.d_policy[:] = cp.asarray(baseline_policy, dtype=cp.int32)
            pi.run()
            pi.save(out_path)

        res = rollout(pi, dxcg)
        summary[f"{xcg:.2f}"] = res
        logger.info(f"[+] xcg={xcg:.2f}: dh={res['h']:.2f} m "
                    f"({res['status']}, t={res['t']:.2f} s)")

        (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
        del pi
        cp.get_default_memory_pool().free_all_blocks()

    logger.info(f"[+] Sweep complete: {OUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
