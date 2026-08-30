"""Does the DP trajectory at mu_0 = 150 deg leave the grid?

Every other entry of the validation agrees with the NLP to within 3.6 %. That
one reports -1002.92 m against the NLP's -500.01 m, and raising the NLP horizon
makes the problem infeasible instead of fixing it, so the suspicion moves to
the DP side: if the dive accelerates past the grid's v_norm ceiling, the policy
is being read outside the box it was solved on and the altitude it accumulates
is extrapolation, not a recovery.

Replays the exact rollout loop of validate_trajectories_with_casadi and reports
where each trajectory spends its time and whether it saturates any axis.
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from main import train_or_load_policy
from analysis.experiments import get_setup_for_level
from analysis.interpolation import get_optimal_action

logging.disable(logging.INFO)

env, states, actions, config = get_setup_for_level(4)()
config.use_mca_timestep = True
config.mca_include_mudot = True
config.dt_max = 0.10
pi, trained = train_or_load_policy(env, states, actions, config,
                                   "banked_glider_L4_mca", retrain=False)
assert not trained, "it retrained; abort"

lo, hi = np.asarray(pi.bounds_low), np.asarray(pi.bounds_high)
print("orden de estado: (gamma, v_norm, mu)")
print(f"  low : gamma {np.rad2deg(lo[0]):7.1f}  v_norm {lo[1]:6.3f}  mu {np.rad2deg(lo[2]):7.1f}")
print(f"  high: gamma {np.rad2deg(hi[0]):7.1f}  v_norm {hi[1]:6.3f}  mu {np.rad2deg(hi[2]):7.1f}")
print()

dt = config.dt_fixed
env.airplane.TIME_STEP = dt
v_stall = env.airplane.STALL_AIRSPEED

print(f"{'mu0':>5} {'pasos':>6} {'T(s)':>6} {'dh(m)':>10} {'v_max':>7} {'v_fin':>7} "
      f"{'pasos en el techo de v':>24}")
for mu0 in (150.0, 120.0, 90.0, 60.0, 30.0):
    gamma, v_norm, mu = np.deg2rad(-60.0), 1.2, np.deg2rad(mu0)
    h = 0.0
    n = 0
    vmax, v_at_ceiling = v_norm, 0
    while gamma < 0.0 and n < 2000:
        sv = np.array([gamma, v_norm, mu], dtype=np.float32)
        action, _ = get_optimal_action(sv, pi)
        v_true = v_norm * v_stall
        h += v_true * np.sin(gamma) * dt
        env.state = np.atleast_2d(sv)
        nxt, _, _, _, _ = env.step(action)
        gamma, v_norm, mu = nxt.flatten()[:3]
        vmax = max(vmax, v_norm)
        if v_norm >= hi[1] - 1e-6:
            v_at_ceiling += 1
        n += 1
    print(f"{mu0:5.0f} {n:6d} {n*dt:6.1f} {h:10.2f} {vmax:7.3f} {v_norm:7.3f} "
          f"{v_at_ceiling:24d}")
