"""What does the DP itself predict at the inverted entry?

Separates two very different diagnoses. If V*(s0) is near the NLP's -500 m, the
policy KNOWS a recovery exists and the closed loop fails to realise it -- an
execution problem. If V*(s0) is itself around -1000 m or worse, the DP believes
the entry is (nearly) unrecoverable and the disagreement is about the problem,
not about the rollout.
"""
import logging, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from main import train_or_load_policy
from analysis.experiments import get_setup_for_level
from analysis.interpolation import get_optimal_action

logging.disable(logging.INFO)
env, states, actions, config = get_setup_for_level(4)()
config.use_mca_timestep = True; config.mca_include_mudot = True; config.dt_max = 0.10
pi, trained = train_or_load_policy(env, states, actions, config, "banked_glider_L4_mca", retrain=False)
assert not trained

V = np.asarray(pi.value_function)
print(f"V*: n={V.size}  min={V.min():.2f}  max={V.max():.2f}  media={V.mean():.2f}")
lo, hi = np.asarray(pi.bounds_low), np.asarray(pi.bounds_high)
shape = np.asarray(pi.grid_shape)

def vstar(gamma_deg, vn, mu_deg):
    s = np.array([np.deg2rad(gamma_deg), vn, np.deg2rad(mu_deg)], dtype=np.float32)
    idx = np.rint((s - lo) / ((hi - lo) / (shape - 1))).astype(int)
    idx = np.clip(idx, 0, shape - 1)
    flat = int(np.ravel_multi_index(tuple(idx), tuple(shape)))
    return V[flat]

print()
print(f"{'mu0':>6} {'V* (m)':>12} {'rollout dh':>12} {'NLP':>10}")
roll = {150:-1002.92, 120:-194.89, 90:-152.31, 60:-121.14, 30:-109.42}
nlp  = {150:-500.01, 120:-193.14, 90:-150.61, 60:-119.48, 30:-107.76}
for mu0 in (150,120,90,60,30):
    print(f"{mu0:6d} {vstar(-60.0,1.2,mu0):12.2f} {roll[mu0]:12.2f} {nlp[mu0]:10.2f}")

print("\nbarrido de mu0 alrededor de 150 (gamma=-60, v=1.2):")
for mu0 in (130,140,145,150,155,160,170,180,190):
    print(f"   mu0={mu0:4d}  V*={vstar(-60.0,1.2,mu0):10.2f}")
