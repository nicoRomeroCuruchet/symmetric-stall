"""Cut-off criterion: the -2 deg has_dived threshold against gamma < 0."""
import sys, logging
from pathlib import Path
import numpy as np
logging.disable(logging.INFO)
from symmetric_stall import train as main
from symmetric_stall.policy_iteration import PolicyIterationStall
from symmetric_stall.aircraft.symmetric_stall import SymmetricStall
from symmetric_stall.utils.utils import get_optimal_action

pi = PolicyIterationStall.load(Path(sys.argv[1]), env=SymmetricStall())
env = pi.env; vs = env.airplane.STALL_AIRSPEED; dt = env.airplane.TIME_STEP

def rodar(v0, umbral_deg):
    obs, _ = env.specific_reset(flight_path_angle=0.0, airspeed_norm=v0,
                                alpha=np.deg2rad(20.0), pitch_rate=0.0)
    t, h, dived, idx = 0.0, 0.0, False, None
    hs = [0.0]
    for _ in range(1500):
        a, _, idx = get_optimal_action(obs, pi, idx)
        obs, _, term, _, _ = env.step(a)
        h += obs[1]*vs*np.sin(obs[0])*dt; t += dt; hs.append(h)
        if obs[0] < np.deg2rad(umbral_deg): dived = True
        if dived and obs[0] >= 0.0:
            return t, h, min(hs), True
        if term and (obs[2] >= np.deg2rad(40) or obs[0] <= -np.pi+0.05):
            return t, h, min(hs), False
    return t, h, min(hs), None

print("%6s | %-26s | %-26s" % ("", "umbral -2 deg (actual)", "gamma < 0 (propuesto)"))
print("%6s | %7s %9s %7s | %7s %9s %7s" % ("V0","dur","dh","h_min","dur","dh","h_min"))
print("-"*70)
for v0 in (0.80,0.85,0.88,0.90,0.91,0.92,0.93,0.94,0.95):
    a = rodar(v0, -2.0); b = rodar(v0, 0.0)
    f = lambda r: "%7.2f %+9.3f %7.3f" % (r[0], r[1], r[2]) if r[3] else \
                  "%7s %9s %7.3f" % ("(15s)", "--", r[2])
    print("%6.2f | %s | %s" % (v0, f(a), f(b)))
