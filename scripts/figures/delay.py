"""How long the procedure takes to release the dive, against the optimum."""
import sys, logging
from pathlib import Path
import numpy as np
logging.disable(logging.INFO)
from symmetric_stall.policy_iteration import PolicyIterationStall
from symmetric_stall.aircraft.symmetric_stall import SymmetricStall
from symmetric_stall.procedures import rollout, ctrl_optimal, make_maneuver

pi = PolicyIterationStall.load(Path(sys.argv[1]), env=SymmetricStall())
V0, A0 = float(sys.argv[2]), 20.0
env = pi.env

r_opt = rollout(env, pi, ctrl_optimal, A0, V0, record=True)
cap = float(np.min(r_opt["hist"]["de"]))
r_caa = rollout(env, pi, make_maneuver("t0", "alpha_hold", cap), A0, V0, record=True)

def hitos(r, nom):
    t = np.asarray(r["hist"]["t"]); de = np.rad2deg(np.asarray(r["hist"]["de"]))
    al = np.rad2deg(np.asarray(r["hist"]["alpha"]))
    i_soltar = int(np.argmax(de < 0)) if np.any(de < 0) else -1
    i_a14 = int(np.argmax(al < 14.0)) if np.any(al < 14.0) else -1
    print("%-14s releases the dive (de<0) at t=%.3f s with alpha=%.2f deg   |   "
          "alpha cruza 14 deg en t=%.3f s   |   alpha_min=%.2f deg"
          % (nom, t[i_soltar], al[i_soltar], t[i_a14], al.min()))
    return t[i_soltar], t[i_a14], al.min()

print("IC: V0=%.2f Vs, alpha0=20 deg\n" % V0)
to, ta_o, amin_o = hitos(r_opt, "optimo")
tc, ta_c, amin_c = hitos(r_caa, "CAA")
print("\n  the optimum releases the dive %.3f s BEFORE the procedure" % (tc - to))
print("  and does so at alpha = %.2f deg, i.e. STILL STALLED (>14)"
      % np.rad2deg(np.asarray(r_opt["hist"]["alpha"]))[int(np.argmax(np.rad2deg(np.asarray(r_opt["hist"]["de"])) < 0))])
print("  procedure overshoot: alpha_min %.2f vs %.2f deg  (%.2f deg more)"
      % (amin_c, amin_o, amin_o - amin_c))
