"""q = gamma_dot + alpha_dot: does it hold along the trajectory?"""
import sys, logging
from pathlib import Path
import numpy as np
logging.disable(logging.INFO)
from symmetric_stall import train as main
from symmetric_stall.policy_iteration import PolicyIterationStall
from symmetric_stall.aircraft.symmetric_stall import SymmetricStall

pi = PolicyIterationStall.load(Path(sys.argv[1]), env=SymmetricStall())
for v0 in (0.85, 0.90):
    h = main.run_dp_simulation(pi, 0.0, v0, 20.0, 0.0)
    t = np.asarray(h["t"]); g = np.asarray(h["gamma"]); a = np.asarray(h["alpha"])
    q = np.asarray(h["q"])
    gd = np.gradient(g, t); ad = np.gradient(a, t)
    print("\n=== V0 = %.2f Vs ===" % v0)
    print("%6s %9s %9s %10s %10s %10s %9s" % (
        "t", "gamma", "alpha", "q", "gamma_dot", "alpha_dot", "gd+ad"))
    for i in list(range(0, len(t), max(1, len(t)//8))) + [len(t)-1]:
        print("%6.2f %+9.2f %+9.2f %+10.3f %+10.3f %+10.3f %+9.3f" % (
            t[i], np.rad2deg(g[i]), np.rad2deg(a[i]), np.rad2deg(q[i]),
            np.rad2deg(gd[i]), np.rad2deg(ad[i]), np.rad2deg(gd[i]+ad[i])))
    err = np.max(np.abs(q - (gd + ad))[5:-5])
    print("  worst |q - (gamma_dot + alpha_dot)| in the interior: %.3e deg/s"
          % np.rad2deg(err))
    print("  q FINAL = %+.3f deg/s   gamma_dot FINAL = %+.3f deg/s"
          % (np.rad2deg(q[-1]), np.rad2deg(gd[-1])))
