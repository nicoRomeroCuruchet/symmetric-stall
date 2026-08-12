"""4-DOF trajectory from the canonical entry, with Riley's thrust.

    alpha0 = 20 deg, V0 = 0.95 Vs, gamma0 = 0, q0 = 0

This is the same entry paper 1 reports at -7.796 m / 7.67 s using the linear
thrust map. Run with THRUST_MODEL=riley it measures how much of that number
depends on a calibration that Appendix A of Riley's own report
Riley vuelve innecesaria.

Usage:  THRUST_MODEL=riley PYTHONPATH=. python plot_ic_canonica.py [prefijo]
"""
import os, sys
import numpy as np
from symmetric_stall import train as main
from symmetric_stall.policy_iteration import PolicyIterationStall
from symmetric_stall.aircraft.symmetric_stall import SymmetricStall

PREF = sys.argv[1] if len(sys.argv) > 1 else "riley_thrust_canonica"

env = SymmetricStall()
pi = PolicyIterationStall.load(main.RESULTS_DIR / "SymmetricStall_policy.npz", env=env)
_, states, _, _ = main.setup_symmetric_stall_experiment()
pi.states_space = states

hist = main.run_dp_simulation(pi, gamma_0_deg=0.0, v_norm_0=0.95,
                              alpha_0_deg=20.0, q_0_deg=0.0)
for name in ("plot_time_response", "plot_results", "plot_trajectory"):
    if hasattr(main, name):
        getattr(main, name)(hist, PREF); break

g = np.rad2deg(hist["gamma"]); a = np.rad2deg(hist["alpha"])
de = np.rad2deg(hist["de"]); d = np.asarray(hist["dt_ctrl"]); h = np.asarray(hist["h"])
print("\n=== paper-1 canonical, THRUST_MODEL=%s ===" % os.environ.get("THRUST_MODEL","paper1"))
print("  altitude         %+9.3f m   (paper 1 publishes -7.796)" % h[-1])
print("  duracion         %9.2f s   (paper 1 publica 7.67)" % hist["t"][-1])
print("  gamma min        %+9.2f deg (paper 1 publica -3.3)" % min(g))
print("  alpha max        %+9.2f deg" % max(a))
print("  mean elevator    %+9.2f deg (paper 1: equivalent control -5)" % de.mean())
print("  throttle medio   %9.3f      fraccion a full %.3f" % (d.mean(), np.mean(d>0.99)))
