"""Trayectoria 4-DOF desde la entrada canonica, con el empuje de Riley.

    alpha0 = 20 deg, V0 = 0.95 Vs, gamma0 = 0, q0 = 0

Es la misma entrada que el paper 1 reporta en -7.796 m / 7.67 s usando el
mapa lineal de empuje. Corrida con THRUST_MODEL=riley mide cuanto de ese
numero depende de una calibracion que el Apendice A del propio informe de
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
for nombre in ("plot_time_response", "plot_results", "plot_trajectory"):
    if hasattr(main, nombre):
        getattr(main, nombre)(hist, PREF); break

g = np.rad2deg(hist["gamma"]); a = np.rad2deg(hist["alpha"])
de = np.rad2deg(hist["de"]); d = np.asarray(hist["dt_ctrl"]); h = np.asarray(hist["h"])
print("\n=== canonica del paper 1, THRUST_MODEL=%s ===" % os.environ.get("THRUST_MODEL","paper1"))
print("  altura           %+9.3f m   (paper 1 publica -7.796)" % h[-1])
print("  duracion         %9.2f s   (paper 1 publica 7.67)" % hist["t"][-1])
print("  gamma min        %+9.2f deg (paper 1 publica -3.3)" % min(g))
print("  alpha max        %+9.2f deg" % max(a))
print("  elevador medio   %+9.2f deg (paper 1: control equivalente -5)" % de.mean())
print("  throttle medio   %9.3f      fraccion a full %.3f" % (d.mean(), np.mean(d>0.99)))
