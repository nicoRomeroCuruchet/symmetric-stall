"""Canonical entry at an arbitrary airspeed, with C_T instrumented.

    python traj_v_baja.py 0.50        (default 0.50 Vs)

At airspeeds well below the stall the dynamic pressure collapses and
C_T = T/(q_bar S) blows up, so _compute_ct's clip at 0.5 becomes active. When
that happens the state is out of domain on four sides at once, and the C_T
panel is precisely what gives it away; hence it is plotted alongside the states
instead of being left in the log.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from symmetric_stall import train as main
from symmetric_stall.policy_iteration import PolicyIterationStall
from symmetric_stall.aircraft.symmetric_stall import SymmetricStall

V0 = float(sys.argv[1]) if len(sys.argv) > 1 else 0.50
BRANCHES = [("riley", "SymmetricStall_policy.npz", "-", "tab:blue"),
         ("paper1", "SymmetricStall_paper1_baseline.npz", "--", "tab:red")]

fig, axes = plt.subplots(2, 3, figsize=(13, 6))
print("paper-1 canonical entry but at %.2f Vs  (grid: V in [0.90, 2.00] Vs)" % V0)
for mode, npz, ls, color in BRANCHES:
    os.environ["THRUST_MODEL"] = mode
    env = SymmetricStall()
    pi = PolicyIterationStall.load(main.RESULTS_DIR / npz, env=env)
    _, states, _, _ = main.setup_symmetric_stall_experiment()
    pi.states_space = states
    h = main.run_dp_simulation(pi, gamma_0_deg=0.0, v_norm_0=V0,
                               alpha_0_deg=20.0, q_0_deg=0.0)

    t = np.asarray(h["t"])
    vn = np.asarray(h["v_norm"])
    # C_T reconstructed along the trajectory, with the same code the plant
    # uses, to see when the 0.5 clip is biting
    ct = np.array([env.airplane._compute_ct(d, v * env.airplane.STALL_AIRSPEED)
                   for d, v in zip(h["dt_ctrl"], vn)])
    out_frac = 100.0 * np.mean(ct > 0.5)
    print("  %-7s h_min %+8.2f m  t_fin %5.2f s  V_min %.3f Vs  "
          "C_T max %.4f  out of table %.0f%% of the time"
          % (mode, np.min(h["h"]), t[-1], vn.min(), ct.max(), out_frac))

    for ax, y, lab in ((axes[0, 0], np.rad2deg(h["gamma"]), r"$\gamma$ (deg)"),
                       (axes[0, 1], np.rad2deg(h["alpha"]), r"$\alpha$ (deg)"),
                       (axes[0, 2], vn, r"$V/V_s$ (--)"),
                       (axes[1, 0], np.rad2deg(h["de"]), r"$\delta_e$ (deg)"),
                       (axes[1, 1], ct, r"$C_T$ (--)"),
                       (axes[1, 2], np.asarray(h["h"]), r"$\Delta h$ (m)")):
        ax.plot(t, y, ls, color=color, lw=1.7, label=mode)
        ax.set_title(lab, fontsize=10)
        ax.set_xlabel("t (s)")
        ax.grid(alpha=0.3, ls=":")

axes[0, 2].axhline(0.9, color="k", lw=0.9, ls=":")
axes[0, 2].annotate("grid edge", (0.02, 0.9), fontsize=7,
                    xycoords=("axes fraction", "data"), va="bottom")
axes[1, 1].axhline(0.5, color="k", lw=0.9, ls=":")
axes[1, 1].annotate("end of table (dCD_T acts above)", (0.02, 0.5), fontsize=7,
                    xycoords=("axes fraction", "data"), va="top")
axes[0, 1].axhline(14.0, color="grey", lw=0.7, ls="--")
axes[0, 0].legend(fontsize=9)
fig.suptitle(r"IC canonica a $%.2f\,V_s$ ($\alpha_0=20^\circ$, $\gamma_0=0$)  "
             "-- FUERA DE LA GRILLA" % V0, fontsize=11)
fig.tight_layout()
out = main.RESULTS_DIR / ("traj_v%03d.png" % round(V0 * 100))
fig.savefig(out, dpi=140)
print("figura: %s" % out)
