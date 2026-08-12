"""Canonical trajectory from several entry airspeeds.

    python traj_canonica_v.py 0.9 0.5

Same entry as paper 1 in everything else -- gamma0 = 0, alpha0 = 20 deg,
q0 = 0 -- varying only V0. A single policy is plotted, the Riley-thrust one:
the published paper-1 policy was trained on the old grid (41 bins from 0.9 Vs)
and cannot be evaluated on this one without reindexing it.

The C_T panel is not decorative: below 0.785 Vs the coefficient exceeds 0.5,
the tables read frozen at their endpoint and dCD_T absorbs the excess. It shows
which part of each curve leans on that.
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from symmetric_stall import train as main
from symmetric_stall.policy_iteration import PolicyIterationStall
from symmetric_stall.aircraft.symmetric_stall import SymmetricStall

V0S = [float(x) for x in sys.argv[1:]] or [0.9, 0.5]
COLORES = ["tab:blue", "tab:red", "tab:green", "tab:purple"]

env = SymmetricStall()
pi = PolicyIterationStall.load(main.RESULTS_DIR / "SymmetricStall_policy.npz",
                               env=env)
_, states, _, _ = main.setup_symmetric_stall_experiment()
pi.states_space = states
v_bins = np.unique(states[:, 1])
print("V grid: %d bins from %.3f to %.3f Vs" % (len(v_bins), v_bins.min(),
                                                  v_bins.max()))

fig, axes = plt.subplots(2, 3, figsize=(13.5, 6.2))
for k, v0 in enumerate(V0S):
    color = COLORES[k % len(COLORES)]
    h = main.run_dp_simulation(pi, gamma_0_deg=0.0, v_norm_0=v0,
                               alpha_0_deg=20.0, q_0_deg=0.0)
    t = np.asarray(h["t"])
    vn = np.asarray(h["v_norm"])
    H = np.asarray(h["h"])
    ct = np.array([env.airplane._compute_ct(d, v * env.airplane.STALL_AIRSPEED)
                   for d, v in zip(h["dt_ctrl"], vn)])
    recupera = t[-1] < 14.9
    print("  V0 = %.2f Vs :  h_min %+8.2f m   h_fin %+8.2f m   t %5.2f s  %s"
          "   gamma_min %+6.2f deg   C_T max %.3f   out of table %3.0f%%"
          % (v0, H.min(), H[-1], t[-1],
             "recupera" if recupera else "NO PICA ",
             np.rad2deg(h["gamma"]).min(), ct.max(), 100 * np.mean(ct > 0.5)))

    for ax, y, lab in ((axes[0, 0], np.rad2deg(h["gamma"]), r"$\gamma$ (deg)"),
                       (axes[0, 1], np.rad2deg(h["alpha"]), r"$\alpha$ (deg)"),
                       (axes[0, 2], vn, r"$V/V_s$ (--)"),
                       (axes[1, 0], np.rad2deg(h["de"]), r"$\delta_e$ (deg)"),
                       (axes[1, 1], ct, r"$C_T$ (--)"),
                       (axes[1, 2], H, r"$\Delta h$ (m)")):
        ax.plot(t, y, "-", color=color, lw=1.7,
                label=r"$V_0=%.2f\,V_s$" % v0)
        ax.set_title(lab, fontsize=10)
        ax.set_xlabel("t (s)")
        ax.grid(alpha=0.3, ls=":")

axes[0, 1].axhline(14.0, color="grey", lw=0.8, ls="--")
axes[0, 2].axhline(0.785, color="k", lw=0.8, ls=":")
axes[0, 2].annotate("C_T exceeds 0.5 below this", (0.02, 0.785), fontsize=7,
                    xycoords=("axes fraction", "data"), va="top")
axes[1, 1].axhline(0.5, color="k", lw=0.8, ls=":")
axes[1, 1].annotate("end of table", (0.02, 0.5), fontsize=7,
                    xycoords=("axes fraction", "data"), va="bottom")
axes[1, 2].axhline(0.0, color="grey", lw=0.8, ls="--")
axes[0, 0].legend(fontsize=9)
fig.suptitle(r"Entrada canonica ($\gamma_0=0$, $\alpha_0=20^\circ$, $q_0=0$), "
             "Riley thrust, grid 0.4-2.0 $V_s$", fontsize=11)
fig.tight_layout()
out = main.RESULTS_DIR / "riley_v040_canonica_v.png"
fig.savefig(out, dpi=140)
print("figura: %s" % out)
