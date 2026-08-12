"""DP optimum against CAA and FAA, canonical entry at 0.85 Vs, Riley thrust.

Reuses the scripted manoeuvres from procedures.py -- it does not rewrite them
-- so that the comparison is the same as paper 1's and the only things that
change are the thrust model and the grid.

    CAA   nose down and power AT THE SAME TIME (2 s ramp from t=0)
    FAA   nose down first, power only once out of the stall

Both arms are capped to the SAME pull authority the optimum reaches at this
entry, as run_maneuvers does: without that the comparison would mix WHEN power
is applied with HOW HARD the pilot pulls.

    THRUST_MODEL=riley PYTHONPATH=. python traj_caa_faa_085.py [V0]
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from symmetric_stall import train as main
from symmetric_stall import procedures as pp
from symmetric_stall.aircraft.symmetric_stall import SymmetricStall
from symmetric_stall.policy_iteration import PolicyIterationStall

V0 = float(sys.argv[1]) if len(sys.argv) > 1 else 0.85
ALPHA0 = 20.0

env = SymmetricStall()
pi = PolicyIterationStall.load(main.RESULTS_DIR / "SymmetricStall_policy.npz",
                               env=env)
_, states, _, _ = main.setup_symmetric_stall_experiment()
pi.states_space = states

# the optimum's pull authority at THIS entry, used to cap the arms
r_opt = pp.rollout(env, pi, pp.ctrl_optimal, ALPHA0, V0, record=True)
tope = float(np.min(r_opt["hist"]["de"]))
print("entry: alpha0 = %.0f deg, V0 = %.2f Vs, gamma0 = 0, q0 = 0" % (ALPHA0, V0))
print("optimum's deepest pull: %.2f deg (used as the cap for CAA and FAA)"
      % np.rad2deg(tope))

# Dos familias.
#
# The scripted ones are Gratton's as they stand: they push to +15 until
# crossing 14 deg and then hold alpha with a proportional law. Their elevator
# looks nothing like the optimum's, so their difference against the optimum
# mixes TWO decisions -- when the power comes in and how the nose is handled.
#
# The "optimal delta_e" ones take the elevator from the optimum itself and
# only change the timing of the power. They isolate the decision CAA and FAA
# actually disagree about, which is that one and not the pitching.
ARMS = [
    ("DP optimum", r_opt, "tab:blue", "-"),
    ("CAA de-opt", pp.rollout(env, pi, pp.make_power_delay(0.0, ramp=True),
                              ALPHA0, V0, record=True), "tab:green", "--"),
    ("FAA de-opt", pp.rollout(env, pi, pp.make_power_gated(ramp=True),
                              ALPHA0, V0, record=True), "tab:purple", "--"),
    ("CAA guionada", pp.rollout(env, pi, pp.make_maneuver("t0", "alpha_hold", tope),
                                ALPHA0, V0, record=True), "tab:orange", "-."),
    ("FAA guionada", pp.rollout(env, pi, pp.make_maneuver("unstall", "alpha_hold", tope),
                                ALPHA0, V0, record=True), "tab:red", "-."),
]

fig, axes = plt.subplots(2, 3, figsize=(13.5, 6.2))
print()
base = None
for name, r, color, ls in ARMS:
    h = r["hist"]
    t = np.asarray(h["t"])
    H = np.asarray(h["h"])
    if base is None:
        base = H.min()
    print("  %-13s h_min %+8.2f m   t %5.2f s   %-10s   gamma_min %+6.2f deg"
          "   %s" % (name, H.min(), r["t"], r["status"],
                     np.rad2deg(h["gamma"]).min(),
                     "" if name == "DP optimum"
                     else "x%.2f of the optimum" % (H.min() / base)))
    for ax, y, lab in ((axes[0, 0], np.rad2deg(h["gamma"]), r"$\gamma$ (deg)"),
                       (axes[0, 1], np.rad2deg(h["alpha"]), r"$\alpha$ (deg)"),
                       (axes[0, 2], np.asarray(h["v_norm"]), r"$V/V_s$ (--)"),
                       (axes[1, 0], np.rad2deg(h["de"]), r"$\delta_e$ (deg)"),
                       (axes[1, 1], np.asarray(h["dt_ctrl"]), r"$\delta_t$ (--)"),
                       (axes[1, 2], H, r"$\Delta h$ (m)")):
        ax.plot(t, y, ls, color=color, lw=1.7, label=name)
        ax.set_title(lab, fontsize=10)
        ax.set_xlabel("t (s)")
        ax.grid(alpha=0.3, ls=":")

axes[0, 1].axhline(14.0, color="grey", lw=0.8, ls="--")
axes[1, 2].axhline(0.0, color="grey", lw=0.8, ls="--")
axes[0, 0].legend(fontsize=9)
fig.suptitle(r"Entrada canonica a $%.2f\,V_s$ ($\alpha_0=%.0f^\circ$, "
             r"$\gamma_0=0$): DP optimum vs CAA vs FAA, Riley thrust"
             % (V0, ALPHA0), fontsize=11)
fig.tight_layout()
out = main.RESULTS_DIR / ("riley_caa_faa_v%03d.png" % round(V0 * 100))
fig.savefig(out, dpi=140)
print("\nfigura: %s" % out)
