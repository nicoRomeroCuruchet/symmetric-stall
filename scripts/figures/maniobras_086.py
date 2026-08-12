"""DP optimum against Gratton's CAA and FAA manoeuvres, at the requested IC.

Reuses the pieces of procedures.py without touching it: same rollout, same
regla de parada (RecoveryMonitor), mismos controladores.

  CAA : power from t = 0                (power_start = "t0")
  FAA : power on unstalling             (power_start = "unstall")
  fase 1: de = +15 deg hasta alpha < 14 deg
  fase 2: alpha_hold  -> proporcional apuntando a alpha = 13 deg
          full_pull   -> de = -25 deg a lazo abierto

The alpha_hold arms are capped to the SAME pull authority the optimum reaches
at that entry, so that the comparison isolates WHEN power is applied and does
not get confused with HOW HARD the pilot pulls.

Uso: maniobras_086.py <policy.npz> <V0> [alpha0_deg]
"""
import sys
import logging
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.disable(logging.INFO)

from symmetric_stall import train as main
from symmetric_stall.policy_iteration import PolicyIterationStall
from symmetric_stall.aircraft.symmetric_stall import SymmetricStall
from symmetric_stall.procedures import rollout, ctrl_optimal, make_maneuver

POLICY = Path(sys.argv[1])
V0 = float(sys.argv[2])
ALPHA0 = float(sys.argv[3]) if len(sys.argv) > 3 else 20.0

env = SymmetricStall()
pi = PolicyIterationStall.load(POLICY, env=env)
print("policy: %s" % POLICY.name)
print("IC: gamma=0, V=%.2f Vs, alpha=%.0f deg, q=0\n" % (V0, ALPHA0))

# the optimum first: its deepest pull caps the alpha_hold arms
r_opt = rollout(env, pi, ctrl_optimal, ALPHA0, V0, record=True)
cap = float(np.min(r_opt["hist"]["de"]))
print("optimum's deepest pull: %.2f deg  (caps the alpha_hold arms)\n"
      % np.rad2deg(cap))

brazos = {
    "optimo (DP)":     None,
    "CAA alpha-hold":  ("t0", "alpha_hold", cap),
    "FAA alpha-hold":  ("unstall", "alpha_hold", cap),
    "CAA full-pull":   ("t0", "full", None),
    "FAA full-pull":   ("unstall", "full", None),
}

res = {}
print("%-16s %10s %8s %11s %12s" % ("manoeuvre", "dh", "t_rec", "alpha_max", "status"))
print("-" * 62)
for nom, cfg in brazos.items():
    if cfg is None:
        r = r_opt
    else:
        r = rollout(env, pi, make_maneuver(*cfg), ALPHA0, V0, record=True)
    res[nom] = r
    print("%-16s %+10.3f %7.2fs %10.2f %12s" % (
        nom, r["h"], r["t"], np.rad2deg(np.max(r["hist"]["alpha"])), r["status"]))

base = res["optimo (DP)"]["h"]
print("\npenalty against the optimum:")
for nom in list(brazos)[1:]:
    print("  %-16s %+8.3f m  (%+6.1f %%)" % (
        nom, res[nom]["h"] - base, 100 * (res[nom]["h"] - base) / abs(base)))

# ---- figure: optimum vs CAA vs FAA (the alpha-hold arms) ----
MOSTRAR = ["optimo (DP)", "CAA alpha-hold", "FAA alpha-hold"]
COLOR = {"optimo (DP)": "#0072B2", "CAA alpha-hold": "#D55E00",
         "FAA alpha-hold": "#009E73"}
PAN = [("gamma", r"$\gamma$ (deg)", np.rad2deg), ("v_norm", r"$V/V_s$", lambda x: x),
       ("alpha", r"$\alpha$ (deg)", np.rad2deg), ("q", r"$q$ (deg/s)", np.rad2deg),
       ("de", r"$\delta_e$ (deg)", np.rad2deg), ("dt_ctrl", r"$\delta_t$", lambda x: x),
       ("h", "altura (m)", lambda x: x)]

fig, axes = plt.subplots(len(PAN), 1, figsize=(7.2, 12.0), sharex=True)
for ax, (k, et, cv) in zip(axes, PAN):
    for nom in MOSTRAR:
        h = res[nom]["hist"]
        ax.plot(h["t"], cv(np.asarray(h[k])), lw=1.4, color=COLOR[nom],
                label=nom, zorder=3)
        if k == "h":
            y = cv(np.asarray(h[k])); i = int(np.argmin(y))
            ax.plot(h["t"][i], y[i], "o", ms=4.5, color=COLOR[nom], mec="white",
                    mew=1.0, zorder=4)
            ax.annotate("%.2f" % y[i], xy=(h["t"][i], y[i]), xytext=(6, 0),
                        textcoords="offset points", fontsize=7.5,
                        color=COLOR[nom], va="center")
    if k == "alpha":
        ax.axhline(14.0, color="0.5", lw=0.8, ls="--", zorder=1)
        ax.annotate(r"$\alpha_s=14^\circ$", xy=(0.99, 14.0),
                    xycoords=("axes fraction", "data"), ha="right", va="bottom",
                    fontsize=7, color="0.35")
    if k == "dt_ctrl":
        ax.set_ylim(-0.02, 1.05)
    if k in ("gamma", "h"):
        ax.axhline(0.0, color="0.85", lw=0.6, zorder=0)
    ax.set_ylabel(et, fontsize=9); ax.grid(True, color="0.92", lw=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color("0.6")
    ax.tick_params(labelsize=8, color="0.6")
axes[-1].set_xlabel("tiempo (s)", fontsize=9)
axes[0].legend(loc="lower left", fontsize=8, frameon=False, ncol=3,
               bbox_to_anchor=(0.0, 1.02))
fig.suptitle(r"DP optimo vs CAA vs FAA — $V_0=%.2f\,V_s$, $\alpha_0=%.0f^\circ$"
             % (V0, ALPHA0), fontsize=10, y=0.985)
fig.tight_layout(rect=[0, 0, 1, 0.965])
out = main.RESULTS_DIR / ("maniobras_v%03d.png" % round(V0 * 100))
fig.savefig(out, dpi=200)
print("\n-> %s" % out)
