"""Ablation: how much of the procedure's penalty is the late TRIGGER and how
much is the power RAMP.

Gratton's procedure differs from the optimum in two ways, and the CAA/FAA table
measures them together:

  1. the trigger: it releases the dive when alpha < 14 (unstalled), whereas the
     optimum releases it at alpha ~ 16.7, i.e. it ANTICIPATES;
  2. the power: a 2 s ramp, against the optimum's step to full.

The two factors are crossed. Usage: ablacion.py <policy.npz> <V0>
"""
import logging
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from symmetric_stall import train as main  # noqa: E402

logging.disable(logging.INFO)

from symmetric_stall.procedures import (  # noqa: E402
    ALPHA_TARGET,
    DE_DOWN,
    GRATTON_RAMP_S,
    K_ALPHA,
    K_Q,
    ctrl_optimal,
    rollout,
)

from symmetric_stall.aircraft.symmetric_stall import SymmetricStall  # noqa: E402
from symmetric_stall.policy_iteration import PolicyIterationStall  # noqa: E402

POLICY, V0 = Path(sys.argv[1]), float(sys.argv[2])
A0 = float(sys.argv[3]) if len(sys.argv) > 3 else 20.0

env = SymmetricStall()
pi = PolicyIterationStall.load(POLICY, env=env)

r_opt = rollout(env, pi, ctrl_optimal, A0, V0, record=True)
CAP = float(np.min(r_opt["hist"]["de_cmd"]))   # command, not achieved: see procedures.cap_en


def pilot(alpha_trigger_deg, power):
    """power: 'step' (full from t=0) or 'ramp' (2 s from t=0)."""
    a_gate = np.deg2rad(alpha_trigger_deg)

    def ctrl(obs, t, opt, ctx):
        alpha, q = obs[2], obs[3]
        if "t_g" not in ctx and alpha < a_gate:
            ctx["t_g"] = t
        if "t_g" not in ctx:
            de = DE_DOWN
        else:
            de = float(np.clip(K_ALPHA * (alpha - ALPHA_TARGET) + K_Q * q,
                               CAP, DE_DOWN))
        thr = 1.0 if power == "step" else min(t / GRATTON_RAMP_S, 1.0)
        return (de, thr)
    return ctrl


print("IC: gamma=0, V=%.2f Vs, alpha=%.0f deg, q=0" % (V0, A0))
print("optimum's pull: %.2f deg (bounds every arm)\n" % np.rad2deg(CAP))

arms = [
    ("optimum (DP)",                 None),
    ("trigger 14 + ramp   (= CAA)",  (14.0, "ramp")),
    ("trigger 14 + step",            (14.0, "step")),
    ("trigger 17 + ramp",            (17.0, "ramp")),
    ("trigger 17 + step",            (17.0, "step")),
]

res = {}
print("%-30s %10s %8s %10s %11s" % ("arm", "dh", "t_rec", "alpha_min", "status"))
print("-" * 74)
for name, cfg in arms:
    r = r_opt if cfg is None else rollout(env, pi, pilot(*cfg), A0, V0, record=True)
    res[name] = r
    print("%-30s %+10.3f %7.2fs %9.2f %11s" % (
        name, r["h"], r["t"], np.rad2deg(np.min(r["hist"]["alpha"])), r["status"]))

base = res["optimum (DP)"]["h"]
caa = res["trigger 14 + ramp   (= CAA)"]["h"]
print("\ntotal penalty of the procedure: %+.3f m" % (caa - base))
print("\ndecomposition (starting from the CAA and fixing one factor at a time):")
d_pow = res["trigger 14 + step"]["h"] - caa
d_trg = res["trigger 17 + ramp"]["h"] - caa
print("  fix ONLY the power   (step instead of ramp):     %+8.3f m" % d_pow)
print("  fix ONLY the trigger (release at 17 not at 14):  %+8.3f m" % d_trg)
print("  fix BOTH:                                        %+8.3f m"
      % (res["trigger 17 + step"]["h"] - caa))
print("  (what is left to the optimum is the shape of the input, not these two)")


# ───────────────────────── figure ─────────────────────────
STYLE = {
    "optimum (DP)":                ("#0072B2", "-",  1.9),
    "trigger 14 + ramp   (= CAA)": ("#D55E00", "-",  1.4),
    "trigger 17 + ramp":           ("#D55E00", ":",  1.4),
    "trigger 14 + step":           ("#009E73", "-",  1.4),
    "trigger 17 + step":           ("#009E73", ":",  1.4),
}
PAN = [("gamma", r"$\gamma$ (deg)", np.rad2deg), ("v_norm", r"$V/V_s$", lambda x: x),
       ("alpha", r"$\alpha$ (deg)", np.rad2deg), ("q", r"$q$ (deg/s)", np.rad2deg),
       ("de", r"$\delta_e$ (deg)", np.rad2deg), ("dt_ctrl", r"$\delta_t$", lambda x: x),
       ("h", "altitude (m)", lambda x: x)]

fig, axes = plt.subplots(len(PAN), 1, figsize=(7.4, 12.2), sharex=True)
for ax, (k, et, cv) in zip(axes, PAN):
    for name, (col, ls, lw) in STYLE.items():
        h = res[name]["hist"]
        ax.plot(h["t"], cv(np.asarray(h[k])), lw=lw, ls=ls, color=col,
                label=name, zorder=3)
        if k == "h":
            y = cv(np.asarray(h[k])); i = int(np.argmin(y))
            ax.annotate("%.1f" % y[i], xy=(h["t"][i], y[i]), xytext=(6, 0),
                        textcoords="offset points", fontsize=7, color=col,
                        va="center")
    if k == "alpha":
        ax.axhline(14.0, color="0.5", lw=0.8, ls="--", zorder=1)
        ax.annotate(r"$\alpha_s=14^\circ$", xy=(0.99, 14.0),
                    xycoords=("axes fraction", "data"), ha="right", va="bottom",
                    fontsize=7, color="0.35")
    if k == "dt_ctrl":
        ax.set_ylim(-0.02, 1.05)
        ax.annotate("2 s ramp vs step", xy=(0.99, 0.45),
                    xycoords=("axes fraction", "data"), ha="right",
                    fontsize=7.5, color="0.35")
    if k in ("gamma", "h"):
        ax.axhline(0.0, color="0.85", lw=0.6, zorder=0)
    ax.set_ylabel(et, fontsize=9); ax.grid(True, color="0.92", lw=0.5)
    ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"): ax.spines[sp].set_color("0.6")
    ax.tick_params(labelsize=8, color="0.6")
axes[-1].set_xlabel("time (s)", fontsize=9)
axes[0].legend(loc="lower left", fontsize=7.5, frameon=False, ncol=2,
               bbox_to_anchor=(0.0, 1.02))
fig.suptitle(r"Trigger vs power ablation — $V_0=%.2f\,V_s$, $\alpha_0=%.0f^\circ$"
             % (V0, A0), fontsize=10, y=0.988)
fig.tight_layout(rect=[0, 0, 1, 0.955])
out = main.RESULTS_DIR / ("ablacion_v%03d.png" % round(V0 * 100))
fig.savefig(out, dpi=200)
print("-> %s" % out)
