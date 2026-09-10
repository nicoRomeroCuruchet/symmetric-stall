"""paper_fig_pilot_axes.py — the three pilot-error axes in one row.

Merges the former fig_procedures (power delay) and the two panels of
fig_pilot_sensitivity (switch delay, pull modes) into a single
three-panel figure, one panel per error axis, so the section's
common-scale comparison is literal. Reads procedures.json; no
rollouts.

Output: fig_pilot_axes.{png,pdf} in paths.out_dir() (+ manuscript
copy img/fig_pilot_axes.pdf if present).
"""
import json
import logging
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

CANONICAL_KEY = "a20_v0.85"

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "stix", "font.size": 12,
    "axes.labelsize": 14, "xtick.labelsize": 12, "ytick.labelsize": 12,
})


def main():
    from symmetric_stall import paths
    report = json.loads(
        (paths.out_dir() / "procedures.json").read_text())
    ref = report["optimal_canonical"]
    ck = CANONICAL_KEY

    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(13.4, 3.9))

    # (a) power delay: the optimal-elevator pilot, late on the
    # throttle. One curve: the study is the pilot's failure to follow
    # the optimal maneuver, so the throttle model stays the optimum's
    # own (slammed); the ramped variant and the doctrinal landmarks
    # (gated FAA logic, Gratton's power-delayed protocol) belong to
    # the maneuver comparison of the Time-Domain section, not here.
    e1 = report["e1_power_delay"]
    taus = sorted(float(k) for k in e1["instant"])
    h = [e1["instant"][f"{t:g}"][ck]["h"] for t in taus]
    ax_a.plot(taus, h, marker="o", ms=4.5, color="#2C4B9E", lw=2.0)
    ax_a.axhline(ref["h"], color="gray", lw=0.9, ls=":")
    ax_a.set_xlabel(r"Power application delay $\tau$ (s)")
    ax_a.set_ylabel(r"$\Delta h$ (m)")
    ax_a.set_title("(a) Late power", fontsize=13)

    # (b) switch delay at the canonical entry, with loss multipliers
    e3b = report["e3b_switch_delay"]
    taus = sorted(float(k) for k in e3b)
    h = [e3b[f"{t:g}"][ck]["h"] for t in taus]
    ax_b.plot(taus, h, marker="o", ms=4.5, color="#2C4B9E", lw=2.0,
              zorder=3)
    ax_b.axhline(ref["h"], color="gray", lw=0.9, ls=":")
    for t, dx, dy, ha in ((0.5, 6, 6, "left"), (1.0, 9, -3, "left")):
        ht = e3b[f"{t:g}"][ck]["h"]
        ax_b.annotate(f"$\\times${ht / ref['h']:.1f}", xy=(t, ht),
                      xytext=(dx, dy), textcoords="offset points",
                      ha=ha, va="center", fontsize=12, color="#2C4B9E")
    ax_b.margins(x=0.18)
    ax_b.set_xlabel(r"Pitch-up switch delay $\tau_s$ (s)")
    ax_b.set_title("(b) Late pull", fontsize=13)

    # (c) the pull axis: closed-loop cap vs open-loop held deflection
    e3c = report["e3c_partial_pull"]
    e3d = report["e3d_held_pull"]
    pulls_cap = sorted((-abs(float(k)) for k in e3c), reverse=True)
    h_cap = [e3c[f"{p:g}"][ck]["h"] for p in pulls_cap]
    ax_c.plot(pulls_cap, h_cap, marker="o", ms=4.5, color="#2C4B9E",
              lw=2.0, zorder=4,
              label="timid pull: never harder than $\\delta_e$, "
                    "flown on $\\alpha$")
    pulls_h = sorted((-float(k) for k in e3d), reverse=True)
    h_held = [e3d[f"{-p:g}"]["h"] for p in pulls_h]
    ax_c.plot(pulls_h, h_held, marker="s", ms=4.5, color="#D62728",
              lw=2.0, zorder=3,
              label="overdone pull: $\\delta_e$ held, no feedback")
    restall = [p for p in pulls_h
               if e3d[f"{-p:g}"]["alpha_max_deg"] > 14.5]
    ax_c.axvspan(min(restall), max(restall), color="0.35", alpha=0.10,
                 zorder=1)
    ax_c.annotate("secondary stall\n($\\alpha > \\alpha_s$)",
                  xy=(0.78, 0.45), xycoords="axes fraction",
                  fontsize=10.5, color="#D62728", ha="center")
    ax_c.annotate("insufficient pull\n(no re-stall)",
                  xy=(pulls_h[0], h_held[0]),
                  xytext=(-2, -14), textcoords="offset points",
                  fontsize=10, color="#D62728", ha="left", va="top")
    ax_c.axhline(ref["h"], color="gray", lw=0.9, ls=":")
    ax_c.axvline(-5.0, color="0.45", lw=0.9, ls="--")
    ax_c.annotate("$\\bar{\\delta}_e \\approx -5^\\circ$",
                  xy=(-5.0, 0.45), xycoords=("data", "axes fraction"),
                  xytext=(4, 0), textcoords="offset points",
                  fontsize=10.5, color="0.35", ha="left")
    ax_c.set_xlim(0.0, -26.0)
    ax_c.set_xlabel(r"Pull-up deflection $\delta_{e}$ (deg)")
    ax_c.set_title("(c) The pull itself", fontsize=13)
    ax_c.legend(loc="lower left", fontsize=10, framealpha=0.95)

    for ax in (ax_a, ax_b, ax_c):
        ax.grid(alpha=0.3)
        ax.set_axisbelow(True)

    fig.tight_layout()
    out = paths.out_dir()
    out_paper = Path("stall-paper/img")
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig_pilot_axes.{ext}", dpi=300,
                    bbox_inches="tight")
        if out_paper.is_dir():
            fig.savefig(out_paper / f"fig_pilot_axes.{ext}", dpi=300,
                        bbox_inches="tight")
    plt.close(fig)
    logger.info("[+] fig_pilot_axes.{png,pdf} written")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
