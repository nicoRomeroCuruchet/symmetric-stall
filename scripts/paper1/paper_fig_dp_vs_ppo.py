"""
paper_fig_dp_vs_ppo.py — Publication-quality Fig. D: DP (Policy Iteration) vs
PPO time-domain comparison for the 4-DOF symmetric stall recovery (Riley model).

Reuses the simulation machinery from PPO_vs_PI.py (same canonical IC:
γ=0°, V/Vs=0.95, α=20°, q=0). Output: results/paper/fig_dp_vs_ppo.{pdf,png}
"""
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from PPO_vs_PI import load_pi, run_simulation
from stable_baselines3 import PPO

from symmetric_stall.aircraft.symmetric_stall import SymmetricStall
from symmetric_stall.utils.utils import get_optimal_action

logger = logging.getLogger(__name__)

C_PI = "#2C4B9E"
C_PPO = "#2CA02C"
ALPHA_STALL_DEG = 14.0

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "-",
    "lines.linewidth": 1.6,
})


def main():
    pi = load_pi(Path("results/SymmetricStall_policy.npz"))
    ppo_model = PPO.load(Path("policy_symmetric_stall/models/best_model.zip"),
                         device="cpu")
    ppo_env = SymmetricStall()

    pi_data = run_simulation(
        pi.env, lambda obs: get_optimal_action(obs, pi)[0])
    ppo_data = run_simulation(
        ppo_env, lambda obs: ppo_model.predict(obs, deterministic=True)[0])

    dh_pi, dh_ppo = pi_data["h"][-1], ppo_data["h"][-1]
    gap = 100.0 * (dh_ppo - dh_pi) / dh_ppo
    logger.info(f"Altitude loss  PI: {dh_pi:.2f} m   PPO: {dh_ppo:.2f} m   "
                f"gap: {gap:.1f}%")

    fig, axs = plt.subplots(4, 2, figsize=(10.5, 9.0), sharex=True)
    axs = axs.ravel()

    panels = [
        ("gamma", r"$\gamma$ (deg)", False),
        ("v",     r"$V/V_s$",        False),
        ("alpha", r"$\alpha$ (deg)", False),
        ("q",     r"$q$ (deg/s)",    False),
        ("de",    r"$\delta_e$ (deg)", True),
        ("dt",    r"$\delta_t$",       True),
        ("h",     r"$\Delta h$ (m)",   False),
    ]

    for ax, (key, ylabel, is_control) in zip(axs, panels):
        plot_fn = ax.step if is_control else ax.plot
        kwargs = {"where": "post"} if is_control else {}

        plot_fn(pi_data["t"], pi_data[key], color=C_PI,
                label="Policy Iteration (exact DP)", **kwargs)
        plot_fn(ppo_data["t"], ppo_data[key], color=C_PPO, linestyle="--",
                label="PPO", **kwargs)
        ax.set_ylabel(ylabel)

        if key == "gamma":
            ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
        if key == "alpha":
            ax.axhline(ALPHA_STALL_DEG, color="crimson", linewidth=0.9,
                       linestyle=":", alpha=0.8)
            ax.annotate(r"$\alpha_{stall}$",
                        xy=(0.985, ALPHA_STALL_DEG), xycoords=("axes fraction", "data"),
                        ha="right", va="bottom", fontsize=9, color="crimson")
        if key == "dt":
            ax.set_ylim(-0.05, 1.05)
        if key == "h":
            ax.axhline(dh_pi, color=C_PI, linewidth=0.8, linestyle=":")
            ax.axhline(dh_ppo, color=C_PPO, linewidth=0.8, linestyle=":")
            ax.annotate(f"{dh_pi:.2f} m", xy=(0.02, dh_pi),
                        xycoords=("axes fraction", "data"),
                        va="bottom", fontsize=9, color=C_PI)
            ax.annotate(f"{dh_ppo:.2f} m", xy=(0.02, dh_ppo),
                        xycoords=("axes fraction", "data"),
                        va="bottom", fontsize=9, color=C_PPO)

    # Summary box in the unused 8th slot
    ax_sum = axs[7]
    ax_sum.axis("off")
    handles, labels = axs[0].get_legend_handles_labels()
    ax_sum.legend(handles, labels, loc="upper center", frameon=False,
                  fontsize=11)
    ax_sum.text(
        0.5, 0.45,
        f"Altitude loss\n"
        f"DP: {dh_pi:.2f} m    PPO: {dh_ppo:.2f} m\n"
        f"reduction: {gap:.1f}%",
        ha="center", va="center", fontsize=11,
        transform=ax_sum.transAxes,
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#F4F4F4",
                  edgecolor="#888888"),
    )

    for ax in (axs[5], axs[6]):
        ax.set_xlabel("Time (s)")

    fig.align_ylabels()
    fig.tight_layout()

    out_dir = Path("results/paper")
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"fig_dp_vs_ppo.{ext}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)
    logger.info(f"[+] Saved to {out_dir / 'fig_dp_vs_ppo.{pdf,png}'}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
