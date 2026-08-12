"""
PPO_vs_PI.py — Side-by-side trajectory comparison (Riley branch)

Runs both DP/Policy-Iteration and PPO from the same canonical IC, overlays the
two trajectories on a 7-panel figure.
"""
import logging
import os
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

# torch 2.12 + Python 3.14 wait_tensor double-registration workaround
import torch
_orig_register_autograd = torch.library.register_autograd
def _safe_register_autograd(*args, **kwargs):
    try:
        return _orig_register_autograd(*args, **kwargs)
    except RuntimeError as e:
        if "already a kernel registered" not in str(e):
            raise
torch.library.register_autograd = _safe_register_autograd

from stable_baselines3 import PPO

os.environ["NUMBA_THREADING_LAYER"] = "omp"

from symmetric_stall.aircraft.symmetric_stall import SymmetricStall
from symmetric_stall.policy_iteration import PolicyIterationStall
from symmetric_stall.utils.utils import get_optimal_action

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Canonical IC (same as main.py and PPO-SymmetricStall.py)
GAMMA_0_DEG  =  0.0
V_NORM_0     =  0.95
ALPHA_0_DEG  = 20.0
Q_0_DEG      =  0.0
MAX_STEPS    = 1500


def run_simulation(
    env: SymmetricStall,
    get_action: Callable[[np.ndarray], np.ndarray],
) -> dict:
    v_stall = env.airplane.STALL_AIRSPEED
    step_dt = env.airplane.TIME_STEP

    obs, _ = env.specific_reset(
        np.deg2rad(GAMMA_0_DEG), V_NORM_0,
        np.deg2rad(ALPHA_0_DEG), np.deg2rad(Q_0_DEG),
    )

    t, h = 0.0, 0.0
    has_dived = False
    hist = {k: [] for k in ("t", "gamma", "v", "alpha", "q", "de", "dt", "h")}

    for _ in range(MAX_STEPS):
        action = get_action(obs)

        hist["t"].append(t)
        hist["gamma"].append(np.rad2deg(obs[0]))
        hist["v"].append(obs[1])
        hist["alpha"].append(np.rad2deg(obs[2]))
        hist["q"].append(np.rad2deg(obs[3]))
        hist["de"].append(np.rad2deg(action[0]))
        hist["dt"].append(action[1])
        hist["h"].append(h)

        obs, _, _, _, _ = env.step(action)
        v_true = obs[1] * v_stall
        h += v_true * np.sin(obs[0]) * step_dt
        t += step_dt

        if t >= 15.0:
            break

        new_gamma = np.rad2deg(obs[0])
        if new_gamma < -2.0:
            has_dived = True

        if has_dived and new_gamma >= 0.0:
            hist["t"].append(t)
            hist["gamma"].append(new_gamma)
            hist["v"].append(obs[1])
            hist["alpha"].append(np.rad2deg(obs[2]))
            hist["q"].append(np.rad2deg(obs[3]))
            hist["de"].append(hist["de"][-1])
            hist["dt"].append(hist["dt"][-1])
            hist["h"].append(h)
            logger.info(f"  Recovery at {t:.2f}s  |  altitude loss: {h:.2f} m")
            break

        # Explicit crash detection (env's `terminated` also fires on γ≥0 success,
        # true from step 1 at canonical IC γ=0).
        if (obs[2] >= np.deg2rad(40) or obs[2] <= np.deg2rad(-40)
                or obs[0] <= -np.pi + 0.05):
            logger.warning(f"  Crash at {t:.2f}s")
            break

    return {k: np.array(v) for k, v in hist.items()}


def load_pi(policy_path: Path) -> PolicyIterationStall:
    env = SymmetricStall()
    pi = PolicyIterationStall.load(policy_path, env=env)

    # Rebuild states_space so get_optimal_action has the grid metadata.
    # MUST match main.py setup_symmetric_stall_experiment exactly.
    bins_space = {
        "flight_path_angle": np.linspace(np.deg2rad(-90), np.deg2rad(5),  56, dtype=np.float32),
        "airspeed_norm":     np.linspace(0.9,             2.0,            41, dtype=np.float32),
        "alpha":             np.linspace(np.deg2rad(-40), np.deg2rad(20), 60, dtype=np.float32),
        "pitch_rate":        np.linspace(np.deg2rad(-50), np.deg2rad(50), 41, dtype=np.float32),
    }
    grid = np.meshgrid(*bins_space.values(), indexing="ij")
    pi.states_space = np.vstack([g.ravel() for g in grid]).astype(np.float32).T
    return pi


def main():
    pi_path  = Path("results/SymmetricStall_policy.npz")
    ppo_path = Path("policy_symmetric_stall/models/best_model.zip")

    logger.info(f"[*] Loading PI policy from: {pi_path.resolve()}")
    pi = load_pi(pi_path)
    pi_env = pi.env
    logger.info(f"[+] PI policy loaded: '{pi_path.name}'")

    logger.info(f"[*] Loading PPO model from: {ppo_path.resolve()}")
    ppo_model = PPO.load(ppo_path, device="cpu")
    ppo_env = SymmetricStall()
    logger.info(f"[+] PPO model loaded: '{ppo_path.name}'")

    def pi_action(obs: np.ndarray) -> np.ndarray:
        action, _, _ = get_optimal_action(obs, pi)
        return action

    def ppo_action(obs: np.ndarray) -> np.ndarray:
        action, _ = ppo_model.predict(obs, deterministic=True)
        return action

    logger.info("[*] Simulating PI policy…")
    pi_data = run_simulation(pi_env, pi_action)

    logger.info("[*] Simulating PPO policy…")
    ppo_data = run_simulation(ppo_env, ppo_action)

    C_PI  = '#2C4B9E'
    C_PPO = '#2CA02C'

    fig, axs = plt.subplots(7, 1, figsize=(9, 18), sharex=True)
    fig.suptitle(
        r"Symmetric Stall Recovery — PPO vs Policy Iteration (Riley)",
        fontsize=14, fontweight='bold',
    )

    panels = [
        ("gamma", r"$\gamma$ (deg)",     False),
        ("v",     r"$V/V_s$",            False),
        ("alpha", r"$\alpha$ (deg)",     False),
        ("q",     r"$q$ (deg/s)",        False),
        ("de",    r"$\delta_e$ (deg)",   True ),
        ("dt",    r"$\delta_t$",         True ),
        ("h",     "Altitude Loss (m)",   False),
    ]

    for ax, (key, ylabel, is_control) in zip(axs, panels):
        plot_fn = ax.step if is_control else ax.plot
        kwargs  = {"where": "post"} if is_control else {}

        plot_fn(pi_data["t"],  pi_data[key],  color=C_PI,  linewidth=2,
                label="Policy Iteration", **kwargs)
        plot_fn(ppo_data["t"], ppo_data[key], color=C_PPO, linewidth=2,
                label="PPO", linestyle="--", **kwargs)

        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="-", alpha=0.4)
        ax.legend(loc="best", fontsize=8)

        if key == "gamma":
            ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)
        if key == "dt":
            ax.set_ylim([-0.05, 1.05])

    axs[-1].set_xlabel("Time (s)")
    plt.tight_layout()

    out_path = Path("results/PPO_vs_PI_trajectory.png")
    out_path.parent.mkdir(exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    logger.info(f"[+] Comparison plot saved to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
