"""
PPO-SymmetricStall-baseline.py — PPO baseline training (Riley aerodynamics).

Mirrors the structure of the well-trained PPO from the 4dof-symmetric-stall-PPO
branch, but uses the Riley wind-tunnel env from this branch. Differences with
this branch's primary `PPO-SymmetricStall.py` (apples-to-apples canonical-IC
optimized):
    - Wider observation bounds (so the policy is not clipped during training).
    - Constant LR (3e-4), not linearly decayed.
    - ENT_COEF = 0.005 (vs 0.015 here).
    - SB3 vanilla EvalCallback (mean reward over 30 random ICs), not the
      canonical-IC selector. Saves best_model.zip by mean normalized reward.
    - Trajectory eval IC: γ = -30° (matches main.py in the PPO branch).
    - No explicit SEED.

Outputs saved to `policy_symmetric_stall_baseline/` to not collide with the
canonical-IC PPO artifacts already on disk.
"""
import argparse
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# torch 2.12 + Python 3.14 _c10d_functional::wait_tensor double-registration
# workaround (PPO.load triggers a lazy import that re-registers the kernel).
import torch
_orig_register_autograd = torch.library.register_autograd
def _safe_register_autograd(*args, **kwargs):
    try:
        return _orig_register_autograd(*args, **kwargs)
    except RuntimeError as e:
        if "already a kernel registered" not in str(e):
            raise
torch.library.register_autograd = _safe_register_autograd

from gymnasium import spaces
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from aircraft.symmetric_stall import SymmetricStall

os.environ["NUMBA_THREADING_LAYER"] = "omp"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Observation bounds: wide, matching the PPO branch (NOT the PI grid). Letting
# the policy see the full physical envelope during exploration improves training
# stability vs clipping at the PI grid edges.
OBS_LOW  = np.array([-np.pi,     0.3, np.deg2rad(-50), np.deg2rad(-70)], dtype=np.float32)
OBS_HIGH = np.array([ np.pi / 4, 2.5, np.deg2rad( 50), np.deg2rad( 70)], dtype=np.float32)

# ── Training hyperparameters (mirror PPO branch) ──────────────────────────────
N_ENVS          = 16
N_STEPS         = 2048
BATCH_SIZE      = 512
N_EPOCHS        = 10
LEARNING_RATE   = 3e-4
GAMMA           = 0.99
GAE_LAMBDA      = 0.95
CLIP_RANGE      = 0.2
ENT_COEF        = 0.005
TOTAL_TIMESTEPS = 10_000_000
POLICY_KWARGS   = dict(net_arch=[256, 256])

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_DIR = Path("policy_symmetric_stall_baseline/models")
LOG_DIR   = Path("policy_symmetric_stall_baseline/logs")

# ── Trajectory eval IC (canonical paper IC: level flight at deep stall) ──────
# Matches main.py and PPO-SymmetricStall.py (γ=0°, deep stall recovery from
# level flight). This is the apples-to-apples IC vs PI's reported result.
GAMMA_0_DEG = 0.0
V_NORM_0    = 0.95
ALPHA_0_DEG = 20.0
Q_0_DEG     = 0.0
MAX_STEPS         = 1500
MAX_EPISODE_STEPS = 1500   # 15 s of simulation (TIME_STEP = 0.01 s)


# ── Environment wrapper ───────────────────────────────────────────────────────

class SymmetricStallEnv(SymmetricStall):
    """Riley env + SB3-compatible observation_space. No reward shaping."""

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(
            low=OBS_LOW, high=OBS_HIGH, dtype=np.float32
        )


def _make_env():
    def _init():
        return TimeLimit(SymmetricStallEnv(), max_episode_steps=MAX_EPISODE_STEPS)
    return _init


# ── Training ──────────────────────────────────────────────────────────────────

def train() -> PPO:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"[*] Spawning {N_ENVS} parallel environments...")
    vec_env = VecNormalize(
        SubprocVecEnv([_make_env() for _ in range(N_ENVS)]),
        norm_obs=False, norm_reward=True, clip_reward=10.0,
    )
    eval_env = VecNormalize(
        DummyVecEnv([lambda: Monitor(TimeLimit(SymmetricStallEnv(), max_episode_steps=MAX_EPISODE_STEPS))]),
        norm_obs=False, norm_reward=True, clip_reward=10.0, training=False,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(MODEL_DIR),
        log_path=str(LOG_DIR),
        eval_freq=max(50_000 // N_ENVS, 1),
        n_eval_episodes=30,
        deterministic=True,
        verbose=1,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(1_000_000 // N_ENVS, 1),
        save_path=str(MODEL_DIR / "checkpoints"),
        name_prefix="ppo",
        verbose=1,
    )

    callbacks = CallbackList([eval_callback, checkpoint_callback])

    logger.info("[*] Building PPO model...")
    model = PPO(
        "MlpPolicy",
        vec_env,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE,
        ent_coef=ENT_COEF,
        policy_kwargs=POLICY_KWARGS,
        device="cpu",
        verbose=1,
    )

    logger.info(f"[*] Training for {TOTAL_TIMESTEPS:,} timesteps...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks)

    vec_env.save(MODEL_DIR / "vecnormalize.pkl")
    vec_env.close()
    logger.info(f"[+] Best model saved to: {(MODEL_DIR / 'best_model.zip').resolve()}")
    return model


# ── Trajectory simulation ─────────────────────────────────────────────────────

def simulate_trajectory(model: PPO) -> dict:
    env     = SymmetricStallEnv()
    v_stall = env.airplane.STALL_AIRSPEED
    step_dt = env.airplane.TIME_STEP

    obs, _ = env.specific_reset(
        np.deg2rad(GAMMA_0_DEG), V_NORM_0,
        np.deg2rad(ALPHA_0_DEG), np.deg2rad(Q_0_DEG),
    )

    t, h      = 0.0, 0.0
    has_dived = False
    hist      = {k: [] for k in ("t", "gamma", "v", "alpha", "q", "de", "dt", "h")}

    for _ in range(MAX_STEPS):
        action, _ = model.predict(obs, deterministic=True)

        hist["t"].append(t)
        hist["gamma"].append(np.rad2deg(obs[0]))
        hist["v"].append(obs[1])
        hist["alpha"].append(np.rad2deg(obs[2]))
        hist["q"].append(np.rad2deg(obs[3]))
        hist["de"].append(np.rad2deg(action[0]))
        hist["dt"].append(action[1])
        hist["h"].append(h)

        obs, _, terminated, _, _ = env.step(action)
        v_true = obs[1] * v_stall
        h     += v_true * np.sin(obs[0]) * step_dt
        t     += step_dt

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
            logger.info(f"[+] Recovery at {t:.2f}s | altitude loss: {h:.2f} m")
            break

        if terminated and not (np.rad2deg(obs[0]) >= 0.0 and not has_dived):
            logger.warning(f"[-] Episode terminated at {t:.2f}s (h={h:.2f} m)")
            break

    return {k: np.array(v) for k, v in hist.items()}


def plot_trajectory(data: dict, prefix: str) -> None:
    C_PPO  = '#2CA02C'
    C_CTRL = '#E87C1E'
    C_ALT  = '#D62728'

    fig, axs = plt.subplots(7, 1, figsize=(8, 16), sharex=True)
    fig.suptitle(r"Symmetric Stall Recovery — PPO Baseline (Riley)", fontsize=14)

    axs[0].plot(data["t"], data["gamma"], color=C_PPO,  linewidth=2, label="PPO")
    axs[0].axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    axs[0].set_ylabel(r'$\gamma$ (deg)')
    axs[1].plot(data["t"], data["v"],     color=C_PPO,  linewidth=2, label="PPO")
    axs[1].set_ylabel(r'$V/V_s$')
    axs[2].plot(data["t"], data["alpha"], color=C_PPO,  linewidth=2, label="PPO")
    axs[2].set_ylabel(r'$\alpha$ (deg)')
    axs[3].plot(data["t"], data["q"],     color=C_PPO,  linewidth=2, label="PPO")
    axs[3].set_ylabel(r'$q$ (deg/s)')
    axs[4].step(data["t"], data["de"],    color=C_CTRL, linewidth=2, where="post", label="PPO")
    axs[4].set_ylabel(r'$\delta_e$ (deg)')
    axs[5].step(data["t"], data["dt"],    color=C_CTRL, linewidth=2, where="post", label="PPO")
    axs[5].set_ylabel(r'$\delta_t$')
    axs[5].set_ylim([-0.05, 1.05])
    axs[6].plot(data["t"], data["h"],     color=C_ALT,  linewidth=2, label="PPO")
    axs[6].set_ylabel('Altitude Loss (m)')
    axs[6].set_xlabel('Time (s)')

    for ax in axs:
        ax.grid(True, linestyle='-', alpha=0.4)
        ax.legend(loc="best")

    plt.tight_layout()
    out_path = Path("results") / f"{prefix}_trajectory.png"
    out_path.parent.mkdir(exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    logger.info(f"[+] Trajectory plot saved to: {out_path.resolve()}")


def plot_policy_heatmap(model: PPO, prefix: str) -> None:
    gamma_bins = np.linspace(np.deg2rad(-90), np.deg2rad(0),  56, dtype=np.float32)
    alpha_bins = np.linspace(np.deg2rad(-5),  np.deg2rad(20), 36, dtype=np.float32)
    v_targets  = [0.9, 1.0, 1.1]
    q_fixed    = 0.0

    gamma_deg = np.rad2deg(gamma_bins)
    alpha_deg = np.rad2deg(alpha_bins)
    A_mesh, G_mesh = np.meshgrid(alpha_deg, gamma_deg, indexing="xy")

    fig, axes = plt.subplots(3, 2, figsize=(8, 8), sharex='col', sharey='row')
    plt.subplots_adjust(wspace=0.1, hspace=0.15, bottom=0.18)
    cmap_str = 'plasma'

    for i, v_target in enumerate(v_targets):
        obs_grid = np.zeros((len(gamma_bins), len(alpha_bins), 4), dtype=np.float32)
        obs_grid[..., 0] = gamma_bins[:, None]
        obs_grid[..., 1] = v_target
        obs_grid[..., 2] = alpha_bins[None, :]
        obs_grid[..., 3] = q_fixed

        obs_flat = obs_grid.reshape(-1, 4)
        actions, _ = model.predict(obs_flat, deterministic=True)

        de_slice = np.rad2deg(actions[:, 0]).reshape(len(gamma_bins), len(alpha_bins))
        dt_slice = actions[:, 1].reshape(len(gamma_bins), len(alpha_bins))

        ax_de = axes[i, 0]
        pcm_de = ax_de.pcolormesh(
            A_mesh, G_mesh, de_slice, cmap=cmap_str, vmin=-25, vmax=15, shading='gouraud'
        )
        if i == 0:
            ax_de.set_title('Policy for Elevator', pad=10)
        ax_de.set_ylabel(r'$\gamma$ (deg)')
        ax_de.set_yticks([0, -30, -60, -90])

        ax_dt = axes[i, 1]
        pcm_dt = ax_dt.pcolormesh(
            A_mesh, G_mesh, dt_slice, cmap=cmap_str, vmin=0, vmax=1, shading='nearest'
        )
        if i == 0:
            ax_dt.set_title('Policy for Throttle', pad=10)

        ax_dt.text(
            1.05, 0.5, f'V/Vs = {v_target}',
            transform=ax_dt.transAxes, va='center', ha='left', fontsize=11
        )

        if i == 2:
            ax_de.set_xlabel(r'$\alpha$ (deg)')
            ax_dt.set_xlabel(r'$\alpha$ (deg)')

    cbar_ax_de = fig.add_axes([0.20, 0.06, 0.25, 0.02])
    cbar_ax_dt = fig.add_axes([0.55, 0.06, 0.25, 0.02])
    fig.colorbar(pcm_de, cax=cbar_ax_de, orientation='horizontal',
                 label=r'$\delta_e$ (deg)', ticks=[-20, 0, 15])
    fig.colorbar(pcm_dt, cax=cbar_ax_dt, orientation='horizontal',
                 label=r'$\delta_t$', ticks=[0.0, 0.5, 1.0])

    out_path = Path("results") / f"{prefix}_heatmaps.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"[+] Policy heatmap saved to: {out_path.resolve()}")


def main():
    global TOTAL_TIMESTEPS, N_ENVS

    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true",
                        help="quick run (100k steps, 4 envs)")
    parser.add_argument("--timesteps", type=int, default=None,
                        help="override total training timesteps")
    parser.add_argument("--eval-only", action="store_true",
                        help="skip training; regenerate trajectory + heatmap from best_model.zip")
    args = parser.parse_args()

    if args.smoke_test:
        TOTAL_TIMESTEPS = 100_000
        N_ENVS = 4
        logger.info(f"[!] SMOKE TEST — {TOTAL_TIMESTEPS:,} steps, {N_ENVS} envs")
    if args.timesteps is not None:
        TOTAL_TIMESTEPS = args.timesteps
        logger.info(f"[!] Override timesteps: {TOTAL_TIMESTEPS:,}")

    if not args.eval_only:
        train()
    else:
        logger.info("[!] EVAL-ONLY — skipping training")

    logger.info("[*] Loading best model for evaluation...")
    best_model = PPO.load(MODEL_DIR / "best_model.zip", device="cpu")

    data = simulate_trajectory(best_model)
    plot_trajectory(data, "PPO_baseline_symmetric_stall")
    plot_policy_heatmap(best_model, "PPO_baseline_symmetric_stall")


if __name__ == "__main__":
    main()
