"""
PPO-SymmetricStall.py — PPO Training for Symmetric Stall Recovery (Riley branch)

Same pipeline as the linearized branch, adapted for the Riley aerodynamic env:
- Focused reset distribution (in aircraft/symmetric_stall.py)
- CanonicalICEvalCallback saves best_model.zip by canonical-IC altitude loss
- Linear LR decay 3e-4 → 0
- ent_coef = 0.015
- NO δe² penalty (PI in this branch has w_control_effort=0 — apples-to-apples)
"""
import argparse
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Workaround for torch 2.12 + Python 3.14 _c10d_functional::wait_tensor
# double-registration bug. Adam in PPO.load() can trigger a lazy import that
# re-registers the kernel. Tolerate idempotent re-registration.
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
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from aircraft.symmetric_stall import SymmetricStall

os.environ["NUMBA_THREADING_LAYER"] = "omp"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Aligned with the PI grid (main.py setup_symmetric_stall_experiment).
OBS_LOW  = np.array([np.deg2rad(-90), 0.9, np.deg2rad(-14), np.deg2rad(-50)], dtype=np.float32)
OBS_HIGH = np.array([np.deg2rad(  5), 2.0, np.deg2rad( 20), np.deg2rad( 50)], dtype=np.float32)

# ── Training hyperparameters ──────────────────────────────────────────────────
N_ENVS          = 16
N_STEPS         = 2048
BATCH_SIZE      = 512
N_EPOCHS        = 10
# Linear decay 3e-4 → 0 over training.
LEARNING_RATE   = lambda progress_remaining: 3e-4 * progress_remaining
GAMMA           = 0.99
GAE_LAMBDA      = 0.95
CLIP_RANGE      = 0.2
# Bumped from default 0.005 to keep policy mean away from action-space boundary.
ENT_COEF        = 0.015
TOTAL_TIMESTEPS = 10_000_000
POLICY_KWARGS   = dict(net_arch=[256, 256])
SEED            = 42

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_DIR = Path("policy_symmetric_stall/models")
LOG_DIR   = Path("policy_symmetric_stall/logs")

# Canonical paper initial condition: level flight at deep stall.
GAMMA_0_DEG = 0.0
V_NORM_0    = 0.95
ALPHA_0_DEG = 20.0
Q_0_DEG     = 0.0
MAX_STEPS         = 1500
MAX_EPISODE_STEPS = 1500   # 15 s of simulation (TIME_STEP=0.01)


# ── Environment wrapper ───────────────────────────────────────────────────────

class SymmetricStallEnv(SymmetricStall):
    """Adds observation_space for SB3 compatibility. No reward shaping —
    PI in this branch optimizes pure altitude loss (w_control_effort=0), so
    PPO does too for apples-to-apples comparison."""

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(
            low=OBS_LOW, high=OBS_HIGH, dtype=np.float32
        )


def _make_env():
    def _init():
        return TimeLimit(SymmetricStallEnv(), max_episode_steps=MAX_EPISODE_STEPS)
    return _init


# ── Canonical-IC eval callback ────────────────────────────────────────────────

class CanonicalICEvalCallback(BaseCallback):
    """Evaluates by rolling out a single deterministic episode from the canonical
    paper IC (γ=0°, V/Vs=0.95, α=20°, q=0°), measuring altitude loss in meters,
    and saving best_model.zip when altitude loss improves. A crash maps to
    h=-1000 so a crashed rollout is never picked as best."""

    def __init__(self, model_dir: Path, eval_freq: int = 50_000, verbose: int = 1):
        super().__init__(verbose)
        self.model_dir = Path(model_dir)
        self.eval_freq = eval_freq
        self.best_h    = -np.inf
        self._next_eval_at = eval_freq

    def _on_step(self) -> bool:
        if self.num_timesteps < self._next_eval_at:
            return True
        self._next_eval_at += self.eval_freq

        env     = SymmetricStallEnv()
        v_stall = env.airplane.STALL_AIRSPEED
        step_dt = env.airplane.TIME_STEP

        obs, _ = env.specific_reset(
            np.deg2rad(GAMMA_0_DEG), V_NORM_0,
            np.deg2rad(ALPHA_0_DEG), np.deg2rad(Q_0_DEG),
        )

        h = 0.0
        has_dived = False
        # The env's terminated flag fires on fpa_success (γ≥0) which is true from
        # step 1 at the canonical IC (γ=0). Mirror main.py: only treat γ≥0 as a
        # real recovery after the trajectory has dived (γ<-2°). Detect crashes
        # explicitly. Otherwise, run to MAX_STEPS.
        for _ in range(MAX_STEPS):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, _, _, _ = env.step(action)
            v_true = obs[1] * v_stall
            h += v_true * np.sin(obs[0]) * step_dt

            fpa, alpha_state = obs[0], obs[2]
            crashed = (
                (alpha_state >= np.deg2rad(40)) or (alpha_state <= np.deg2rad(-40))
                or (fpa <= -np.pi + 0.05)
            )
            if crashed:
                h = -1000.0
                break

            new_gamma_deg = np.rad2deg(fpa)
            if new_gamma_deg < -2.0:
                has_dived = True
            if has_dived and new_gamma_deg >= 0.0:
                break

        if h > self.best_h:
            self.best_h = h
            save_path = self.model_dir / "best_model.zip"
            self.model.save(save_path)
            if self.verbose:
                print(f"[CanonicalEval @ {self.num_timesteps:>10,}] new best h={h:7.2f} m → saved {save_path.name}")
        else:
            if self.verbose:
                print(f"[CanonicalEval @ {self.num_timesteps:>10,}]          h={h:7.2f} m  (best {self.best_h:7.2f})")

        return True


# ── Training ──────────────────────────────────────────────────────────────────

def train() -> PPO:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    set_random_seed(SEED)
    logger.info(f"[*] Seeded numpy/torch/sb3 with {SEED}")

    logger.info(f"[*] Spawning {N_ENVS} parallel environments...")
    vec_env  = VecNormalize(
        SubprocVecEnv([_make_env() for _ in range(N_ENVS)]),
        norm_obs=False, norm_reward=True, clip_reward=10.0,
    )
    vec_env.seed(SEED)

    eval_callback = CanonicalICEvalCallback(
        model_dir=MODEL_DIR,
        eval_freq=50_000,
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
        seed=SEED,
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

        obs, _, _, _, _ = env.step(action)
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

        # Explicit crash detection (env's `terminated` also fires on γ≥0 success,
        # which is true from step 1 at the canonical IC γ=0).
        if (obs[2] >= np.deg2rad(40) or obs[2] <= np.deg2rad(-40)
                or obs[0] <= -np.pi + 0.05):
            logger.warning(f"[-] Crash at {t:.2f}s")
            break

    return {k: np.array(v) for k, v in hist.items()}


def plot_trajectory(data: dict, prefix: str) -> None:
    C_PPO  = '#2CA02C'
    C_CTRL = '#E87C1E'
    C_ALT  = '#D62728'

    fig, axs = plt.subplots(7, 1, figsize=(8, 16), sharex=True)
    fig.suptitle(r"Symmetric Stall Recovery — PPO Policy (Riley)", fontsize=14)

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
    global TOTAL_TIMESTEPS, N_ENVS, SEED

    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true",
                        help="quick run (100k steps, 4 envs)")
    parser.add_argument("--timesteps", type=int, default=None,
                        help="override total training timesteps")
    parser.add_argument("--seed", type=int, default=None,
                        help=f"RNG seed (default: {SEED})")
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
    if args.seed is not None:
        SEED = args.seed
        logger.info(f"[!] Override seed: {SEED}")

    if not args.eval_only:
        train()
    else:
        logger.info("[!] EVAL-ONLY — skipping training")

    logger.info("[*] Loading best model for evaluation...")
    best_model = PPO.load(MODEL_DIR / "best_model.zip", device="cpu")

    data = simulate_trajectory(best_model)
    plot_trajectory(data, "PPO_symmetric_stall")
    plot_policy_heatmap(best_model, "PPO_symmetric_stall")


if __name__ == "__main__":
    main()
