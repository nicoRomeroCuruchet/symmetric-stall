"""
paper_fig_q_values.py — Publication-quality Fig. E: 1-D slices of the
action-value function Q(s, de | dt=1) for the 4-DOF symmetric stall recovery
(Riley model), at two representative states:

  (a) deep-stall entry state (canonical IC): sharp argmax -> the nose-down
      stall break is unambiguously optimal;
  (b) sliding-mode state (taken from the converged PI trajectory during the
      alpha-pinning phase): Q is nearly flat across the nose-up range ->
      the bang-bang chattering is cost-free to smooth, which is exactly
      what PPO converges to.

Since the solver ran with all shaping weights at zero (crash penalty only),
Q is expressed directly in metres of altitude-loss-to-go.

Output: results/paper/fig_q_values.{pdf,png}
"""
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from PPO_vs_PI import load_pi, run_simulation
from stable_baselines3 import PPO

from symmetric_stall.aircraft.symmetric_stall import SymmetricStall
from symmetric_stall.utils.utils import barycentric_interp_value, get_optimal_action

logger = logging.getLogger(__name__)

C_PI = "#2C4B9E"
C_PPO = "#2CA02C"
SLIDING_MODE_T = 4.0   # s — middle of the alpha-pinning phase

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "lines.linewidth": 1.6,
})


def q_slice(pi, state: np.ndarray, throttle: float = 1.0):
    """Q(s, a) for every discrete action with the given throttle setting.
    Mirrors utils.get_optimal_action_greedy (kernel reward: pure altitude
    term + crash penalty, gamma = 1)."""
    airplane = pi.env.airplane
    dt = airplane.TIME_STEP
    v_stall = airplane.STALL_AIRSPEED

    saved = (airplane.flight_path_angle, airplane.airspeed_norm,
             airplane.alpha, airplane.pitch_rate)

    mask = np.isclose(pi.action_space[:, 1], throttle)
    de_vals, q_vals = [], []
    next_state = np.empty(4, dtype=np.float32)

    for a_idx in np.flatnonzero(mask):
        de, dt_ctrl = (float(pi.action_space[a_idx, 0]),
                       float(pi.action_space[a_idx, 1]))

        (airplane.flight_path_angle, airplane.airspeed_norm,
         airplane.alpha, airplane.pitch_rate) = map(float, state)
        airplane.command_airplane(de, dt_ctrl)

        next_state[:] = (airplane.flight_path_angle, airplane.airspeed_norm,
                         airplane.alpha, airplane.pitch_rate)

        reward = dt * next_state[1] * v_stall * np.sin(next_state[0])
        if abs(next_state[2]) >= 0.698132 or next_state[0] <= -np.pi + 0.05:
            reward -= 1000.0 * v_stall

        v_next = barycentric_interp_value(
            next_state, pi.value_function, pi.bounds_low, pi.bounds_high,
            pi.grid_shape, pi.strides, pi.corner_bits,
        )
        de_vals.append(np.rad2deg(de))
        q_vals.append(reward + v_next)

    (airplane.flight_path_angle, airplane.airspeed_norm,
     airplane.alpha, airplane.pitch_rate) = saved

    order = np.argsort(de_vals)
    return np.array(de_vals)[order], np.array(q_vals)[order]


def main():
    pi = load_pi(Path("results/SymmetricStall_policy.npz"))
    ppo_model = PPO.load(Path("policy_symmetric_stall/models/best_model.zip"),
                         device="cpu")

    # State (a): canonical deep-stall IC
    state_a = np.array([0.0, 0.95, np.deg2rad(20.0), 0.0], dtype=np.float32)

    # State (b): grab it from the converged PI trajectory at t ~ 4 s
    pi_data = run_simulation(pi.env, lambda o: get_optimal_action(o, pi)[0])
    i = int(np.argmin(np.abs(pi_data["t"] - SLIDING_MODE_T)))
    state_b = np.array([
        np.deg2rad(pi_data["gamma"][i]), pi_data["v"][i],
        np.deg2rad(pi_data["alpha"][i]), np.deg2rad(pi_data["q"][i]),
    ], dtype=np.float32)
    logger.info(
        f"Sliding-mode state @ t={pi_data['t'][i]:.2f}s: "
        f"gamma={pi_data['gamma'][i]:.2f} deg, V/Vs={pi_data['v'][i]:.3f}, "
        f"alpha={pi_data['alpha'][i]:.2f} deg, q={pi_data['q'][i]:.2f} deg/s")

    fig, axs = plt.subplots(1, 2, figsize=(10.0, 3.6))

    titles = [
        rf"(a) Deep-stall entry: $\gamma=0^\circ$, $V/V_s=0.95$, "
        rf"$\alpha=20^\circ$, $q=0$",
        rf"(b) Sliding-mode phase ($t \approx {SLIDING_MODE_T:.0f}$ s): "
        rf"$\gamma={pi_data['gamma'][i]:.1f}^\circ$, "
        rf"$V/V_s={pi_data['v'][i]:.2f}$, "
        rf"$\alpha={pi_data['alpha'][i]:.1f}^\circ$, "
        rf"$q={pi_data['q'][i]:.1f}^\circ$/s",
    ]

    for ax, state, title in zip(axs, (state_a, state_b), titles):
        de, q = q_slice(pi, state, throttle=1.0)
        dq = q - q.max()  # suboptimality of each action, in metres
        ax.plot(de, dq, color=C_PI, marker="o", markersize=3.5,
                label=r"$Q(s, \delta_e, \delta_t{=}1) - \max_a Q$")

        i_best = int(np.argmax(q))
        ax.plot(de[i_best], 0.0, marker="*", markersize=14,
                color=C_PI, linestyle="none", label="DP argmax")

        ppo_act, _ = ppo_model.predict(state, deterministic=True)
        ax.axvline(np.rad2deg(ppo_act[0]), color=C_PPO, linestyle="--",
                   linewidth=1.4, label=r"PPO $\delta_e$")

        ax.annotate(f"spread: {q.max() - q.min():.3f} m",
                    xy=(0.03, 0.06), xycoords="axes fraction", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#F4F4F4",
                              edgecolor="#AAAAAA"))

        ax.set_xlabel(r"$\delta_e$ (deg)")
        ax.set_title(title, fontsize=9.5)
        ax.legend(loc="center right")

    # Shared y-scale is the point of the figure: panel (a) has a sharp
    # preference, panel (b) is flat to within millimetres.
    ymin = min(ax.get_ylim()[0] for ax in axs)
    for ax in axs:
        ax.set_ylim(ymin * 1.08, -ymin * 0.06)

    axs[0].set_ylabel(r"$Q - \max_a Q$ (m)")

    fig.tight_layout()
    out_dir = Path("results/paper")
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"fig_q_values.{ext}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)
    logger.info("[+] Saved results/paper/fig_q_values.{pdf,png}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
