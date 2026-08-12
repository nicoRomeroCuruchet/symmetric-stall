"""
train.py — Pure Markovian DP Simulation
Executes the mathematically rigorous DP policy.
Includes persistent policy caching and precise physical altitude tracking.

The state grid is selected with the `grid` argument (see GRIDS), NOT by editing
this file. The thrust model and the CG offset arrive through environment
variables that are read AT IMPORT TIME by the plant, so they are set from
`symmetric_stall.cli` before this module is imported. See `runconfig.py`.
"""
import logging
import os
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

os.environ["NUMBA_THREADING_LAYER"] = "omp"

from symmetric_stall import runconfig  # noqa: E402
from symmetric_stall.aircraft.symmetric_stall import SymmetricStall  # noqa: E402
from symmetric_stall.policy_iteration import (  # noqa: E402
    PolicyIterationStall,
    PolicyIterationStallConfig,
)
from symmetric_stall.utils.utils import (  # noqa: E402
    get_optimal_action,
    get_optimal_action_greedy,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")
POLICY_DIR = Path("data/policies")

# ── State grids ──────────────────────────────────────────────────────────
#
# This used to be selected by EDITING this file with `set_grilla_paper.py` (now
# in attic/), which left the comments describing one grid while the code ran
# another, and kept no record of which grid had trained each .npz. There are
# now two named presets, and the name travels into the .npz.
#
# The two axes that do NOT change between presets:
#   flight_path_angle : 56 bins over [-90, +5] deg     (~1.7 deg)
#   pitch_rate        : 41 bins over [-50, +50] deg/s  (~2.5 deg/s)

GRIDS: dict[str, dict] = {
    # Paper-1 grid (56x41x60x41 = 5,648,160 states, ~43 min on a 3090).
    # Kept in order to separate the effect of the grid from the effect of the
    # thrust model: running it with THRUST_MODEL=riley isolates each
    # contribution. CAUTION: its alpha axis reaches down to -40 deg, outside
    # what Riley tabulates -- see the note on the riley preset below.
    "paper1": {
        "airspeed_norm": (0.9, 2.0, 41),
        "alpha_deg": (-40.0, 20.0, 60),
    },
    # Riley grid (56x81x80x41 = 14,878,080 states, 47 iter / 4h18 on a 3090).
    # This is the one that trains the policy behind every figure in the paper.
    #
    # airspeed_norm: 81 bins (step 0.020 Vs, previously 41 bins of 0.0275 over
    # [0.9, 2.0]). The floor drops to 0.4 Vs so that the low-speed region lies
    # INSIDE the grid instead of being resolved by saturating against the edge.
    # Below 0.785 Vs the physical C_T exceeds 0.5 and leaves Riley's tables;
    # that is where the Appendix B dCD_T increment kicks in, which is precisely
    # the term the model carries for that case.
    #
    # alpha: 80 bins (~0.63 deg). The axis matches EXACTLY the domain Riley
    # tabulates: his alpha points are
    #     [-10 -5 0 5 10 12 14 16 18 20 25 30 35 40]
    # and outside that range np.interp CLAMPS, i.e. it returns the same C_L and
    # C_D for -12 as for -38. With the floor at -40 (the paper1 preset), 30 of
    # its 60 bins -- half the axis -- run on fabricated coefficients.
    #
    # THE CEILING at +40 is the one that matters and is NOT to be touched: that
    # is where the crash plane lives (alpha >= 40), same as in the 8-DOF
    # branch. With the ceiling at +20 the kernel charged the crash penalty and
    # then interpolation clamped back into a NON-terminal cell: the state
    # re-charged it on every sweep and the value diverged. Measured on the
    # 6-DOF branch, which still has the old ceiling: 4.99% of its grid is worth
    # LESS than crashing, with a median of 9 crashes and a worst case of 67. If
    # failure were absorbing, nothing could be worth less than -1000*v_stall.
    #
    # THE FLOOR at -10 does not reintroduce that problem, because there is
    # nothing to charge below it: alpha <= -40 falls outside the grid and never
    # fires. What remains is a reflecting wall -- the aircraft cannot go past
    # -10 deg of incidence -- and that is a declared modelling constraint, not
    # a diverging value. To be verified against the trained policy: what
    # fraction of the trajectories leans on that floor.
    "riley": {
        "airspeed_norm": (0.4, 2.0, 81),
        "alpha_deg": (-10.0, 40.0, 80),
    },
}

DEFAULT_GRID = "riley"


def grid_shape(grid: str = DEFAULT_GRID) -> tuple[int, int, int, int]:
    """(gamma, V, alpha, q) bin counts for the preset — used to name files."""
    g = GRIDS[grid]
    return (56, g["airspeed_norm"][2], g["alpha_deg"][2], 41)


# ── Experiment Setup ─────────────────────────────────────────────────────


def setup_symmetric_stall_experiment(grid: str = DEFAULT_GRID) -> tuple[
    gym.Env, np.ndarray, np.ndarray, PolicyIterationStallConfig
]:
    """Configures the experiment with physically bounded state grids.

    `grid` selects a preset from GRIDS. The default is "riley", the grid that
    trains the policy used in the paper.
    """
    if grid not in GRIDS:
        raise ValueError(
            f"unknown grid: {grid!r}. Options: {sorted(GRIDS)}"
        )
    spec = GRIDS[grid]
    v_lo, v_hi, v_n = spec["airspeed_norm"]
    a_lo, a_hi, a_n = spec["alpha_deg"]

    env = SymmetricStall()

    bins_space = {
        # 56 bins (~1.7° resolution)
        "flight_path_angle": np.linspace(
            np.deg2rad(-90), np.deg2rad(5), 56, dtype=np.float32
        ),
        "airspeed_norm": np.linspace(v_lo, v_hi, v_n, dtype=np.float32),
        "alpha": np.linspace(
            np.deg2rad(a_lo), np.deg2rad(a_hi), a_n, dtype=np.float32
        ),
        # 41 bins (~2.5°/s resolution)
        "pitch_rate": np.linspace(
            np.deg2rad(-50), np.deg2rad(50), 41, dtype=np.float32
        ),
    }
    mesh = np.meshgrid(*bins_space.values(), indexing="ij")
    states_space = np.vstack(
        [g.ravel() for g in mesh]
    ).astype(np.float32).T

    # 41 × 11 = 451 actions. δe at 1°/level (was 2°/level with 21 bins),
    # δt at 0.1/level (was 0.167/level with 7 bins). Same refinement applied
    # in the linearized branch — net change there was negligible, but Riley's
    # smoother CL saturation may respond differently to finer action grid.
    de_vals = np.linspace(
        np.deg2rad(-25), np.deg2rad(15), 41, dtype=np.float32
    )
    dt_vals = np.linspace(0.0, 1.0, 11, dtype=np.float32)
    action_grid = np.meshgrid(de_vals, dt_vals, indexing="ij")
    action_space = np.vstack(
        [a.ravel() for a in action_grid]
    ).astype(np.float32).T

    logger.info(
        "[grid] preset '%s' %s = %s states | V [%.2f, %.2f] Vs | "
        "alpha [%.0f, %.0f] deg | thrust %s",
        grid, grid_shape(grid), f"{states_space.shape[0]:,}",
        v_lo, v_hi, a_lo, a_hi, runconfig.describe()["thrust_model"],
    )

    config = PolicyIterationStallConfig(
        gamma=1.0,
        theta=5e-6,
        n_steps=1000,
        log=False,
        log_interval=10,
        w_q_penalty=0.0,
        w_control_effort=0.0,
        w_alpha_barrier_pos=0.0,
        w_alpha_barrier_neg=0.0,
        # Matches env.step()'s -1000*v_stall catastrophic-failure penalty so
        # PI explicitly costs intra-RK4 transitions into the crash region,
        # instead of silently bootstrapping V from the clamped grid edge.
        # PPO already sees this reward via env.step(), so apples-to-apples
        # is preserved (both algorithms now optimize the same reward signal).
        w_crash_penalty=1000.0,
        w_throttle_bonus=0.0,
    )
    return env, states_space, action_space, config


# ── Policy Training / Loading ────────────────────────────────────────────


def assert_grid_matches(pi, grid: str = DEFAULT_GRID, source=None) -> None:
    """Fail if the loaded policy was not trained on grid preset `grid`.

    The figure scripts REBUILD the state space with
    `setup_symmetric_stall_experiment()` and attach it to the loaded policy
    (`pi.states_space = states`). If the policy came from a different grid the
    pairing is silent and the trajectories come out wrong without warning. The
    .npz stores its own `grid_shape`, so the mismatch can be caught.
    """
    expected = grid_shape(grid)
    actual = tuple(int(n) for n in pi.grid_shape)
    if actual != expected:
        where = f" ({source})" if source else ""
        raise ValueError(
            f"The policy{where} was trained on grid {actual} but is being used "
            f"with preset '{grid}' = {expected}. These are different grids: "
            f"either pass the right preset, or use a different policy.\n"
            f"  presets: " + ", ".join(f"{k}={grid_shape(k)}" for k in GRIDS)
        )
    meta = getattr(pi, "run_metadata", None)
    if meta and meta.get("grid") and meta["grid"] != grid:
        logger.warning(
            "[!] The policy claims to have been trained on grid '%s' but is "
            "being used as '%s'. The shapes agree, so continuing.",
            meta["grid"], grid,
        )


def default_policy_path(
    env: gym.Env, config: PolicyIterationStallConfig, grid: str = DEFAULT_GRID
) -> Path:
    """`data/policies/SymmetricStall_riley_56x81x80x41_thrust-riley.npz`."""
    shape = "x".join(str(n) for n in grid_shape(grid))
    scheme = "_mca" if config.use_mca_timestep else ""
    name = (
        f"{env.unwrapped.__class__.__name__}_{grid}_{shape}"
        f"_{runconfig.slug()}{scheme}.npz"
    )
    return POLICY_DIR / name


def train_or_load_policy(
    env: gym.Env,
    states: np.ndarray,
    actions: np.ndarray,
    config: PolicyIterationStallConfig,
    prefix: str,
    warm_start_ppo: Path | None = None,
    grid: str = DEFAULT_GRID,
    policy_path: Path | None = None,
) -> PolicyIterationStall:
    """Trains or loads pre-trained tensors from disk.

    If `warm_start_ppo` is set and no cached policy exists, PI's d_policy
    is initialized from the PPO actor at the given path before run().

    The .npz filename encodes grid + thrust model + CG. It used to be always
    `SymmetricStall_policy.npz`, so two runs on different grids overwrote each
    other and there was no way to tell them apart.
    """
    if policy_path is None:
        policy_path = default_policy_path(env, config, grid)
    policy_path.parent.mkdir(parents=True, exist_ok=True)

    if policy_path.exists():
        logger.info(f"[+] Existing policy found: {policy_path.name}. Loading...")
        try:
            pi = PolicyIterationStall.load(policy_path, env=env)
            assert_grid_matches(pi, grid, policy_path)
            pi.states_space = states
            logger.info("[+] Policy loaded successfully. Skipping training.")
            return pi
        except Exception as e:
            logger.error(f"[-] Failed to load policy: {e}. Forcing retrain...")

    logger.info(f"[*] Training new policy for {prefix}...")
    pi = PolicyIterationStall(env, states, actions, config)

    if warm_start_ppo is not None:
        pi.warm_start_policy_from_ppo(warm_start_ppo)

    # run() does not save; the single write happens here, with the metadata.
    pi.run()

    pi.save(policy_path, metadata={
        "grid": grid,
        "grid_shape": list(grid_shape(grid)),
        "n_states": int(np.prod(grid_shape(grid))),
        **runconfig.describe(),
        "gamma": config.gamma,
        "theta": config.theta,
        "n_micro": config.n_micro,
        "use_mca_timestep": bool(config.use_mca_timestep),
        # How the run ended, not just how it was configured. Without these a
        # policy cannot say whether it converged or ran out of iterations.
        "n_policy_steps": int(getattr(pi, "n_policy_steps", 0)),
        "final_residual": float(getattr(pi, "final_residual", float("nan"))),
        "n_states_chattering": int(getattr(pi, "n_states_chattering", 0)),
        "chattering_tolerance": int(np.prod(grid_shape(grid)) * 1e-4),
    })
    logger.info(f"[+] Policy cached to {policy_path.resolve()}")
    return pi


# ── DP Simulation ────────────────────────────────────────────────────────


def run_dp_simulation(
    pi: PolicyIterationStall,
    gamma_0_deg: float,
    v_norm_0: float,
    alpha_0_deg: float,
    q_0_deg: float,
    max_steps: int = 1500,
    greedy: bool = False,
    actuator_tau: float = 0.0,
) -> dict[str, list]:
    """
    Runs a pure DP simulation and returns the full state history.

    All angular values in the returned history are in radians.

    `greedy=False` (default, Approach A): convex-action averaging.
    `greedy=True`  (Approach B): re-derive argmax Q over the discrete
                                  action set at each control step.

    `actuator_tau` (s): time constant of an EMA filter applied to the
    raw PI policy output. Models real actuator bandwidth.
        0.0  → disabled (raw policy passthrough)
        0.02 → mild smoothing (~8 Hz cutoff)
        0.05 → moderate (~3 Hz cutoff)
        0.10 → strong (~1.5 Hz cutoff)
        0.20 → heavy (~0.8 Hz cutoff, may miss fast transients)
    """
    env = pi.env
    v_stall = env.airplane.STALL_AIRSPEED
    dt = env.airplane.TIME_STEP
    control_dt = 0.01
    steps_per_control = max(1, int(round(control_dt / dt)))

    # EMA coefficient: action_new = α·raw + (1-α)·prev
    # α = 1.0  → no smoothing (passthrough)
    # α → 0.0  → action stays at previous forever
    if actuator_tau > 0.0:
        ema_alpha = control_dt / (actuator_tau + control_dt)
        logger.info(
            f"[*] EMA actuator filter: τ={actuator_tau:.3f}s "
            f"(α={ema_alpha:.3f}, cutoff~{1.0 / (2 * np.pi * actuator_tau):.1f} Hz)"
        )
    else:
        ema_alpha = 1.0

    action_fn = get_optimal_action_greedy if greedy else get_optimal_action
    logger.info(
        f"[*] Eval policy: {'B (greedy argmax over discrete actions)' if greedy else 'A (convex action blend)'}"
    )

    obs, _ = env.specific_reset(
        np.deg2rad(gamma_0_deg),
        v_norm_0,
        np.deg2rad(alpha_0_deg),
        np.deg2rad(q_0_deg),
    )

    hist = {
        k: []
        for k in [
            "t", "gamma", "v_norm", "alpha", "q", "de", "dt_ctrl", "h"
        ]
    }

    t, h = 0.0, 0.0
    has_dived = False
    action = np.array([0.0, 0.0], dtype=np.float32)
    action_prev = None
    current_action_idx = None

    for step in range(max_steps):
        if step % steps_per_control == 0:
            if greedy:
                action_raw, _, current_action_idx = action_fn(obs, pi)
            else:
                action_raw, _, current_action_idx = action_fn(
                    obs, pi, current_action_idx
                )
            action_raw = np.asarray(action_raw, dtype=np.float32)

            if action_prev is not None and ema_alpha < 1.0:
                action = ema_alpha * action_raw + (1.0 - ema_alpha) * action_prev
            else:
                action = action_raw
            action_prev = action

        hist["t"].append(t)
        hist["gamma"].append(obs[0])
        hist["v_norm"].append(obs[1])
        hist["alpha"].append(obs[2])
        hist["q"].append(obs[3])
        hist["de"].append(action[0])
        hist["dt_ctrl"].append(action[1])
        hist["h"].append(h)

        obs, _, terminated, _, _ = env.step(action)

        v_true = obs[1] * v_stall
        h += v_true * np.sin(obs[0]) * dt
        t += dt

        new_gamma = obs[0]
        if new_gamma < np.deg2rad(-1.0):
            has_dived = True

        if has_dived and new_gamma >= 0.0:
            # Append final recovered state
            hist["t"].append(t)
            hist["gamma"].append(new_gamma)
            hist["v_norm"].append(obs[1])
            hist["alpha"].append(obs[2])
            hist["q"].append(obs[3])
            hist["de"].append(action[0])
            hist["dt_ctrl"].append(action[1])
            hist["h"].append(h)
            logger.info(
                f"[+] Recovery success at {t:.2f}s, Δh={h:.2f}m"
            )
            break

        if terminated:
            if obs[2] >= np.deg2rad(40):
                logger.warning(f"[-] +Alpha limit at {t:.2f}s")
                break
            elif obs[2] <= np.deg2rad(-40):
                logger.warning(f"[-] -Alpha limit at {t:.2f}s")
                break
            elif obs[0] <= -np.pi + 0.05:
                logger.warning(f"[-] Catastrophic dive at {t:.2f}s")
                break

    return hist


# ── Plotting ─────────────────────────────────────────────────────────────


def plot_time_response(hist: dict, prefix: str) -> None:
    """Generates the 7-panel time-response figure from a simulation history.

    Publication styling: serif/STIX fonts matching the paper body, vector
    PDF alongside the PNG, panel letters, the elevator-switch event t*
    marked across all panels, the stall boundary in the alpha panel, and
    the final altitude loss annotated.
    """
    t = np.asarray(hist["t"])
    gamma_deg = np.rad2deg(hist["gamma"])
    v_norm = np.asarray(hist["v_norm"])
    alpha_deg = np.rad2deg(hist["alpha"])
    q_deg = np.rad2deg(hist["q"])
    de_deg = np.rad2deg(hist["de"])
    dt_ctrl = np.asarray(hist["dt_ctrl"])
    h = np.asarray(hist["h"])

    # Color convention matching Grillo et al. (2023): state variables in
    # blue, control commands in orange, altitude loss in red.
    C_STATE, C_CTRL, C_ALT = "#1F77B4", "#FF7F0E", "#D62728"
    C_REF = "0.45"  # reference/event lines

    # Elevator switch event: first nose-up command after the initial hold.
    i_sw = int(np.argmax(de_deg < 0.0)) if np.any(de_deg < 0.0) else None
    t_sw = t[i_sw] if i_sw else None

    rc = {
        "font.family": "serif", "mathtext.fontset": "stix",
        "font.size": 12, "axes.labelsize": 13,
        "xtick.labelsize": 11, "ytick.labelsize": 11,
        "axes.spines.top": False, "axes.spines.right": False,
    }
    with plt.rc_context(rc):
        # Layout: paired states (gamma|V, alpha|q), paired controls
        # (de|dt), taller full-width altitude-loss panel at the bottom.
        fig = plt.figure(figsize=(9.0, 9.0))
        gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1, 1.4],
                              hspace=0.30, wspace=0.26)
        specs = [
            (gs[0, 0], gamma_deg, r"$\gamma$ (deg)", C_STATE, "plot"),
            (gs[0, 1], v_norm, r"$V/V_s$ (--)", C_STATE, "plot"),
            (gs[1, 0], alpha_deg, r"$\alpha$ (deg)", C_STATE, "plot"),
            (gs[1, 1], q_deg, r"$q$ (deg/s)", C_STATE, "plot"),
            (gs[2, 0], de_deg, r"$\delta_e$ (deg)", C_CTRL, "step"),
            (gs[2, 1], dt_ctrl, r"$\delta_t$ (--)", C_CTRL, "step"),
            (gs[3, :], h, r"$\Delta h$ (m)", C_ALT, "plot"),
        ]

        axs = []
        for k, (spec, data, ylabel, color, style) in enumerate(specs):
            ax = fig.add_subplot(spec)
            if style == "step":
                ax.step(t, data, color=color, linewidth=1.8, where="post")
            else:
                ax.plot(t, data, color=color, linewidth=1.8)
            ax.set_ylabel(ylabel)
            ax.set_xlim(t[0], t[-1])
            ax.grid(True, linestyle=":", alpha=0.55)
            if t_sw is not None:
                ax.axvline(t_sw, color=C_REF, linestyle=":", linewidth=1.0)
            ax.text(0.0, 1.02, f"({'abcdefg'[k]})",
                    transform=ax.transAxes, fontsize=11,
                    va="bottom", ha="left")
            axs.append(ax)

        # Reference lines and event annotations.
        axs[0].axhline(0.0, color=C_REF, linestyle="--", linewidth=0.9)
        axs[1].axhline(1.0, color=C_REF, linestyle="--", linewidth=0.9)
        axs[2].axhline(14.0, color=C_REF, linestyle="--", linewidth=0.9)
        axs[2].annotate(r"$\alpha_s = 14^\circ$ (power-off)",
                        xy=(0.98, 14.0), xycoords=("axes fraction", "data"),
                        xytext=(0, 5), textcoords="offset points",
                        ha="right", fontsize=10, color=C_REF)
        axs[3].axhline(0.0, color=C_REF, linestyle="--", linewidth=0.9)
        axs[4].axhline(0.0, color=C_REF, linestyle="--", linewidth=0.9)
        axs[5].set_ylim([-0.05, 1.05])
        if t_sw is not None:
            axs[4].annotate(r"$t^\ast$",
                            xy=(t_sw, 0.02), xycoords=("data", "axes fraction"),
                            xytext=(5, 0), textcoords="offset points",
                            fontsize=11, color=C_REF)
        axs[6].plot(t[-1], h[-1], marker="o", ms=5, color=C_ALT)
        axs[6].annotate(f"{h[-1]:.2f} m at $t = {t[-1]:.1f}$ s",
                        xy=(t[-1], h[-1]), xytext=(-10, 10),
                        textcoords="offset points", ha="right",
                        fontsize=11, color=C_ALT)

        for ax in axs[:-1]:
            ax.tick_params(labelbottom=False)
        axs[-1].set_xlabel("Time (s)")
        fig.align_ylabels(axs)

        RESULTS_DIR.mkdir(exist_ok=True)
        for ext in ("png", "pdf"):
            out_path = RESULTS_DIR / f"{prefix}_trajectory.{ext}"
            plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"[*] Plot saved to: "
                    f"{(RESULTS_DIR / (prefix + '_trajectory.png')).resolve()}")


def plot_heatmaps(
    pi: PolicyIterationStall,
    prefix: str,
    alpha_window_deg: tuple[float, float] = (-5.1, 20.1),
) -> None:
    """Generates the 3x3 heatmap figure (elevator, throttle, altitude loss).

    `alpha_window_deg` crops the displayed alpha range; pass (-40.1, 20.1)
    to render the full grid width instead of the default recovery window.
    """
    logger.info("[*] Extracting 4D Tensors for heatmaps...")

    gamma_bins = np.unique(pi.states_space[:, 0])
    v_bins = np.unique(pi.states_space[:, 1])
    alpha_bins = np.unique(pi.states_space[:, 2])
    q_bins = np.unique(pi.states_space[:, 3])

    shape_4d = (
        len(gamma_bins), len(v_bins), len(alpha_bins), len(q_bins)
    )
    V_4D = pi.value_function.reshape(shape_4d)
    Pol_4D = pi.policy.reshape(shape_4d)

    q_idx = int(np.argmin(np.abs(q_bins - 0.0)))
    v_targets = [0.9, 1.0, 1.1]

    gamma_mask = (gamma_bins >= np.deg2rad(-90.1)) & (gamma_bins <= 0.01)
    alpha_mask = (
        (alpha_bins >= np.deg2rad(alpha_window_deg[0]))
        & (alpha_bins <= np.deg2rad(alpha_window_deg[1]))
    )

    gamma_deg = np.rad2deg(gamma_bins[gamma_mask])
    alpha_deg = np.rad2deg(alpha_bins[alpha_mask])
    A_mesh, G_mesh = np.meshgrid(alpha_deg, gamma_deg, indexing="xy")

    fig, axes = plt.subplots(
        3, 3, figsize=(11, 8), sharex="col", sharey="row"
    )
    plt.subplots_adjust(wspace=0.1, hspace=0.15, bottom=0.2)

    for i, v_target in enumerate(v_targets):
        v_idx = int(np.argmin(np.abs(v_bins - v_target)))
        v_slice = V_4D[:, v_idx, :, q_idx][gamma_mask][:, alpha_mask]
        p_slice = Pol_4D[:, v_idx, :, q_idx][gamma_mask][:, alpha_mask]

        de_slice = np.array([
            [np.rad2deg(pi.action_space[p_slice[g, a], 0])
             for a in range(p_slice.shape[1])]
            for g in range(p_slice.shape[0])
        ], dtype=np.float32)

        dt_slice = np.array([
            [pi.action_space[p_slice[g, a], 1]
             for a in range(p_slice.shape[1])]
            for g in range(p_slice.shape[0])
        ], dtype=np.float32)

        alt_loss_slice = -v_slice

        # Elevator
        axes[i, 0].pcolormesh(
            A_mesh, G_mesh, de_slice,
            cmap="plasma", vmin=-25, vmax=15, shading="gouraud",
        )
        # Throttle
        axes[i, 1].pcolormesh(
            A_mesh, G_mesh, dt_slice,
            cmap="plasma", vmin=0, vmax=1, shading="nearest",
        )
        # Altitude loss
        axes[i, 2].pcolormesh(
            A_mesh, G_mesh, alt_loss_slice,
            cmap="plasma", vmin=0, vmax=100, shading="gouraud",
        )

        axes[i, 2].text(
            1.05, 0.5, f"V/Vs = {v_target}",
            transform=axes[i, 2].transAxes,
            va="center", ha="left", fontsize=11,
        )

    titles = ["Policy for Elevator", "Policy for Throttle", "Altitude Loss"]
    for j, title in enumerate(titles):
        axes[0, j].set_title(title, pad=10)

    alpha_ticks = (
        [0, 10, 20] if alpha_window_deg[0] > -6.0
        else [-40, -30, -20, -10, 0, 10, 20]
    )
    for j in range(3):
        axes[2, j].set_xlabel(r"$\alpha$ (deg)")
        axes[2, j].set_xticks(alpha_ticks)

    for i in range(3):
        axes[i, 0].set_ylabel(r"$\gamma$ (deg)")
        axes[i, 0].set_yticks([0, -30, -60, -90])

    # Colorbars
    cbar_specs = [
        (0, r"$\delta_e$ (deg)", [-20, 0, 15]),
        (1, r"$\delta_t$", [0.0, 0.5, 1.0]),
        (2, "Altitude Loss (m)", [0, 100]),
    ]
    for j, (col, label, ticks) in enumerate(cbar_specs):
        cax = fig.add_axes([0.15 + 0.27 * j, 0.05, 0.2, 0.02])
        mappable = axes[2, col].collections[0]
        fig.colorbar(
            mappable, cax=cax, orientation="horizontal",
            label=label, ticks=ticks,
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{prefix}_heatmaps.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"[+] Heatmaps saved to {out_path.resolve()}")


# ── CasADi Validation ────────────────────────────────────────────────────


def validate_with_casadi(pi: PolicyIterationStall, prefix: str) -> None:
    """
    Compares the DP policy trajectory against a CasADi/IPOPT continuous-time
    optimal trajectory on the same 7-panel figure.
    """
    try:
        from casadi_stall_optimizer import CasadiStallOptimizer
    except ImportError:
        logger.error(
            "casadi_stall_optimizer.py not found — skipping CasADi."
        )
        return

    k_thrust = pi.env.airplane.THROTTLE_LINEAR_MAPPING

    gamma_0, v_norm_0, alpha_0, q_0 = 0.0, 0.95, 20.0, 0.0

    # 1. Run DP simulation
    logger.info("[*] Running DP simulation for CasADi seed...")
    hist = run_dp_simulation(pi, gamma_0, v_norm_0, alpha_0, q_0)
    dp_T = hist["t"][-1]
    logger.info(f"[+] DP: T={dp_T:.2f}s, Δh={hist['h'][-1]:.2f}m")

    # 2. Build warm-start seed resampled to CasADi node count
    n_nodes = 100
    t_orig = np.array(hist["t"])
    t_states = np.linspace(0, dp_T, n_nodes + 1)
    t_ctrls = np.linspace(0, dp_T, n_nodes)

    dp_seed = {
        "T": dp_T,
        "gamma": np.interp(t_states, t_orig, hist["gamma"]),
        "v_norm": np.interp(t_states, t_orig, hist["v_norm"]),
        "alpha": np.interp(t_states, t_orig, hist["alpha"]),
        "q": np.interp(t_states, t_orig, hist["q"]),
        "h": np.interp(t_states, t_orig, hist["h"]),
        "de": np.interp(t_ctrls, t_orig[:-1], hist["de"][:-1]),
        "dt_ctrl": np.interp(
            t_ctrls, t_orig[:-1], hist["dt_ctrl"][:-1]
        ),
    }

    # 3. Solve CasADi OCP
    optimizer = CasadiStallOptimizer(k_thrust=k_thrust)
    cas = optimizer.solve_trajectory(
        gamma_0=np.deg2rad(gamma_0),
        v_norm_0=v_norm_0,
        alpha_0=np.deg2rad(alpha_0),
        q_0=np.deg2rad(q_0),
        n_nodes=n_nodes,
        dp_seed=dp_seed,
    )

    # 4. Plot comparison
    _plot_dp_vs_casadi(hist, cas, prefix)


def _plot_dp_vs_casadi(
    hist: dict, cas: dict, prefix: str
) -> None:
    """Renders the 7-panel DP vs CasADi comparison figure."""
    dp_t = np.array(hist["t"])
    dp_h = hist["h"]

    status = "Converged" if cas["converged"] else "Infeasible (debug)"

    COLOR_DP = "#532C8A"
    COLOR_CAS = "#E8742A"
    LW = 2.0

    panels = [
        ("gamma", r"$\gamma$ (deg)", True),
        ("v_norm", r"$V/V_s$", False),
        ("alpha", r"$\alpha$ (deg)", True),
        ("q", r"$q$ (deg/s)", True),
        ("de", r"$\delta_e$ (deg)", True),
        ("dt_ctrl", r"$\delta_t$", False),
        ("h", "Altitude Loss (m)", False),
    ]

    fig, axs = plt.subplots(len(panels), 1, figsize=(9, 17), sharex=False)
    fig.suptitle(
        f"DP vs CasADi/IPOPT — Stall Recovery\n"
        f"DP: Δh={dp_h[-1]:.1f}m, T={dp_t[-1]:.2f}s   |   "
        f"CasADi ({status}): "
        f"Δh={cas['h'][-1]:.1f}m, T={cas['T']:.2f}s",
        fontsize=12,
    )

    for ax, (key, ylabel, to_deg) in zip(axs, panels):
        dp_data = np.rad2deg(hist[key]) if to_deg else hist[key]
        ax.plot(dp_t, dp_data, color=COLOR_DP, lw=LW, label="DP Policy")
        ax.plot(
            cas["t"], cas[key],
            color=COLOR_CAS, lw=LW, ls="--", label="CasADi NLP",
        )
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="-", alpha=0.35)
        ax.legend(loc="best", fontsize=8)

    axs[0].axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    axs[5].set_ylim([-0.05, 1.05])
    axs[-1].set_xlabel("Time (s)")

    plt.tight_layout()
    out_path = RESULTS_DIR / f"{prefix}_DP_vs_CasADi.png"
    out_path.parent.mkdir(exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    logger.info(f"[+] Comparison plot saved to {out_path.resolve()}")


# ── Entry Point ──────────────────────────────────────────────────────────


def run(args) -> None:
    """Entered from `symmetric_stall.cli`, which has already set the model env."""
    runconfig.warn_if_code_default(logger)

    prefix = "symmetric_stall"
    env, states, actions, config = setup_symmetric_stall_experiment(args.grid)
    if args.mca:
        config.use_mca_timestep = True
        prefix = "symmetric_stall_mca"
    warm_start_path = Path(args.warm_start_ppo) if args.warm_start_ppo else None
    pi = train_or_load_policy(
        env, states, actions, config, prefix,
        warm_start_ppo=warm_start_path,
        grid=args.grid,
        policy_path=Path(args.out) if args.out else None,
    )

    if args.compare:
        logger.info("=" * 60)
        logger.info("[*] APPROACH A — convex action blend")
        logger.info("=" * 60)
        hist_a = run_dp_simulation(
            pi, 0.0, 0.95, 20.0, 0.0,
            greedy=False, actuator_tau=args.actuator_tau,
        )
        h_a, t_a = hist_a["h"][-1], hist_a["t"][-1]

        logger.info("=" * 60)
        logger.info("[*] APPROACH B — re-derived discrete greedy")
        logger.info("=" * 60)
        hist_b = run_dp_simulation(
            pi, 0.0, 0.95, 20.0, 0.0,
            greedy=True, actuator_tau=args.actuator_tau,
        )
        h_b, t_b = hist_b["h"][-1], hist_b["t"][-1]

        logger.info("=" * 60)
        if args.actuator_tau > 0:
            logger.info(f"[!] EMA filter τ = {args.actuator_tau:.3f}s applied to both A and B")
        logger.info(f"[A] Δh = {h_a:7.3f} m  |  t_rec = {t_a:.2f} s")
        logger.info(f"[B] Δh = {h_b:7.3f} m  |  t_rec = {t_b:.2f} s")
        logger.info(f"[Δ] B − A = {h_b - h_a:+.3f} m  ({(h_b - h_a) / abs(h_a) * 100:+.2f}%)")
        logger.info("=" * 60)

        tau_suffix = f"_tau{args.actuator_tau:g}" if args.actuator_tau > 0 else ""
        plot_time_response(hist_b, f"{prefix}_greedyB{tau_suffix}")
        plot_time_response(hist_a, f"{prefix}_blendA{tau_suffix}")
    else:
        hist = run_dp_simulation(
            pi, 0.0, 0.95, 20.0, 0.0,
            greedy=args.greedy_eval, actuator_tau=args.actuator_tau,
        )
        suffix = "_greedyB" if args.greedy_eval else ""
        if args.actuator_tau > 0:
            suffix += f"_tau{args.actuator_tau:g}"
        plot_time_response(hist, f"{prefix}{suffix}")

    plot_heatmaps(pi, prefix)
    # validate_with_casadi(pi, prefix)


if __name__ == "__main__":  # pragma: no cover
    from symmetric_stall.cli import main

    main()
