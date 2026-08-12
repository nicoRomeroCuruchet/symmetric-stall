import numba as nb
import numpy as np


@nb.njit(cache=True)
def barycentric_interp_value(
    state: np.ndarray,
    V: np.ndarray,
    bounds_low: np.ndarray,
    bounds_high: np.ndarray,
    grid_shape: np.ndarray,
    strides: np.ndarray,
    corner_bits: np.ndarray,
) -> float:
    """
    Multilinear interpolation of a scalar field V at one off-grid state.
    Mirrors `get_barycentric_4d` in PolicyIteration.py's CUDA kernel exactly:
    same clamping, same i = floor(n) with edge-case decrement, same 2^N
    tensor-product weights. Used by `get_optimal_action_greedy` to bootstrap
    Q-values without committing the convex-action-averaging approximation.
    """
    n_dims = state.shape[0]
    n_corners = corner_bits.shape[0]
    step_sizes = (bounds_high - bounds_low) / (grid_shape - 1)

    base_idx = np.zeros(n_dims, dtype=np.int32)
    t = np.zeros(n_dims, dtype=np.float32)

    for d in range(n_dims):
        p = max(bounds_low[d], min(state[d], bounds_high[d]))
        cell = (p - bounds_low[d]) / step_sizes[d]
        idx_d = int(cell)
        if idx_d >= grid_shape[d] - 1:
            idx_d = grid_shape[d] - 2
        base_idx[d] = idx_d
        t[d] = (p - (bounds_low[d] + idx_d * step_sizes[d])) / step_sizes[d]

    v_interp = 0.0
    for c in range(n_corners):
        w = 1.0
        flat_idx = 0
        for d in range(n_dims):
            bit = corner_bits[c, d]
            w *= t[d] if bit else (1.0 - t[d])
            flat_idx += (base_idx[d] + bit) * strides[d]
        v_interp += w * V[flat_idx]

    return v_interp


@nb.njit(cache=True)
def get_barycentric_weights_and_indices(
    points: np.ndarray,
    bounds_low: np.ndarray,
    bounds_high: np.ndarray,
    grid_shape: np.ndarray,
    strides: np.ndarray,
    corner_bits: np.ndarray,
) -> tuple:
    """
    Calculate O(1) Barycentric interpolation weights and flat indices.

    Bypasses expensive algorithmic searches in favor of direct mathematical
    mapping for N-dimensional regular grids.
    """
    n_points, n_dims = points.shape
    n_corners = corner_bits.shape[0]

    weights = np.zeros((n_points, n_corners), dtype=np.float32)
    indices = np.zeros((n_points, n_corners), dtype=np.int32)

    step_sizes = (bounds_high - bounds_low) / (grid_shape - 1)

    for i in range(n_points):
        base_idx = np.zeros(n_dims, dtype=np.int32)
        t = np.zeros(n_dims, dtype=np.float32)

        for d in range(n_dims):
            # Clip points to bounds to avoid out-of-bounds indexing
            p = max(bounds_low[d], min(points[i, d], bounds_high[d]))

            # O(1) grid cell calculation
            cell = (p - bounds_low[d]) / step_sizes[d]
            idx_d = int(cell)

            # Handle edge case at the upper boundary
            if idx_d >= grid_shape[d] - 1:
                idx_d = grid_shape[d] - 2

            base_idx[d] = idx_d
            t[d] = (p - (bounds_low[d] + idx_d * step_sizes[d])) / step_sizes[d]

        # Compute weights and flat indices for all 2^N corners
        for c in range(n_corners):
            w = 1.0
            flat_idx = 0
            for d in range(n_dims):
                bit = corner_bits[c, d]
                w *= t[d] if bit else (1.0 - t[d])
                flat_idx += (base_idx[d] + bit) * strides[d]

            weights[i, c] = w
            indices[i, c] = flat_idx

    return weights, indices


def get_optimal_action(
    state: np.ndarray,
    optimal_policy: any,
    current_action_idx: int = None,
    hysteresis: float = 0.1,
) -> tuple:
    """
    Approximate the optimal action for a given state using barycentric interpolation.

    Computes a convex combination of the actions voted by the 16 surrounding vertices,
    weighted by their barycentric weights. This produces a continuous action that
    eliminates discontinuities at grid cell boundaries.

    Returns:
        (optimal_action, probabilities, None)
    """
    state_2d = np.atleast_2d(state).astype(np.float32)

    lambdas, points_indexes = get_barycentric_weights_and_indices(
        state_2d,
        optimal_policy.bounds_low,
        optimal_policy.bounds_high,
        optimal_policy.grid_shape,
        optimal_policy.strides,
        optimal_policy.corner_bits,
    )

    lambdas = lambdas.flatten()
    flat_indices = points_indexes.flatten()

    best_action_indices = optimal_policy.policy[flat_indices]

    n_neighbors = len(flat_indices)
    relevant_policies = np.zeros(
        (n_neighbors, optimal_policy.n_actions), dtype=np.float32
    )
    relevant_policies[np.arange(n_neighbors), best_action_indices] = 1.0

    probabilities = lambdas @ relevant_policies

    if not np.isclose(np.sum(probabilities), 1.0, atol=1e-2):
        raise ValueError(
            f"Interpolated probabilities do not sum to 1. "
            f"Sum: {np.sum(probabilities)}"
        )

    # Convex combination: weighted average of all actions by their probability mass
    optimal_action = probabilities @ optimal_policy.action_space

    return optimal_action, probabilities, None


def get_optimal_action_greedy(
    state: np.ndarray,
    optimal_policy: any,
) -> tuple:
    """
    Approach B: re-derive PI's greedy action at an off-grid state by
    evaluating Q(s, a) for every discrete action in `pi.action_space` and
    returning the argmax. This mirrors `policy_improve_kernel` exactly:

        next_state = RK4(state, a)
        V_next     = barycentric_interp(V, next_state)
        Q(s, a)    = reward(s, a) + gamma * V_next

    Unlike `get_optimal_action` (Approach A — convex blend of corner
    actions), the returned action is always one of the discrete candidates
    PI considered, so bang-bang structure is preserved.

    NOTE: the per-step reward below mirrors the CUDA kernel with only
    `w_crash_penalty = 1000.0` active (matches the current `main.py`
    config — apples-to-apples vs PPO since env.step() also applies the
    -1000·v_stall failure penalty). All other shaping weights are zero.
    If you change those weights when re-training, update this reward
    block to match the kernel.

    Returns: (optimal_action, None, best_action_idx)
    """
    env = optimal_policy.env
    airplane = env.airplane
    dt = airplane.TIME_STEP
    v_stall = airplane.STALL_AIRSPEED
    gamma_discount = 1.0  # matches PolicyIterationStallConfig.gamma = 1.0

    saved = (
        airplane.flight_path_angle, airplane.airspeed_norm,
        airplane.alpha, airplane.pitch_rate,
    )

    n_actions = len(optimal_policy.action_space)
    best_q = -np.inf
    best_a_idx = 0
    next_state_buf = np.empty(4, dtype=np.float32)

    for a_idx in range(n_actions):
        de = float(optimal_policy.action_space[a_idx, 0])
        dt_ctrl = float(optimal_policy.action_space[a_idx, 1])

        airplane.flight_path_angle = float(state[0])
        airplane.airspeed_norm = float(state[1])
        airplane.alpha = float(state[2])
        airplane.pitch_rate = float(state[3])

        airplane.command_airplane(de, dt_ctrl)

        new_g = airplane.flight_path_angle
        new_v = airplane.airspeed_norm
        new_a = airplane.alpha
        new_q = airplane.pitch_rate

        # Pure altitude-loss reward + crash penalty matching CUDA kernel.
        reward = dt * new_v * v_stall * np.sin(new_g)
        if (
            abs(new_a) >= 0.698132   # |α| >= 40°
            or new_g <= -np.pi + 0.05
        ):
            reward -= 1000.0 * v_stall

        next_state_buf[0] = new_g
        next_state_buf[1] = new_v
        next_state_buf[2] = new_a
        next_state_buf[3] = new_q

        v_next = barycentric_interp_value(
            next_state_buf,
            optimal_policy.value_function,
            optimal_policy.bounds_low,
            optimal_policy.bounds_high,
            optimal_policy.grid_shape,
            optimal_policy.strides,
            optimal_policy.corner_bits,
        )

        q_val = reward + gamma_discount * v_next
        if q_val > best_q:
            best_q = q_val
            best_a_idx = a_idx

    (
        airplane.flight_path_angle, airplane.airspeed_norm,
        airplane.alpha, airplane.pitch_rate,
    ) = saved

    optimal_action = np.array(
        optimal_policy.action_space[best_a_idx], dtype=np.float32,
    )
    return optimal_action, None, best_a_idx
