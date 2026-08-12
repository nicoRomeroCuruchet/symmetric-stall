import numpy as np
from gymnasium import spaces

from symmetric_stall.aircraft.airplane_env import AirplaneEnv
from symmetric_stall.aircraft.symmetric_full_grumman import SymmetricFullGrumman

# Crash thresholds. Same names and values as aircraft/spin.py on the 8-DOF
# branch, which is where they were debugged: there each threshold coincides
# EXACTLY with a grid edge, so that the state box is absorbing and every state
# reaches a terminal. Riley tabulates up to alpha = 40 deg, so the +40 ceiling
# is backed by the data.
ALPHA_CRASH = np.deg2rad(40.0)
GAMMA_CRASH = -np.pi + 0.05
CRASH_EPS = 1e-6


class SymmetricStall(AirplaneEnv):
    """
    Pure Markovian Environment for Symmetric Stall Recovery.
    Matches the exact physics and rewards of the PPO literature.
    """
    def __init__(self, render_mode=None):
        self.airplane = SymmetricFullGrumman()
        super().__init__(self.airplane)
        self.action_space = spaces.Box(
            np.array([np.deg2rad(-25), 0.0], np.float32),
            np.array([np.deg2rad(15),  1.0], np.float32),
            shape=(2,), dtype=np.float32
        )

    def _get_obs(self):
        """Standard 1D observation array for Gym compliance."""
        return np.array([
            self.airplane.flight_path_angle,
            self.airplane.airspeed_norm,
            self.airplane.alpha,
            self.airplane.pitch_rate
        ], dtype=np.float32)

    def _get_info(self):
        return {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Reset distribution focused on the stall-recovery startup region: high α
        # (near or above stall), γ near level, V near or below stall, q small.
        # The PI grid (observation space) is still wider; the policy can explore
        # beyond this region during an episode if the dynamics carry it there.
        min_spawn_state = [np.deg2rad(-15), 0.90, np.deg2rad(14),  np.deg2rad(-15)]
        max_spawn_state = [np.deg2rad(  5), 1.05, np.deg2rad(20),  np.deg2rad( 15)]

        flight_path_angle, airspeed_norm, alpha, pitch_rate = self.np_random.uniform(
            min_spawn_state, max_spawn_state
        )
        self.airplane.reset(flight_path_angle, airspeed_norm, alpha, pitch_rate)

        return self._get_obs(), self._get_info()

    def specific_reset(self, flight_path_angle, airspeed_norm, alpha, pitch_rate):
        """Forces the environment into a specific mathematical state."""
        self.airplane.reset(flight_path_angle, airspeed_norm, alpha, pitch_rate)
        return self._get_obs(), self._get_info()

    def step(self, action: list):
        """
        Pure step function: No history, no hidden filters.
        Strict compliance with the Markov property.
        """
        elevator = action[0]
        delta_throttle = action[1]

        # 1. Integrate physical kinematics
        self.airplane.command_airplane(elevator, delta_throttle)

        # 2. Retrieve new physical state
        fpa = self.airplane.flight_path_angle
        v_norm = self.airplane.airspeed_norm
        alpha = self.airplane.alpha

        # 3. Base Physical Reward: True physical height loss in meters
        reward = (
            self.airplane.TIME_STEP * v_norm * np.sin(fpa) * self.airplane.STALL_AIRSPEED
        )

        # 4. Evaluate specific terminal conditions
        fpa_success = (fpa >= 0.0)

        alpha_crash = (alpha >= np.deg2rad(40)) or (alpha <= np.deg2rad(-40))
        fpa_crash = (fpa <= -np.pi + 0.05)

        failure = fpa_crash or alpha_crash
        terminated = fpa_success or failure

        # Apply catastrophic penalty only if boundaries are violated
        if failure:
            reward = -1000.0 * self.airplane.STALL_AIRSPEED

        return self._get_obs(), reward, terminated, False, self._get_info()

    def terminal(self, states: np.ndarray):
        """
        Vectorized terminal check exclusively for GPU Policy Iteration initialization.
        Expects a 2D array of states (N, 4).
        """
        fpa = np.asarray(states[:, 0])
        alpha = np.asarray(states[:, 2])

        fpa_success = (fpa >= 0.0)

        # CRASH_EPS: the comparison is between the grid edge, which is
        # float32, and the threshold, which is float64. The alpha = -40 deg
        # plane lands 0.34 ULP ABOVE its threshold, so without a tolerance the
        # `<=` is false and the cell is not marked terminal. With no failure
        # terminal at all the value diverges under gamma = 1 and evaluation
        # cannot converge: it sticks at the float32 ULP (Delta = 2.0 at
        # |V| ~ 2e7). Same tolerance already used on the 8-DOF branch
        # (aircraft/spin.py:37).
        #
        # NOTE: this only bites on the paper1 grid, where -40 deg IS an edge.
        # On the riley grid alpha bottoms out at -10, so the lower crash plane
        # lies outside the box and never fires.
        alpha_crash = ((alpha >= ALPHA_CRASH - CRASH_EPS)
                       | (alpha <= -ALPHA_CRASH + CRASH_EPS))
        fpa_crash = (fpa <= GAMMA_CRASH + CRASH_EPS)

        failure = fpa_crash | alpha_crash
        terminate = fpa_success | failure

        rewards = np.zeros_like(fpa)

        # FIX: Align GPU initialization penalty with the step function
        rewards[failure] = -1000.0 * self.airplane.STALL_AIRSPEED

        return terminate, rewards
