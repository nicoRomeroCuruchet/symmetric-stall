import json
import os
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path

import gymnasium as gym
import numpy as np
from loguru import logger

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    cp = np
    GPU_AVAILABLE = False


# Fused Reduction Kernel: Computes max(abs(A - B)) with 0 bytes of auxiliary VRAM allocation
max_abs_diff_kernel = cp.ReductionKernel(
    in_params='float32 x, float32 y',
    out_params='float32 z',
    map_expr='abs(x - y)',
    reduce_expr='max(a, b)',
    post_map_expr='z = a',
    identity='0.0f',
    name='max_abs_diff_stall'
)


@dataclass
class PolicyIterationStallConfig:
    """
    Configuration parameters for the 4-DOF Symmetric Stall Policy Iteration.
    """
    maximum_iterations: int = 20_000
    gamma: float = 1.0
    theta: float = 1e-4
    n_steps: int = 100
    log: bool = False
    log_interval: int = 150
    img_path: Path = field(default_factory=lambda: Path("./img"))

    # --- Reward Shaping Weights (injected into CUDA kernel at compile time) ---
    w_q_penalty: float = 2.0           # Pitch damping: penalizes q² per timestep
    w_alpha_barrier_pos: float = 100.0  # Alpha barrier above positive stall
    w_alpha_barrier_neg: float = 10.0   # Alpha barrier below negative stall
    w_crash_penalty: float = 1000.0     # Crash penalty multiplier (×V_stall)
    w_control_effort: float = 1.0       # Control effort: penalizes δe²
    w_throttle_bonus: float = 0.2       # Throttle incentive bonus

    # --- CG sensitivity (injected into CUDA kernel at compile time) ---
    # CG shift aft of the Riley reference (0.25 c̄), as a fraction of chord:
    # Cm_cg = Cm_ref + CL · dxcg_over_chord. 0.0 reproduces the nominal model.
    dxcg_over_chord: float = 0.0

    # --- Timestep scheme (ported from the 3DOF branch dt-ablation) ---
    # use_mca_timestep=False + dt_fixed=0.01 + n_micro=10 reproduces the
    # original hardcoded behavior (macro dt = TIME_STEP = 0.01 s, 10 RK4
    # micro-steps of 1 ms). use_mca_timestep=True switches the macro step to
    # the MCA local-consistency interval dt_h(x) = 1/(Σ_i |f_i|/h_i) clamped
    # to dt_max, evaluated at the current policy's action and SHARED across
    # candidate actions in the improvement argmax (a per-action dt biases the
    # Q-comparison — see mca_per_action, the control arm of the A/B study).
    # 3DOF precedent: MCA failed to converge at fine grids under gamma=1.
    use_mca_timestep: bool = False
    dt_fixed: float = 0.01
    dt_max: float = 0.05
    n_micro: int = 10
    mca_per_action: bool = False

    # CG position measured FROM Riley's reference (0.25 c on the centreline),
    # in METRES: aft = towards the tail, right = towards the right wing,
    # below = downwards. All zero reproduces the nominal model.
    # They default to the environment, same as the plant (aircraft/grumman.py):
    # if the kernel were always compiled with Riley's CG while the plant read
    # it from the environment, a CPU rollout and the GPU policy would be two
    # different aircraft. Setting them explicitly in the config still wins.
    cg_aft_m: float = field(
        default_factory=lambda: float(os.environ.get("CG_AFT_M", 0.0)))
    cg_right_m: float = field(
        default_factory=lambda: float(os.environ.get("CG_RIGHT_M", 0.0)))
    cg_below_m: float = field(
        default_factory=lambda: float(os.environ.get("CG_BELOW_M", 0.0)))


class PolicyIterationStall:
    """
    High-performance Procedural Policy Iteration for 4-DOF Symmetric Stall Recovery.

    State space: (γ, V/Vs, α, q) — flight path angle, normalized airspeed,
                                     angle of attack, pitch rate.
    Action space: (δe, δt) — elevator deflection, throttle.

    Embeds the full Grumman AA-1 Yankee aerodynamic model (CL, CD, Cm with stall
    saturation) and 4D Barycentric interpolation directly into CUDA C++ kernels.
    """

    def __init__(
        self,
        env: gym.Env,
        states_space: np.ndarray,
        action_space: np.ndarray,
        config: PolicyIterationStallConfig,
    ) -> None:

        if not GPU_AVAILABLE:
            raise RuntimeError("CuPy is required for procedural on-the-fly CUDA kernels.")

        self.env = env
        self.states_space = np.ascontiguousarray(states_space, dtype=np.float32)
        self.action_space = np.ascontiguousarray(action_space, dtype=np.float32)
        self.config = config

        self.n_states, self.n_dims = self.states_space.shape
        self.n_actions = len(self.action_space)
        self.n_corners = 2**self.n_dims  # 16 for 4D

        # Aerodynamic constants from Grumman base class
        airplane = env.airplane
        self.v_stall = airplane.STALL_AIRSPEED
        self.k_thrust = airplane.THROTTLE_LINEAR_MAPPING
        self.dt = airplane.TIME_STEP

        self._precompute_grid_metadata()
        self._allocate_tensors_and_compile()

    def _precompute_grid_metadata(self) -> None:
        """Extract bounds, shape, and strides for the 4D CUDA interpolation."""
        self.bounds_low = np.min(self.states_space, axis=0).astype(np.float32)
        self.bounds_high = np.max(self.states_space, axis=0).astype(np.float32)

        self.grid_shape = np.array(
            [len(np.unique(self.states_space[:, d])) for d in range(self.n_dims)],
            dtype=np.int32,
        )

        self.strides = np.zeros(self.n_dims, dtype=np.int32)
        stride = 1
        for d in range(self.n_dims - 1, -1, -1):
            self.strides[d] = stride
            stride *= self.grid_shape[d]

        self.corner_bits = np.array(
            list(product([0, 1], repeat=self.n_dims)), dtype=np.int32
        )
        logger.info(f"Grid precomputed. Shape: {self.grid_shape}, Strides: {self.strides}")

    def _allocate_tensors_and_compile(self) -> None:
        """Allocates minimal required memory and compiles the 4-DOF CUDA Kernels."""
        logger.info("Allocating procedural tensors and compiling 4-DOF CUDA JIT Kernels...")

        # Push constants to VRAM
        self.d_states = cp.asarray(self.states_space, dtype=cp.float32)
        self.d_actions = cp.asarray(self.action_space, dtype=cp.float32)
        self.d_bounds_low = cp.asarray(self.bounds_low, dtype=cp.float32)
        self.d_bounds_high = cp.asarray(self.bounds_high, dtype=cp.float32)
        self.d_grid_shape = cp.asarray(self.grid_shape, dtype=cp.int32)
        self.d_strides = cp.asarray(self.strides, dtype=cp.int32)

        # 1D Policy mapping state indices to the best action index
        self.d_policy = cp.zeros(self.n_states, dtype=cp.int32)

        self.d_value_function = cp.zeros(self.n_states, dtype=cp.float32)
        self.d_new_value_function = cp.zeros(self.n_states, dtype=cp.float32)

        # Compute terminal states in Python, push mask to GPU
        terminal_mask, terminal_rewards = self.env.terminal(self.states_space)
        self.d_terminal_mask = cp.asarray(terminal_mask, dtype=cp.bool_)

        if np.isscalar(terminal_rewards):
            self.d_value_function[self.d_terminal_mask] = terminal_rewards
        else:
            self.d_value_function[self.d_terminal_mask] = cp.asarray(
                terminal_rewards[terminal_mask], dtype=cp.float32
            )

        self.d_new_value_function[:] = self.d_value_function[:]

        # Compile the RawModule
        self._compile_cuda_module()
        logger.success("4-DOF CUDA Kernels compiled. VRAM usage optimized.")

    def _compile_cuda_module(self) -> None:
        """
        Compiles the 4-DOF aerodynamics and Bellman operations into CUDA.
        Implements a Markovian Alpha Barrier to enforce proper stall recovery physics.
        """
        reward_defines = f'''
        #define W_Q_PENALTY {self.config.w_q_penalty:.6f}f
        #define W_ALPHA_BARRIER_POS {self.config.w_alpha_barrier_pos:.6f}f
        #define W_ALPHA_BARRIER_NEG {self.config.w_alpha_barrier_neg:.6f}f
        #define THRUST_RILEY {1 if os.environ.get('THRUST_MODEL','paper1').lower()=='riley' else 0}
        #define W_CRASH_PENALTY {self.config.w_crash_penalty:.6f}f
        #define W_CONTROL_EFFORT {self.config.w_control_effort:.6f}f
        #define W_THROTTLE_BONUS {self.config.w_throttle_bonus:.6f}f
        #define DXCG_OVER_CHORD {self.config.dxcg_over_chord:.6f}f
        #define CG_AFT_M {self.config.cg_aft_m + self.config.dxcg_over_chord * 1.22!r}f
        #define CG_RIGHT_M {self.config.cg_right_m!r}f
        #define CG_BELOW_M {self.config.cg_below_m!r}f
        '''
        cuda_source = reward_defines + r'''
        extern "C" {

        __device__ const float MASS = 715.3152f; // Riley Table I: 1577 lb x 0.45359237
        __device__ const float S = 9.114717f;   // Table I: 98.11 ft2
        __device__ const float CHORD = 1.2192f; // Table I: 4.00 ft x 0.3048
        __device__ const float RHO = 1.225f;
        __device__ const float G = 9.81f;
        __device__ const float I_YY = 1000.5936f; // Riley Table I: 738 slug-ft2 x 1.35581795

        __device__ const float SPAN = 7.455408f;  // 24.46 ft x 0.3048. Riley's Table I says 26.46 on the wing row and 24.46 under overall dimensions; his own AR = 6.10 and the 98.11 ft2 area both give 24.46, and so does the AA-1X flight data.

        /* Transfer of moments to the CG. Riley refers ALL of his moments to
           the CG on the fuselage centreline at 25% of the mean aerodynamic
           chord (report, p. 5, and Table I). With the CG elsewhere:

               M_CG = M_ref + (r_ref - r_CG) x F

           which must be applied to the BODY forces, not to L and D:

               C_N = C_L cos a + C_D sin a        C_A = C_D cos a - C_L sin a

           Expanded, with the CG measured FROM Riley's reference (dx towards
           the tail, dy towards the right wing, dz downwards, in metres):

               dC_l = (dy*C_N + dz*C_Y) / b
               dC_m = (dx*C_N + dz*C_A) / c
               dC_n = (dx*C_Y - dy*C_A) / b

           The sign checks itself: an aft CG (dx>0) with positive lift gives
           nose-up, i.e. less stable. Poliak's thesis (2025) p. 25 eqs. (12)
           and (16) writes the transfer with the opposite sign, which gives
           nose-down where nose-up belongs; the standard derivation is used
           here.

           Exact mirror of Grumman._delta_momentos_cg in aircraft/grumman.py. */
        __device__ void delta_momentos_cg(float cl, float cd, float cy,
                                          float alpha, float *dcl_roll,
                                          float *dcm, float *dcn) {
            if (CG_AFT_M == 0.0f && CG_RIGHT_M == 0.0f && CG_BELOW_M == 0.0f) {
                *dcl_roll = 0.0f; *dcm = 0.0f; *dcn = 0.0f; return;
            }
            float ca_ = cosf(alpha), sa_ = sinf(alpha);
            float c_n = cl * ca_ + cd * sa_;
            float c_a = cd * ca_ - cl * sa_;
            *dcl_roll = (CG_RIGHT_M * c_n + CG_BELOW_M * cy) / SPAN;
            *dcm      = (CG_AFT_M   * c_n + CG_BELOW_M * c_a) / CHORD;
            *dcn      = (CG_AFT_M   * cy  - CG_RIGHT_M * c_a) / SPAN;
        }


        // Riley (1985) NASA-TM-86309 Table III - alpha breakpoints (radians)
        __device__ const float CL_A_TBL[14] = {
            -0.17453292f,-0.0872664601f,0.0f,0.0872664601f,0.17453292f,0.209439516f,0.244346097f,0.279252678f,0.314159274f,0.34906584f,0.436332315f,0.52359879f,0.610865235f,0.69813168f
        };

        __device__ float cl_interp(float alpha, const float* tbl) {
            if (alpha <= CL_A_TBL[0])  return tbl[0];
            if (alpha >= CL_A_TBL[13]) return tbl[13];
            for (int i = 0; i < 13; ++i) {
                if (alpha <= CL_A_TBL[i+1]) {
                    float t = (alpha - CL_A_TBL[i]) / (CL_A_TBL[i+1] - CL_A_TBL[i]);
                    return tbl[i] + t * (tbl[i+1] - tbl[i]);
                }
            }
            return tbl[13];
        }

        __device__ float bilinear_interp(float alpha, float ct,
                                          const float* tbl0, const float* tbl05) {
            float t = fminf(fmaxf(ct / 0.5f, 0.0f), 1.0f);
            return cl_interp(alpha, tbl0) + t * (cl_interp(alpha, tbl05) - cl_interp(alpha, tbl0));
        }

        /* Riley (1985) App. A, Eq. (A9): T_sl = T0 + T1*V, tabulated against
           the intermediate throttle of Eq. (A3). lbf and ft/s. */
        __device__ const float THR_DTP[6] = { 0.0f,0.2f,0.4f,0.6f,0.8f,1.0f };
        __device__ const float THR_T0[6]  = { -237.0f,-100.0f,40.0f,182.0f,314.0f,452.0f };
        __device__ const float THR_T1[6]  = { 0.100f,-0.060f,-0.280f,-0.510f,-0.675f,-0.820f };

        __device__ float compute_ct(float throttle, float vt, float k_thrust) {
            float v = fmaxf(vt, 0.1f);
            float q_bar = 0.5f * RHO * v * v;
            float thrust;
#if THRUST_RILEY
            /* Mirrors aircraft/grumman.py::_compute_ct with THRUST_MODEL=riley.
               Both must move together or the kernel trains one aeroplane while
               the CPU model evaluates another. */
            float dtp = 0.65f * throttle + 0.35f;
            float x = fminf(fmaxf(dtp, 0.0f), 1.0f) * 5.0f;
            int i = min((int)x, 4);
            float f = x - (float)i;
            float t0 = THR_T0[i] + f * (THR_T0[i+1] - THR_T0[i]);
            float t1 = THR_T1[i] + f * (THR_T1[i+1] - THR_T1[i]);
            thrust = (t0 + t1 * v * 3.280839895f) * 4.4482216f;
#else
            thrust = k_thrust * throttle;
#endif
            /* (A11) verbatim, unclipped: the table interpolation is what
               saturates, and delta_cd_thrust absorbs the excess. */
            return thrust / (q_bar * S);
        }

        /* Riley (1985) App. B: dCD_T = 0 inside [0, 0.5], -0.80(CT-0.5)cos(a)
           above it and -0.80*CT*cos(a) below. Identically zero in the
           tabulated range, so it touches nothing already published. */
        __device__ float delta_cd_thrust(float ct_raw, float alpha) {
            float excess = ct_raw > 0.5f ? (ct_raw - 0.5f)
                                          : (ct_raw < 0.0f ? ct_raw : 0.0f);
            return -0.80f * excess * cosf(alpha);
        }

        // --- CT=0 tables (Riley 1985, Table III, power-off) ---

        // CL_o: flat-top 14-18 deg (max=1.26), gradual post-stall drop
        __device__ const float CL_O_TBL_CT0[14] = {
            -0.41f, -0.01f,  0.41f,  0.84f,  1.16f,
             1.23f,  1.26f,  1.26f,  1.26f,  1.25f,
             1.22f,  1.17f,  1.13f,  1.08f
        };

        // CL_q: pitch damping, CT=0
        __device__ const float CL_Q_TBL_CT0[14] = {
             2.41f,  2.41f,  2.42f,  2.46f,  2.59f,
             2.96f,  3.72f,  4.73f,  5.29f,  5.16f,
             5.05f,  5.06f,  5.08f,  5.08f
        };

        // CM_q: pitch damping, CT=0
        __device__ const float CM_Q_TBL_CT0[14] = {
             -7.0000f,  -7.0000f,  -7.0400f,  -7.1500f,  -7.5200f,
             -8.6200f, -10.8000f, -13.7300f, -15.3800f, -15.0000f,
            -14.6600f, -14.7100f, -14.7700f, -14.7700f
        };

        // CM_o: nonlinear nose-down moment, CT=0
        __device__ const float CM_O_TBL_CT0[14] = {
             0.2700f,  0.1580f,  0.0760f,  0.0020f, -0.0800f,
            -0.1180f, -0.1670f, -0.2250f, -0.2770f, -0.3160f,
            -0.4080f, -0.4800f, -0.5560f, -0.6060f
        };

        // CD_o: strong post-stall rise, CT=0
        /* Riley Table III(b): the two elevator drag terms, per radian. */
        __device__ const float CD_DE_TBL_CT0[14] = {
            -0.0790682063f,-0.0504203029f,-0.0217724033f,0.00687549589f,0.035523396f,0.0469825566f,0.0584417172f,0.0699008778f,0.0813600346f,0.092819199f,0.121467099f,0.150114998f,0.178762898f,0.207410797f
        };
        __device__ const float CD_DE_TBL_CT05[14] = {
            -0.14782317f,-0.0847977847f,-0.0217724033f,0.0412529744f,0.104278356f,0.129488513f,0.154698655f,0.179908812f,0.205118969f,0.230329111f,0.293354481f,0.356379867f,0.419405252f,0.482430637f
        };
        __device__ const float CD_DE2_TBL_CT0[14] = {
            0.0984842628f,0.0984842628f,0.0984842628f,0.0984842628f,0.0984842628f,0.0984842628f,0.0984842628f,0.0984842628f,0.0984842628f,0.0984842628f,0.0984842628f,0.0984842628f,0.0984842628f,0.0984842628f
        };
        __device__ const float CD_DE2_TBL_CT05[14] = {
            0.0f,0.0984842628f,0.196968526f,0.279038727f,0.344694912f,0.364391774f,0.377523005f,0.390654236f,0.393937051f,0.390654236f,0.331563681f,0.239645034f,0.121463925f,0.0f
        };
        __device__ const float CD_O_TBL_CT0[14] = {
            0.0666f, 0.0486f, 0.0526f, 0.0846f, 0.1456f,
            0.1856f, 0.2446f, 0.3136f, 0.3786f, 0.4486f,
            0.6186f, 0.7786f, 0.9255f, 1.0636f
        };

        // CL_de (elevator effectiveness), CT=0, /rad
        __device__ const float CL_DE_TBL_CT0[14] = {
            0.355233967f,0.360963523f,0.355233967f,0.332315624f,0.303667724f,0.292208582f,0.286478996f,0.28074941f,0.275019854f,0.269290239f,0.252101511f,0.240642354f,0.223453611f,0.211994454f
        };

        // CM_de (elevator effectiveness), CT=0, /rad
        __device__ const float CM_DE_TBL_CT0[14] = {
            -1.10580897f,-1.10580897f,-1.10580897f,-1.03132439f,-0.945380688f,-0.939651132f,-0.933921516f,-0.92819196f,-0.92819196f,-0.92819196f,-0.92819196f,-0.859436989f,-0.74484539f,-0.572957993f
        };

        // --- CT=0.5 tables (Riley 1985, Table III, power-on, thrust embedded) ---

        // CL_o at CT=0.5: max 1.72 at 18 deg
        __device__ const float CL_O_TBL_CT05[14] = {
            -0.67f, -0.14f,  0.41f,  0.97f,  1.42f,
             1.54f,  1.62f,  1.67f,  1.72f,  1.76f,
             1.85f,  1.92f,  1.99f,  2.05f
        };

        // CL_q at CT=0.5
        __device__ const float CL_Q_TBL_CT05[14] = {
             3.012f,  3.012f,  3.029f,  3.222f,  3.594f,
             4.351f,  6.072f,  6.382f,  6.988f,  6.833f,
             6.561f,  6.127f,  5.966f,  5.811f
        };

        // CD_o at CT=0.5: negative at low alpha (propulsor generates more forward force than drag)
        __device__ const float CD_O_TBL_CT05[14] = {
            -0.3273f, -0.3499f, -0.3474f, -0.3139f, -0.2483f,
            -0.2057f, -0.1435f, -0.0709f, -0.0018f,  0.0727f,
             0.2561f,  0.4322f,  0.5979f,  0.7572f
        };

        // CM_o at CT=0.5
        __device__ const float CM_O_TBL_CT05[14] = {
             0.2700f,  0.1580f,  0.0760f,  0.0020f, -0.0800f,
            -0.1180f, -0.1670f, -0.2250f, -0.2770f, -0.3160f,
            -0.4080f, -0.4800f, -0.5560f, -0.6060f
        };

        // CM_q at CT=0.5
        __device__ const float CM_Q_TBL_CT05[14] = {
             -8.75f,  -8.75f,  -8.80f,  -9.36f, -10.44f,
            -12.64f, -17.64f, -18.54f, -20.30f, -19.85f,
            -19.06f, -17.80f, -17.33f, -16.88f
        };

        // CL_de at CT=0.5, /rad
        __device__ const float CL_DE_TBL_CT05[14] = {
            0.796411574f,0.767763674f,0.750574946f,0.704738319f,0.624524236f,0.595876336f,0.578687549f,0.561498821f,0.538580477f,0.515662193f,0.458366424f,0.418259352f,0.349504381f,0.315126896f
        };

        // CM_de at CT=0.5, /rad
        __device__ const float CM_DE_TBL_CT05[14] = {
            -2.1428628f,-2.25172472f,-2.2574544f,-2.26318407f,-2.20015883f,-2.06264877f,-1.9136796f,-1.78189933f,-1.65011895f,-1.54125702f,-1.29488516f,-1.22040045f,-1.08862019f,-0.876625717f
        };

        // Derivatives with respect to alpha-dot (Riley Table III). They
        // multiply alpha_dot*c/(2V), already dimensionless, so they go WITHOUT
        // the degree-to-radian conversion of the control derivatives. They
        // change sign between 12 and 14 degrees: damping below the stall,
        // anti-damping above it.
        __device__ const float CL_ADOT_TBL_CT0[14] = {
            0.68900f, 0.68900f, 0.67500f, 0.63700f, 0.51000f,
            0.13100f, -0.62000f, -0.59600f, -0.13100f, 0.00000f,
            0.11700f, 0.10000f, 0.07900f, 0.07900f
        };
        __device__ const float CL_ADOT_TBL_CT05[14] = {
            0.86100f, 0.86100f, 0.84300f, 0.83300f, 0.70300f,
            0.19300f, -1.01200f, -0.80600f, -0.17200f, 0.00000f,
            0.15100f, 0.12100f, 0.09300f, 0.09300f
        };
        __device__ const float CM_ADOT_TBL_CT0[14] = {
            0.00000f, 0.00000f, -1.96000f, -1.85000f, -1.48000f,
            -0.38000f, 1.80000f, 1.73000f, 0.38000f, 0.00000f,
            -0.34000f, -0.29000f, -0.23000f, -0.23000f
        };
        __device__ const float CM_ADOT_TBL_CT05[14] = {
            -2.50000f, -2.50000f, -2.45000f, -2.42000f, -2.06000f,
            -0.56000f, 2.94000f, 2.34000f, 0.50000f, 0.00000f,
            -0.44000f, -0.35000f, -0.27000f, -0.27000f
        };

        // Continuous 4-DOF dynamics (gamma_dot, vn_dot, alpha_dot, q_dot).
        // Extracted verbatim from the former inline k1..k4 stages of
        // rk4_step_4dof (pure code motion - same float operation order).
        __device__ void derivs_4dof(
            float gamma, float vn, float alpha, float q,
            float de, float throttle, float v_stall, float k_thrust,
            float& d_g, float& d_v, float& d_a, float& d_q
        ) {
            float vt = fmaxf(vn * v_stall, 0.1f);
            float q_hat = q * CHORD / (2.0f * vt);
            float ct = compute_ct(throttle, vt, k_thrust);
            float cl_de = bilinear_interp(alpha, ct, CL_DE_TBL_CT0, CL_DE_TBL_CT05);
            float cm_de = bilinear_interp(alpha, ct, CM_DE_TBL_CT0, CM_DE_TBL_CT05);
            float cl = bilinear_interp(alpha, ct, CL_O_TBL_CT0, CL_O_TBL_CT05)
                       + cl_de * de
                       + bilinear_interp(alpha, ct, CL_Q_TBL_CT0, CL_Q_TBL_CT05) * q_hat;
            float cd = bilinear_interp(alpha, ct, CD_O_TBL_CT0, CD_O_TBL_CT05);
            cd += bilinear_interp(alpha, ct, CD_DE_TBL_CT0, CD_DE_TBL_CT05) * de;
            cd += bilinear_interp(alpha, ct, CD_DE2_TBL_CT0, CD_DE2_TBL_CT05)
                  * de * de;
            cd += delta_cd_thrust(ct, alpha);
            float cm = bilinear_interp(alpha, ct, CM_O_TBL_CT0, CM_O_TBL_CT05)
                       + cm_de * de
                       + bilinear_interp(alpha, ct, CM_Q_TBL_CT0, CM_Q_TBL_CT05) * q_hat;
            float cl_adot = bilinear_interp(alpha, ct, CL_ADOT_TBL_CT0, CL_ADOT_TBL_CT05);
            float cm_adot = bilinear_interp(alpha, ct, CM_ADOT_TBL_CT0, CM_ADOT_TBL_CT05);

            float qS = 0.5f * RHO * S * vt * vt;

            // Implicit alpha_dot loop, solved in closed form. Same
            // expression as aircraft/symmetric_full_grumman.py:
            //   gamma_d = [A*cl + A*cl_adot*k*q - (G/V)cos(gamma)]
            //             / (1 + A*cl_adot*k)
            // with k = c/(2V) and A = qS/(m V). The denominator stays within
            // 1.000 +/- 0.005 over the grid's airspeed range.
            float k_ad = CHORD / (2.0f * vt);
            float A_ad = qS / (MASS * vt);
            d_g = (A_ad * cl + A_ad * cl_adot * k_ad * q - (G / vt) * cosf(gamma))
                  / (1.0f + A_ad * cl_adot * k_ad);
            d_a = q - d_g;

            float cm_full = cm + cm_adot * k_ad * d_a;

            // Transfer to the CG: with cl and cd already closed by alpha-dot.
            float cl_full = cl + cl_adot * k_ad * d_a;
            float dcl_roll_, dcm_, dcn_;
            delta_momentos_cg(cl_full, cd, 0.0f, alpha, &dcl_roll_, &dcm_, &dcn_);
            cm_full += dcm_;

            float D = qS * cd;
            float My = qS * CHORD * cm_full;
            d_v = (-G * sinf(gamma) - D / MASS) / v_stall;
            d_q = My / I_YY;
        }

        // MCA local-consistency interpolation interval (Kushner-Dupuis),
        // ported from the 3DOF branch: dt_h = 1 / (sum_i |f_i|/h_i), clamped
        // to dt_max. h_i derived from the same grid metadata the barycentric
        // interpolation uses. 3DOF lesson: in the improvement argmax this dt
        // must be SHARED across candidate actions (evaluated at the current
        // policy's action) - a per-action horizon biases the Q-comparison
        // and fragments the policy map.
        __device__ float mca_dt_4dof(
            float gamma, float vn, float alpha, float q,
            float de, float throttle, float v_stall, float k_thrust,
            const float* b_low, const float* b_high, const int* g_shape,
            float dt_max
        ) {
            float k1_g, k1_v, k1_a, k1_q;
            derivs_4dof(gamma, vn, alpha, q, de, throttle, v_stall, k_thrust,
                        k1_g, k1_v, k1_a, k1_q);
            float h0 = (b_high[0] - b_low[0]) / (float)(g_shape[0] - 1);
            float h1 = (b_high[1] - b_low[1]) / (float)(g_shape[1] - 1);
            float h2 = (b_high[2] - b_low[2]) / (float)(g_shape[2] - 1);
            float h3 = (b_high[3] - b_low[3]) / (float)(g_shape[3] - 1);
            float rate = fabsf(k1_g) / h0 + fabsf(k1_v) / h1
                       + fabsf(k1_a) / h2 + fabsf(k1_q) / h3;
            float dt = 1.0f / (rate + 1e-6f);
            return fminf(dt, dt_max);
        }

        __device__ void rk4_step_4dof(
            float& gamma, float& vn, float& alpha, float& q,
            float de, float throttle, float dt_micro, int n_micro,
            float v_stall, float k_thrust, float& total_reward
        ) {
            total_reward = 0.0f;

            for(int m = 0; m < n_micro; ++m) {
                float k1_g, k1_v, k1_a, k1_q, k2_g, k2_v, k2_a, k2_q;
                float k3_g, k3_v, k3_a, k3_q, k4_g, k4_v, k4_a, k4_q;

                // --- k1 ---
                derivs_4dof(gamma, vn, alpha, q,
                            de, throttle, v_stall, k_thrust,
                            k1_g, k1_v, k1_a, k1_q);
                // --- k2 ---
                derivs_4dof(gamma + 0.5f*dt_micro*k1_g, vn + 0.5f*dt_micro*k1_v,
                            alpha + 0.5f*dt_micro*k1_a, q + 0.5f*dt_micro*k1_q,
                            de, throttle, v_stall, k_thrust,
                            k2_g, k2_v, k2_a, k2_q);
                // --- k3 ---
                derivs_4dof(gamma + 0.5f*dt_micro*k2_g, vn + 0.5f*dt_micro*k2_v,
                            alpha + 0.5f*dt_micro*k2_a, q + 0.5f*dt_micro*k2_q,
                            de, throttle, v_stall, k_thrust,
                            k3_g, k3_v, k3_a, k3_q);
                // --- k4 ---
                derivs_4dof(gamma + dt_micro*k3_g, vn + dt_micro*k3_v,
                            alpha + dt_micro*k3_a, q + dt_micro*k3_q,
                            de, throttle, v_stall, k_thrust,
                            k4_g, k4_v, k4_a, k4_q);

                gamma += (dt_micro / 6.0f) * (k1_g + 2.0f*k2_g + 2.0f*k3_g + k4_g);
                vn    += (dt_micro / 6.0f) * (k1_v + 2.0f*k2_v + 2.0f*k3_v + k4_v);
                alpha += (dt_micro / 6.0f) * (k1_a + 2.0f*k2_a + 2.0f*k3_a + k4_a);
                q     += (dt_micro / 6.0f) * (k1_q + 2.0f*k2_q + 2.0f*k3_q + k4_q);

                // PURE MARKOVIAN PENALTIES
                // 1. Primary Physical Cost: Altitude loss
                total_reward += dt_micro * vn * v_stall * sinf(gamma);

                if (gamma >= 0.0f) { break; }

                // 2. Pitch Damping Penalty
                total_reward -= W_Q_PENALTY * (q * q) * dt_micro;

                // 3. MARKOVIAN ALPHA BARRIER (Stall Prevention)
                // Thresholds aligned with Riley (1985): 14 deg = 0.24435 rad, -10 deg = -0.17453 rad
                if (alpha > 0.24435f) {
                    total_reward -= W_ALPHA_BARRIER_POS * (alpha - 0.24435f) * dt_micro;
                } else if (alpha < -0.17453f) {
                    total_reward -= W_ALPHA_BARRIER_NEG * (-alpha - 0.17453f) * dt_micro;
                }

                if (alpha >= 0.698132f || alpha <= -0.698132f || gamma <= -3.09159f) {
                    total_reward -= W_CRASH_PENALTY * v_stall;
                    break;
                }
            }

            // 4. Control Effort Penalty
            total_reward -= W_CONTROL_EFFORT * (de * de) * (dt_micro * n_micro);
            total_reward += W_THROTTLE_BONUS * throttle * fmaxf(1.0f - vn, 0.0f) * (dt_micro * n_micro);
        }

        __device__ void get_barycentric_4d(
            float s0, float s1, float s2, float s3,
            const float* b_low, const float* b_high, const int* g_shape, const int* strides,
            int* idxs, float* wgts
        ) {
            float n0 = (s0 - b_low[0]) / (b_high[0] - b_low[0]) * (g_shape[0] - 1);
            float n1 = (s1 - b_low[1]) / (b_high[1] - b_low[1]) * (g_shape[1] - 1);
            float n2 = (s2 - b_low[2]) / (b_high[2] - b_low[2]) * (g_shape[2] - 1);
            float n3 = (s3 - b_low[3]) / (b_high[3] - b_low[3]) * (g_shape[3] - 1);

            n0 = fmaxf(0.0f, fminf(n0, (float)(g_shape[0] - 1)));
            n1 = fmaxf(0.0f, fminf(n1, (float)(g_shape[1] - 1)));
            n2 = fmaxf(0.0f, fminf(n2, (float)(g_shape[2] - 1)));
            n3 = fmaxf(0.0f, fminf(n3, (float)(g_shape[3] - 1)));

            int i0 = (int)n0; int i1 = (int)n1; int i2 = (int)n2; int i3 = (int)n3;

            if (i0 == g_shape[0] - 1) i0--;
            if (i1 == g_shape[1] - 1) i1--;
            if (i2 == g_shape[2] - 1) i2--;
            if (i3 == g_shape[3] - 1) i3--;

            float d0 = n0 - i0; float d1 = n1 - i1;
            float d2 = n2 - i2; float d3 = n3 - i3;

            #pragma unroll
            for (int a = 0; a < 2; ++a) {
                #pragma unroll
                for (int b = 0; b < 2; ++b) {
                    #pragma unroll
                    for (int c = 0; c < 2; ++c) {
                        #pragma unroll
                        for (int d = 0; d < 2; ++d) {
                            int corner = a * 8 + b * 4 + c * 2 + d;
                            idxs[corner] = (i0 + a) * strides[0] + (i1 + b) * strides[1]
                                         + (i2 + c) * strides[2] + (i3 + d) * strides[3];
                            wgts[corner] = (a ? d0 : (1.0f - d0))
                                         * (b ? d1 : (1.0f - d1))
                                         * (c ? d2 : (1.0f - d2))
                                         * (d ? d3 : (1.0f - d3));
                        }
                    }
                }
            }
        }

        __global__ void policy_eval_kernel(
            const float* states, const float* actions, const int* policy,
            const float* V, float* new_V, const bool* is_term,
            const float* b_low, const float* b_high, const int* g_shape, const int* strides,
            int n_states, float gamma_discount, float dt_fixed, float v_stall, float k_thrust,
            int use_mca, float dt_max, int n_micro
        ) {
            int s_idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (s_idx >= n_states) return;
            if (is_term[s_idx]) { new_V[s_idx] = V[s_idx]; return; }

            int a_idx = policy[s_idx];
            float gamma = states[s_idx * 4 + 0], vn = states[s_idx * 4 + 1];
            float alpha = states[s_idx * 4 + 2], q = states[s_idx * 4 + 3];
            float de = actions[a_idx * 2 + 0], throttle = actions[a_idx * 2 + 1];
            float reward;

            // Macro step: fixed dt_fixed, or the endogenous MCA interval at
            // this state (evaluated at the policy's own action - which is
            // exactly the action being backed up here).
            float dt_eff = dt_fixed;
            if (use_mca) {
                dt_eff = mca_dt_4dof(gamma, vn, alpha, q, de, throttle,
                                     v_stall, k_thrust, b_low, b_high, g_shape, dt_max);
            }

            // n_micro RK4 micro-steps of dt_eff/n_micro each (default 10 -> 1ms
            // at dt_eff=0.01). Written as *(1/n) so n_micro=10 reproduces the
            // former dt*0.1f bit-exactly.
            rk4_step_4dof(gamma, vn, alpha, q, de, throttle,
                          dt_eff * (1.0f / (float)n_micro), n_micro, v_stall, k_thrust, reward);

            int idxs[16]; float wgts[16];
            get_barycentric_4d(gamma, vn, alpha, q, b_low, b_high, g_shape, strides, idxs, wgts);

            float expected_v = 0.0f;
            #pragma unroll
            for (int i = 0; i < 16; ++i) { expected_v = fmaf(wgts[i], V[idxs[i]], expected_v); }
            new_V[s_idx] = reward + gamma_discount * expected_v;
        }

        __global__ void policy_improve_kernel(
            const float* states, const float* actions, int* policy,
            const float* V, const bool* is_term,
            const float* b_low, const float* b_high, const int* g_shape, const int* strides,
            int n_states, int n_actions, float gamma_discount, float dt_fixed,
            float v_stall, float k_thrust,
            int use_mca, float dt_max, int mca_per_action, int n_micro,
            int* policy_changes
        ) {
            int s_idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (s_idx >= n_states) return;
            if (is_term[s_idx]) return;

            float init_gamma = states[s_idx * 4 + 0], init_vn = states[s_idx * 4 + 1];
            float init_alpha = states[s_idx * 4 + 2], init_q = states[s_idx * 4 + 3];
            float max_q_val = -1e9f; int best_a = 0;

            // Shared per-state timestep, evaluated at the CURRENT policy's
            // action, so all candidate actions integrate over the SAME
            // horizon (fair Q-comparison). mca_per_action=1 is the biased
            // control arm (each candidate gets its own dt) - 3DOF showed it
            // fragments the policy; kept only for the A/B study.
            float dt_shared = dt_fixed;
            if (use_mca && !mca_per_action) {
                int pa = policy[s_idx];
                dt_shared = mca_dt_4dof(init_gamma, init_vn, init_alpha, init_q,
                                        actions[pa * 2 + 0], actions[pa * 2 + 1],
                                        v_stall, k_thrust, b_low, b_high, g_shape, dt_max);
            }

            for (int a = 0; a < n_actions; ++a) {
                float gamma = init_gamma, vn = init_vn, alpha = init_alpha, q = init_q;
                float de = actions[a * 2 + 0], throttle = actions[a * 2 + 1], reward;

                float dt_eff = dt_shared;
                if (use_mca && mca_per_action) {
                    dt_eff = mca_dt_4dof(init_gamma, init_vn, init_alpha, init_q,
                                         de, throttle, v_stall, k_thrust,
                                         b_low, b_high, g_shape, dt_max);
                }

                rk4_step_4dof(gamma, vn, alpha, q, de, throttle,
                              dt_eff * (1.0f / (float)n_micro), n_micro, v_stall, k_thrust, reward);

                int idxs[16]; float wgts[16];
                get_barycentric_4d(gamma, vn, alpha, q, b_low, b_high, g_shape, strides, idxs, wgts);

                float expected_v = 0.0f;
                #pragma unroll
                for (int i = 0; i < 16; ++i) { expected_v = fmaf(wgts[i], V[idxs[i]], expected_v); }

                float q_val = reward + gamma_discount * expected_v;
                if (q_val > max_q_val) { max_q_val = q_val; best_a = a; }
            }

            if (policy[s_idx] != best_a) {
                policy[s_idx] = best_a;
                atomicAdd(policy_changes, 1);
            }
        }
        }
        '''
        module = cp.RawModule(code=cuda_source)
        self.eval_kernel = module.get_function('policy_eval_kernel')
        self.improve_kernel = module.get_function('policy_improve_kernel')
        self.threads_per_block = 256
        self.blocks_per_grid = (self.n_states + self.threads_per_block - 1) // self.threads_per_block

    def _pull_tensors_from_gpu(self) -> None:
        """
        Retrieves converged policy to CPU RAM using zero-copy transfers.
        """
        logger.info("Retrieving converged matrices from VRAM to CPU RAM...")

        gpu_tensors_to_free = [
            'd_new_value_function', 'd_terminal_mask',
            'd_states', 'd_actions',
            'd_bounds_low', 'd_bounds_high',
            'd_grid_shape', 'd_strides',
        ]
        for attr in gpu_tensors_to_free:
            if hasattr(self, attr):
                delattr(self, attr)

        cp.get_default_memory_pool().free_all_blocks()
        logger.info("Released unused GPU tensors and freed VRAM pool.")

        self.value_function = np.empty(self.n_states, dtype=np.float32)
        self.policy = np.empty(self.n_states, dtype=np.int32)

        chunk_size = 5_000_000
        for i in range(0, self.n_states, chunk_size):
            end = min(i + chunk_size, self.n_states)
            self.d_value_function[i:end].get(out=self.value_function[i:end])
            self.d_policy[i:end].get(out=self.policy[i:end])

        del self.d_value_function, self.d_policy
        cp.get_default_memory_pool().free_all_blocks()

        logger.success("Matrices successfully pulled to Host RAM. All VRAM released.")

    def warm_start_policy_from_ppo(
        self, ppo_path: Path, batch_size: int = 100_000,
    ) -> None:
        """
        Seed pi.d_policy with a trained PPO actor:

          1. Evaluate the PPO deterministic action at each grid state.
          2. Snap each continuous action to the nearest discrete action
             in pi.action_space (k-d tree query in 2D action space).
          3. Push the resulting (n_states,) int32 index array to VRAM.

        PI's policy_evaluation will rebuild V from scratch under this
        seeded policy, then policy_improvement / policy_evaluation
        iterate to convergence as usual. Bellman is a contraction, so
        the final converged V* is independent of the seed — warm start
        only changes convergence speed, not the limit.

        Why no V_φ warm start: SB3's PPO trains with VecNormalize
        (norm_reward=True), so its critic outputs values in normalized
        reward units. One pass of policy_evaluation under the seeded
        policy rebuilds V in the correct (un-normalized, meters of
        altitude loss) units automatically.

        Caveat: if PPO's training obs space was tighter than PI's grid
        (e.g. PI extended α to [-40°, 20°] while PPO was trained on
        [-14°, 20°]), the seeded policy in the extended region will be
        PPO's behaviour at the clipped boundary — generally not great,
        but PI refines it during run().
        """
        try:
            from scipy.spatial import cKDTree
            from stable_baselines3 import PPO
        except ImportError as e:
            raise RuntimeError(
                "warm_start_policy_from_ppo requires stable_baselines3 + scipy"
            ) from e

        logger.info(f"[*] Warm-start: loading PPO actor from {ppo_path}")
        ppo = PPO.load(ppo_path, device="cpu")

        n = self.n_states
        a_indices = np.empty(n, dtype=np.int32)

        logger.info(
            f"[*] Warm-start: querying PPO at {n:,} grid states "
            f"(batch={batch_size:,})..."
        )
        tree = cKDTree(self.action_space)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_obs = self.states_space[start:end]
            cont_actions, _ = ppo.predict(batch_obs, deterministic=True)
            _, idx = tree.query(cont_actions, k=1)
            a_indices[start:end] = idx.astype(np.int32)
            if (start // batch_size) % 10 == 0:
                logger.info(
                    f"    warmed {end:>10,} / {n:,}  ({100 * end / n:5.1f}%)"
                )

        self.d_policy[:] = cp.asarray(a_indices, dtype=cp.int32)
        logger.success(
            f"[+] Warm-start complete: {n:,} grid actions seeded from PPO."
        )

    def policy_evaluation(self) -> float:
        """Executes purely procedural evaluation on GPU."""
        delta = float("inf")
        SYNC_INTERVAL = 25

        for i in range(self.config.maximum_iterations):
            self.eval_kernel(
                (self.blocks_per_grid,), (self.threads_per_block,),
                (
                    self.d_states, self.d_actions, self.d_policy,
                    self.d_value_function, self.d_new_value_function, self.d_terminal_mask,
                    self.d_bounds_low, self.d_bounds_high, self.d_grid_shape, self.d_strides,
                    np.int32(self.n_states), np.float32(self.config.gamma),
                    np.float32(self.config.dt_fixed), np.float32(self.v_stall), np.float32(self.k_thrust),
                    np.int32(1 if self.config.use_mca_timestep else 0),
                    np.float32(self.config.dt_max), np.int32(self.config.n_micro)
                )
            )

            d_delta = max_abs_diff_kernel(self.d_new_value_function, self.d_value_function)
            self.d_value_function, self.d_new_value_function = self.d_new_value_function, self.d_value_function

            if i % SYNC_INTERVAL == 0 or i == self.config.maximum_iterations - 1:
                delta = float(d_delta.get())
                if delta < self.config.theta:
                    logger.success(f"GPU Evaluation converged at step {i} with Δ={delta:.5e}")
                    return delta

        msg = f"GPU Evaluation hit max iterations ({self.config.maximum_iterations})"
        logger.warning(f"{msg} with Δ={delta:.5e}")
        return delta

    def policy_improvement(self) -> bool:
        """Executes procedural policy greedy improvement on GPU."""
        d_policy_changes = cp.zeros(1, dtype=cp.int32)

        self.improve_kernel(
            (self.blocks_per_grid,), (self.threads_per_block,),
            (
                self.d_states, self.d_actions, self.d_policy,
                self.d_value_function, self.d_terminal_mask,
                self.d_bounds_low, self.d_bounds_high, self.d_grid_shape, self.d_strides,
                np.int32(self.n_states), np.int32(self.n_actions),
                np.float32(self.config.gamma), np.float32(self.config.dt_fixed),
                np.float32(self.v_stall),
                np.float32(self.k_thrust),
                np.int32(1 if self.config.use_mca_timestep else 0),
                np.float32(self.config.dt_max),
                np.int32(1 if self.config.mca_per_action else 0),
                np.int32(self.config.n_micro),
                d_policy_changes
            )
        )

        changes = int(d_policy_changes.get()[0])

        # FIX: Action Chattering Tolerance
        # Allow ~0.01% of states to oscillate between adjacent actions due to
        # floating-point noise. NOTE: this means policy iteration can declare
        # convergence while up to n_states*1e-4 states are still changing their
        # optimal action (1,487 on the riley grid, 564 on paper1). See
        # `n_states_chattering` in the saved metadata for the actual count.
        tolerance_threshold = int(self.n_states * 0.0001)

        policy_stable = (changes <= tolerance_threshold)

        #: states still changing their optimal action at the last sweep. Kept
        #: so that a run which stops under the tolerance can say how far under
        #: it actually was, instead of reporting a bare "converged".
        self.n_states_chattering = changes

        if not policy_stable:
            logger.info(
                f"GPU Policy updated: {changes} states changed optimal action. "
                f"(Tolerance: {tolerance_threshold})"
            )

        return policy_stable

    def run(self, save_path: Path | None = None) -> None:
        """Execute the complete Policy Iteration architecture.

        Saves ONLY when `save_path` is given. It used to always call save(),
        and save() falls back to the current working directory when it gets
        None, so every caller that saved afterwards -- train.py and
        paper_cg_sweep_solve.py both do -- wrote the policy twice: once
        anonymously into the CWD and once where it was meant to go. That is
        where the stray SymmetricStall_policy.npz at the root of the udesa
        tree came from, byte-identical to the one under results/politicas/.
        """
        self.n_policy_steps = 0
        self.final_residual = float("inf")
        self.n_states_chattering = 0
        for n in range(self.config.n_steps):
            logger.info(f"--- Iteration {n + 1}/{self.config.n_steps} ---")
            self.final_residual = self.policy_evaluation()
            is_stable = self.policy_improvement()
            self.n_policy_steps = n + 1

            if is_stable:
                if self.n_states_chattering:
                    logger.success(
                        f"Algorithm converged at iteration {n + 1} with "
                        f"{self.n_states_chattering} states still changing "
                        f"action, under the {int(self.n_states * 0.0001)} "
                        f"chattering tolerance."
                    )
                else:
                    logger.success(
                        f"Algorithm converged optimally at iteration {n + 1}: "
                        f"no state changed its action."
                    )
                break

        self._pull_tensors_from_gpu()
        if save_path is not None:
            self.save(save_path)

    def save(
        self, filepath: Path | None = None, metadata: dict | None = None
    ) -> None:
        """Serialize and save the trained model to disk.

        `metadata` is stored as JSON under the `run_metadata` key: it is what
        lets you tell, by looking at a policy, which thrust model, which CG and
        which grid trained it. Without it an .npz is anonymous.
        """
        if filepath is None:
            filepath = Path.cwd() / f"{self.env.unwrapped.__class__.__name__}_policy.npz"

        filepath = filepath.with_suffix(".npz")
        filepath.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Serializing policy to {filepath.resolve()}...")

        np.savez(
            filepath,
            run_metadata=np.array(json.dumps(metadata or {}, sort_keys=True)),
            value_function=self.value_function,
            policy=self.policy,
            bounds_low=self.bounds_low,
            bounds_high=self.bounds_high,
            grid_shape=self.grid_shape,
            strides=self.strides,
            corner_bits=self.corner_bits,
            action_space=self.action_space,
            use_mca_timestep=np.int32(1 if self.config.use_mca_timestep else 0),
            dt_fixed=np.float32(self.config.dt_fixed),
            dt_max=np.float32(self.config.dt_max),
            n_micro=np.int32(self.config.n_micro),
            mca_per_action=np.int32(1 if self.config.mca_per_action else 0),
        )

        logger.success(f"Policy saved successfully to {filepath.resolve()}")

    @classmethod
    def load(cls, filepath: Path, env: gym.Env = None) -> "PolicyIterationStall":
        """Load a saved policy instance from a serialized .npz archive."""
        filepath = filepath.with_suffix(".npz")

        logger.info(f"Loading policy from {filepath.resolve()}...")
        data = np.load(filepath)

        instance = cls.__new__(cls)
        instance.env = env
        instance.config = PolicyIterationStallConfig()

        instance.value_function = data["value_function"]
        instance.policy = data["policy"]
        instance.bounds_low = data["bounds_low"]
        instance.bounds_high = data["bounds_high"]
        instance.grid_shape = data["grid_shape"]
        instance.strides = data["strides"]
        instance.corner_bits = data["corner_bits"]
        instance.action_space = data["action_space"]

        # .npz archives predating runconfig.py do not carry it: those are the
        # anonymous policies that motivated adding it.
        if "run_metadata" in data.files:
            instance.run_metadata = json.loads(str(data["run_metadata"]))
            logger.info(f"Policy trained with: {instance.run_metadata}")
        else:
            instance.run_metadata = None
            logger.warning(
                "[!] This .npz does not record its training configuration "
                "(thrust model, CG, grid). It predates runconfig.py."
            )

        # Restore the timestep scheme used to train this policy (older
        # archives predate these fields — fall back to config defaults).
        if "use_mca_timestep" in data.files:
            instance.config.use_mca_timestep = bool(data["use_mca_timestep"])
            instance.config.dt_fixed = float(data["dt_fixed"])
            instance.config.dt_max = float(data["dt_max"])
            instance.config.n_micro = int(data["n_micro"])
            instance.config.mca_per_action = bool(data["mca_per_action"])

        instance.n_actions = len(instance.action_space)
        instance.n_dims = len(instance.bounds_low)

        instance.states_space = None

        logger.success(f"Policy loaded successfully from {filepath.resolve()}")
        return instance
