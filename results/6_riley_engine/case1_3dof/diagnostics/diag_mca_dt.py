"""Roll the MCA policy out with the MCA timestep, not with a fixed one.

The validation integrates the closed loop at config.dt_fixed = 0.10 s. That is
right for a policy solved with a fixed step, and it is what the comment above
that line argues for. But this policy was solved with the MCA endogenous step,
which is a function of the state:

    rate = |gamma_dot|/h_gamma + |v_dot|/h_v  (+ |mu_dot|/h_mu if mca_full)
    dt   = min(1/rate, dt_max)

Executing it at a constant 0.10 s therefore evaluates it on a different system
from the one it was optimised for -- the same class of error the comment
describes, in the opposite direction. Four of the five entries level off in
~5 s either way; the inverted one runs the full 200 s step cap and never levels,
which is where the -1002.92 m in the validation figure comes from.

This replays both integrations side by side. If the MCA rollout levels the
inverted entry, the figure's outlier is a rollout bug and not a DP result.
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from main import train_or_load_policy
from analysis.experiments import get_setup_for_level
from analysis.interpolation import get_optimal_action

logging.disable(logging.INFO)

env, states, actions, config = get_setup_for_level(4)()
config.use_mca_timestep = True
config.mca_include_mudot = True
config.dt_max = 0.10
pi, trained = train_or_load_policy(env, states, actions, config,
                                   "banked_glider_L4_mca", retrain=False)
assert not trained, "it retrained; abort"

ap = env.airplane
v_stall = ap.STALL_AIRSPEED
lo, hi = np.asarray(pi.bounds_low), np.asarray(pi.bounds_high)
shape = np.asarray(pi.grid_shape)
h_gamma, h_vn, h_mu = (hi - lo) / (shape - 1)      # cell sizes, same as kernel


def derivatives(gamma, vn, mu, cl, mu_dot):
    """Mirror of get_derivatives in the CUDA kernel."""
    v_true = vn * v_stall
    q = 0.5 * ap.AIR_DENSITY * (ap.WING_SURFACE_AREA / ap.MASS)
    alpha = (cl - ap.CL_0) / ap.CL_ALPHA
    cd = ap.CD_0 + ap.CD_ALPHA * alpha + ap.CD_ALPHA2 * alpha ** 2
    g_dot = q * v_true * cl * np.cos(mu) - (ap.GRAVITY / v_true) * np.cos(gamma)
    v_dot = (-ap.GRAVITY * np.sin(gamma) - q * v_true ** 2 * cd) / v_stall
    return g_dot, v_dot, mu_dot


def mca_dt(gamma, vn, mu, cl, mu_dot):
    k1g, k1v, k1m = derivatives(gamma, vn, mu, cl, mu_dot)
    rate = abs(k1g) / h_gamma + abs(k1v) / h_vn
    if config.mca_include_mudot:
        rate += abs(k1m) / h_mu
    return min(1.0 / (rate + 1e-6), config.dt_max)


def rollout(mu0_deg, use_mca_dt):
    gamma, vn, mu = np.deg2rad(-60.0), 1.2, np.deg2rad(mu0_deg)
    h, t, n = 0.0, 0.0, 0
    while gamma < 0.0 and n < 2000:
        sv = np.array([gamma, vn, mu], dtype=np.float32)
        action, _ = get_optimal_action(sv, pi)
        cl, mu_dot = float(action[0]), float(action[1])
        dt = mca_dt(gamma, vn, mu, cl, mu_dot) if use_mca_dt else config.dt_fixed
        h += vn * v_stall * np.sin(gamma) * dt
        # RK4 on the same three states, with this dt
        k1 = derivatives(gamma, vn, mu, cl, mu_dot)
        k2 = derivatives(gamma + .5*dt*k1[0], vn + .5*dt*k1[1], mu + .5*dt*k1[2], cl, mu_dot)
        k3 = derivatives(gamma + .5*dt*k2[0], vn + .5*dt*k2[1], mu + .5*dt*k2[2], cl, mu_dot)
        k4 = derivatives(gamma + dt*k3[0], vn + dt*k3[1], mu + dt*k3[2], cl, mu_dot)
        gamma += dt/6*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
        vn    += dt/6*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
        mu    += dt/6*(k1[2] + 2*k2[2] + 2*k3[2] + k4[2])
        vn = min(max(vn, lo[1]), hi[1])
        t += dt
        n += 1
    return n, t, h, (gamma >= 0.0)


print(f"cell sizes: h_gamma={np.rad2deg(h_gamma):.4f} deg  h_vn={h_vn:.5f}  "
      f"h_mu={np.rad2deg(h_mu):.4f} deg   dt_max={config.dt_max}")
print()
print(f"{'mu0':>5} | {'dt fijo 0.10':^30} | {'dt MCA':^30}")
print(f"{'':>5} | {'pasos':>6}{'T(s)':>8}{'dh(m)':>10}{'niv':>5} | "
      f"{'pasos':>6}{'T(s)':>8}{'dh(m)':>10}{'niv':>5}")
print("-" * 76)
for mu0 in (150.0, 120.0, 90.0, 60.0, 30.0):
    a = rollout(mu0, use_mca_dt=False)
    b = rollout(mu0, use_mca_dt=True)
    f = lambda r: f"{r[0]:6d}{r[1]:8.1f}{r[2]:10.2f}{'si' if r[3] else 'NO':>5}"
    print(f"{mu0:5.0f} | {f(a)} | {f(b)}")
print()
print("NLP (CasADi) para referencia: 150->-500.01  120->-193.14  90->-150.61  "
      "60->-119.48  30->-107.76")
