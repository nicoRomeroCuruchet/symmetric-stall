"""Check that the CPU model and the CUDA kernel compute the SAME dynamics.

The aerodynamic tables are duplicated -- in aircraft/grumman.py and in the
.cu embedded in policy_iteration.py -- and so are the formulas. Nothing keeps
them in sync: the kernel trains the policy and the CPU evaluates it, so if they
diverge, the policy is optimised for an aircraft slightly different from the one
later simulated.

Comparing the tables is not enough: two identical tables with different formulas
give different results. This compares the DERIVATIVES evaluated over random
states across the whole grid, which is where any difference shows up.

Run after touching anything aerodynamic:

    python verificar_cpu_vs_kernel.py

Sale 0 si coinciden a precision de float32, 1 si no.
"""
import logging
import os
import pathlib
import sys

import numpy as np

logging.disable(logging.INFO)

TOL_REL = 1e-4          # generous, for float32 accumulated over ~20 operations
N_ESTADOS = 4000
SEMILLA = 3

DEFINES = "\n".join(
    f"#define {k} {v}f" for k, v in [
        ("W_Q_PENALTY", "0.0"), ("W_ALPHA_BARRIER_POS", "0.0"),
        ("W_ALPHA_BARRIER_NEG", "0.0"), ("W_CRASH_PENALTY", "1000.0"),
        ("W_CONTROL_EFFORT", "0.0"), ("W_THROTTLE_BONUS", "0.0"),
        ("DXCG_OVER_CHORD", "0.0"),
    ])
# THRUST_RILEY carries no f suffix: it is an integer flag for the
# preprocessor. And it must come from the SAME environment variable as the CPU
# model, or this script compares the kernel in one mode against the CPU in the
# other -- which is exactly what it reported the first time it was run with
# riley.
DEFINES += f"\n#define THRUST_RILEY {1 if os.environ.get('THRUST_MODEL','paper1').lower()=='riley' else 0}\n"

# The CG comes from the SAME environment variables as the plant. Pinning it to
# zero here would compare the kernel at Riley's CG against a CPU with the CG
# shifted -- the same mistake already made once with THRUST_MODEL.
DEFINES += (
    "\n#define CG_AFT_M " + repr(float(os.environ.get("CG_AFT_M", 0.0))) + "f"
    + "\n#define CG_RIGHT_M " + repr(float(os.environ.get("CG_RIGHT_M", 0.0))) + "f"
    + "\n#define CG_BELOW_M " + repr(float(os.environ.get("CG_BELOW_M", 0.0))) + "f\n"
)

WRAPPER = r'''
extern "C" __global__ void test_derivs(
        const float* st, const float* ac, int n,
        float v_stall, float k_thrust, float* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float dg, dv, da, dq;
    derivs_4dof(st[i*4+0], st[i*4+1], st[i*4+2], st[i*4+3],
                ac[i*2+0], ac[i*2+1], v_stall, k_thrust, dg, dv, da, dq);
    out[i*4+0] = dg; out[i*4+1] = dv; out[i*4+2] = da; out[i*4+3] = dq;
}
'''


def cuda_body():
    import symmetric_stall.policy_iteration as _pi
    txt = pathlib.Path(_pi.__file__).read_text()
    marker = "cuda_source = reward_defines + r'''"
    i = txt.index(marker) + len(marker)
    return txt[i:txt.index("'''", i)]


def derivs_cpu(ap, g, v, a, q, de, th, vs):
    """Calls THE PLANT. This used to be a hand copy of its expressions, and a
    hand copy only verifies that the kernel agrees with the copy."""
    return ap.derivatives(g, v, a, q, de, th)


def main():
    import cupy as cp
    from symmetric_stall.aircraft.symmetric_full_grumman import SymmetricFullGrumman

    kern = cp.RawModule(code=DEFINES + cuda_body() + WRAPPER,
                        options=('-std=c++11',)).get_function("test_derivs")
    ap = SymmetricFullGrumman()
    vs, kt = ap.STALL_AIRSPEED, ap.THROTTLE_LINEAR_MAPPING

    rng = np.random.default_rng(SEMILLA)
    n = N_ESTADOS
    # the airspeed floor follows the grid's: below 0.785 Vs
    # activates dCD_T, and if the sampling does not reach down there the term
    # is never exercised
    S = np.stack([rng.uniform(-1.5, 0.08, n), rng.uniform(0.4, 2.0, n),
                  rng.uniform(-0.69, 0.34, n),
                  rng.uniform(-0.87, 0.87, n)], 1).astype(np.float32)
    Ac = np.stack([rng.uniform(np.deg2rad(-25), np.deg2rad(15), n),
                   rng.uniform(0, 1, n)], 1).astype(np.float32)

    out = cp.zeros((n, 4), dtype=cp.float32)
    kern((n // 256 + 1,), (256,),
         (cp.asarray(S), cp.asarray(Ac), np.int32(n),
          np.float32(vs), np.float32(kt), out))
    gpu = cp.asnumpy(out)

    cpu = np.array([derivs_cpu(ap, *map(float, S[i]), *map(float, Ac[i]), vs)
                    for i in range(n)])

    rel = np.abs(gpu - cpu) / np.maximum(np.abs(cpu), 1e-6)
    # the verdict comes from the MAXIMUM, not the median: a term that only
    # appears in one corner of the domain -- a mistranscribed table near 40
    # degrees, dCD_T below 0.785 Vs -- does not move the median of 4000
    # samples, and that was exactly what had to be detected
    worst = 0.0
    print(f"{n} random states, CPU vs CUDA kernel:")
    for i, nom in enumerate(["gamma_dot", "v_dot", "alpha_dot", "q_dot"]):
        med = float(np.median(rel[:, i]))
        mx = float(rel[:, i].max())
        p99 = float(np.percentile(rel[:, i], 99))
        worst = max(worst, mx)
        j = int(rel[:, i].argmax())
        print(f"  {nom:10s} median {med:.2e}   p99 {p99:.2e}   MAX {mx:.2e}"
              f"   (worst case: V={S[j,1]:.2f} Vs, alpha={np.rad2deg(S[j,2]):+.1f} deg,"
              f" dt={Ac[j,1]:.2f})")
    # v_dot crosses zero (net drag changes sign with C_T), so the relative
    # error blows up there with nothing being wrong. The verdict uses a mixed
    # criterion: relative where the derivative has magnitude, absolute where it
    # cancels. The scale is each channel's standard deviation.
    scale = np.maximum(cpu.std(axis=0), 1e-9)
    mixed = np.abs(gpu - cpu) / np.maximum(np.abs(cpu), 0.01 * scale)
    print()
    for i, nom in enumerate(["gamma_dot", "v_dot", "alpha_dot", "q_dot"]):
        j = int(mixed[:, i].argmax())
        print(f"  {nom:10s} scale {scale[i]:.3e}   abs MAX {np.abs(gpu-cpu)[:, i].max():.3e}"
              f"   mixed MAX {mixed[:, i].max():.2e}"
              f"   (cpu={cpu[j, i]:+.3e} gpu={gpu[j, i]:+.3e})")
    worst = float(mixed.max())
    ok = worst < TOL_REL
    print(f"\n{'MATCH' if ok else 'DIFFER'} "
          f"(worst MAXIMUM {worst:.2e}, tolerance {TOL_REL:.0e})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
