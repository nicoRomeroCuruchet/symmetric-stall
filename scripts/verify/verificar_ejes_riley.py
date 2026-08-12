"""The wind-axes plant against Riley's Appendix B in body axes.

Riley mixes two axis systems on purpose, and it is easy to get wrong:

    forces   -> STABILITY axes:  C_D,s  C_Y,s  C_L,s   (these MUST be rotated)
    moments  -> BODY axes:       C_l,b  C_m,b  C_n,b   (these go straight in)

    "In calculating the external forces, use was made of wind-tunnel
     measurements obtained in the stability-axis system. The transformation
     [Fx,b; Fy,b; Fz,b] = R(alpha) [Fx,s; Fy,s; Fz,s] were employed"

Our 4-DOF model rotates nothing: it writes the equations directly in wind
axes. That is LEGITIMATE only because beta = 0 here, and with beta = 0 the
stability axis coincides with the wind axis. But "it is legitimate" is an
argument, and this measures it.

The reference side implements Riley's whole chain and is independent: it solves
its OWN alpha-dot fixed point by iteration, without taking the value the plant
computes in closed form. If the two agreed merely because one hands the answer
to the other, the comparison would prove nothing.

Usage:  THRUST_MODEL=riley PYTHONPATH=. python verificar_ejes_riley.py
"""
import sys

import numpy as np

from symmetric_stall.aircraft.symmetric_full_grumman import SymmetricFullGrumman

N = 4000
SEMILLA = 20260807
TOL = 1e-8          # both branches are float64; this is rounding slack


def traslado_cg(a, cl, cd, cy, al):
    """(dC_l, dC_m, dC_n) by an EXPLICIT cross product, not by calling the
    planta: si la planta se equivoca en un signo, esto no la acompana.

        M_CG = M_ref + (r_ref - r_CG) x F,   body axes (x forward,
       y right wing, z down);  F/qS = (-C_A, C_Y, -C_N).
    """
    if a.CG_AFT == 0.0 and a.CG_RIGHT == 0.0 and a.CG_BELOW == 0.0:
        return 0.0, 0.0, 0.0
    c_n = cl * np.cos(al) + cd * np.sin(al)
    c_a = cd * np.cos(al) - cl * np.sin(al)
    r_cg = np.array([-a.CG_AFT, a.CG_RIGHT, a.CG_BELOW])
    dM = np.cross(-r_cg, np.array([-c_a, cy, -c_n]))
    return (dM[0] / a.WING_SPAN, dM[1] / a.CHORD, dM[2] / a.WING_SPAN)


def coefficients(a, vt, al, q, de, th, alpha_dot):
    """Riley's C_L, C_D and C_m in STABILITY axes, for a given alpha_dot."""
    q_hat = q * a.CHORD / (2.0 * vt)
    a_hat = alpha_dot * a.CHORD / (2.0 * vt)
    ct = a._compute_ct(th, vt)
    bi = lambda t0, t5: float(a._bilinear_interp(al, ct, t0, t5))
    cl = (bi(a._CL_O_TABLE, a._CL_O_TABLE_CT05)
          + bi(a._CL_DE_TABLE_CT0, a._CL_DE_TABLE_CT05) * de
          + bi(a._CL_Q_TABLE, a._CL_Q_TABLE_CT05) * q_hat
          + bi(a._CL_ADOT_TABLE_CT0, a._CL_ADOT_TABLE_CT05) * a_hat)
    cd = (bi(a._CD_O_TABLE, a._CD_O_TABLE_CT05)
          + bi(a._CD_DE_TABLE_CT0, a._CD_DE_TABLE_CT05) * de
          + bi(a._CD_DE2_TABLE_CT0, a._CD_DE2_TABLE_CT05) * de * de
          + float(a._delta_cd_thrust(ct, al)))
    cm = (bi(a._CM_O_TABLE, a._CM_O_TABLE_CT05)
          + bi(a._CM_DE_TABLE_CT0, a._CM_DE_TABLE_CT05) * de
          + bi(a._CM_Q_TABLE, a._CM_Q_TABLE_CT05) * q_hat
          + bi(a._CM_ADOT_TABLE_CT0, a._CM_ADOT_TABLE_CT05) * a_hat)
    cm += traslado_cg(a, cl, cd, 0.0, al)[1]
    return cl, cd, cm


def riley_ejes_cuerpo(a, gam, vn, al, q, de, th):
    """Apendice B literal: estabilidad -> rotacion de alpha -> ejes cuerpo."""
    vs = a.STALL_AIRSPEED
    m, g = a.MASS, a.GRAVITY
    vt = max(vn * vs, 0.1)
    qS = 0.5 * a.AIR_DENSITY * a.WING_SURFACE_AREA * vt * vt
    theta = gam + al                    # simetrico: theta = gamma + alpha
    u, w = vt * np.cos(al), vt * np.sin(al)

    # punto fijo propio de alpha_dot: alpha_dot -> C_L -> fuerzas -> alpha_dot
    alpha_dot = 0.0
    for _ in range(200):
        cl, cd, cm = coefficients(a, vt, al, q, de, th, alpha_dot)
        # fuerzas en ejes de estabilidad
        Fxs, Fzs = -cd * qS, -cl * qS
        # LA ROTACION del Apendice B
        Fxb = np.cos(al) * Fxs - np.sin(al) * Fzs
        Fzb = np.sin(al) * Fxs + np.cos(al) * Fzs
        # ecuaciones de cuerpo (v = p = r = 0, phi = 0)
        u_dot = -q * w - g * np.sin(theta) + Fxb / m
        w_dot = q * u + g * np.cos(theta) + Fzb / m
        nuevo = (u * w_dot - w * u_dot) / (vt * vt)
        if abs(nuevo - alpha_dot) < 1e-14:
            alpha_dot = nuevo
            break
        alpha_dot = nuevo
    else:
        raise RuntimeError("the alpha_dot fixed point did not converge")

    v_dot = (u * u_dot + w * w_dot) / vt
    # moments: Riley gives them in BODY axes, they are not rotated
    q_dot = (qS * a.CHORD * cm) / a.I_YY
    return q - alpha_dot, v_dot / vs, alpha_dot, q_dot


def main():
    a = SymmetricFullGrumman()
    rng = np.random.default_rng(SEMILLA)
    names = ["gamma_dot", "v_dot", "alpha_dot", "q_dot"]
    worst = np.zeros(4)
    worst_case = [None] * 4

    for _ in range(N):
        s = (rng.uniform(-1.5, 0.08), rng.uniform(0.4, 2.0),
             rng.uniform(-0.69, 0.34), rng.uniform(-0.87, 0.87))
        ac = (rng.uniform(np.deg2rad(-25), np.deg2rad(15)), rng.uniform(0, 1))
        p = np.asarray(a.derivatives(*s, *ac), dtype=np.float64)
        r = np.asarray(riley_ejes_cuerpo(a, *s, *ac), dtype=np.float64)
        rel = np.abs(p - r) / np.maximum(np.abs(r), 1e-6)
        for i in range(4):
            if rel[i] > worst[i]:
                worst[i] = rel[i]
                worst_case[i] = (np.rad2deg(s[0]), s[1], np.rad2deg(s[2]))

    print(f"{N} random states over the whole grid "
          f"(V from 0.4 to 2.0 Vs, alpha from -40 to 20 deg)\n")
    print("  wind-axes plant  vs  Appendix B rotated into body axes:")
    for i, name in enumerate(names):
        g0, v0, a0 = worst_case[i]
        print(f"    {name:10s} worst rel = {worst[i]:.2e}"
              f"   (gamma {g0:+6.1f} deg, V {v0:.2f} Vs, alpha {a0:+6.1f} deg)")
    ok = worst.max() < TOL
    print(f"\n{'IDENTICAL' if ok else 'DIFFERENT'} "
          f"(worst {worst.max():.2e}, tolerance {TOL:.0e})")
    if ok:
        print("With beta = 0 the stability axis IS the wind axis, so Riley's "
              "rotation\nis already contained in the wind-axes form. No term "
              "is missing.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
