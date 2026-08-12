"""La planta en ejes viento contra el Apendice B de Riley en ejes cuerpo.

Riley mezcla dos sistemas de ejes a proposito, y es facil equivocarse:

    fuerzas   -> ejes de ESTABILIDAD:  C_D,s  C_Y,s  C_L,s   (HAY que rotarlas)
    momentos  -> ejes CUERPO:          C_l,b  C_m,b  C_n,b   (van directo)

    "In calculating the external forces, use was made of wind-tunnel
     measurements obtained in the stability-axis system. The transformation
     [Fx,b; Fy,b; Fz,b] = R(alpha) [Fx,s; Fy,s; Fz,s] were employed"

Nuestro modelo 4-DOF no rota nada: escribe las ecuaciones directamente en
ejes viento. Eso es LEGITIMO solo porque aca beta = 0, y con beta = 0 el eje
de estabilidad coincide con el de viento. Pero "es legitimo" es un argumento,
y esto lo mide.

El lado de referencia implementa la cadena de Riley entera y es independiente:
resuelve su PROPIO punto fijo de alpha-punto iterando, sin tomar el valor que
calcula la planta en forma cerrada. Si los dos coincidieran solo porque uno le
pasa la respuesta al otro, la comparacion no probaria nada.

Usage:  THRUST_MODEL=riley PYTHONPATH=. python verificar_ejes_riley.py
"""
import sys

import numpy as np

from symmetric_stall.aircraft.symmetric_full_grumman import SymmetricFullGrumman

N = 4000
SEMILLA = 20260807
TOL = 1e-8          # las dos ramas son float64; esto es holgura de redondeo


def traslado_cg(a, cl, cd, cy, al):
    """(dC_l, dC_m, dC_n) por producto vectorial EXPLICITO, no llamando a la
    planta: si la planta se equivoca en un signo, esto no la acompana.

        M_CG = M_ref + (r_ref - r_CG) x F,   ejes de cuerpo (x adelante,
        y ala derecha, z abajo);  F/qS = (-C_A, C_Y, -C_N).
    """
    if a.CG_AFT == 0.0 and a.CG_RIGHT == 0.0 and a.CG_BELOW == 0.0:
        return 0.0, 0.0, 0.0
    c_n = cl * np.cos(al) + cd * np.sin(al)
    c_a = cd * np.cos(al) - cl * np.sin(al)
    r_cg = np.array([-a.CG_AFT, a.CG_RIGHT, a.CG_BELOW])
    dM = np.cross(-r_cg, np.array([-c_a, cy, -c_n]))
    return (dM[0] / a.WING_SPAN, dM[1] / a.CHORD, dM[2] / a.WING_SPAN)


def coeficientes(a, vt, al, q, de, th, alpha_dot):
    """C_L, C_D y C_m de Riley en ejes de ESTABILIDAD, para un alpha_dot dado."""
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
        cl, cd, cm = coeficientes(a, vt, al, q, de, th, alpha_dot)
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
        raise RuntimeError("el punto fijo de alpha_dot no convergio")

    v_dot = (u * u_dot + w * w_dot) / vt
    # momentos: Riley los da en ejes CUERPO, no se rotan
    q_dot = (qS * a.CHORD * cm) / a.I_YY
    return q - alpha_dot, v_dot / vs, alpha_dot, q_dot


def main():
    a = SymmetricFullGrumman()
    rng = np.random.default_rng(SEMILLA)
    nombres = ["gamma_dot", "v_dot", "alpha_dot", "q_dot"]
    peor = np.zeros(4)
    peor_caso = [None] * 4

    for _ in range(N):
        s = (rng.uniform(-1.5, 0.08), rng.uniform(0.4, 2.0),
             rng.uniform(-0.69, 0.34), rng.uniform(-0.87, 0.87))
        ac = (rng.uniform(np.deg2rad(-25), np.deg2rad(15)), rng.uniform(0, 1))
        p = np.asarray(a.derivatives(*s, *ac), dtype=np.float64)
        r = np.asarray(riley_ejes_cuerpo(a, *s, *ac), dtype=np.float64)
        rel = np.abs(p - r) / np.maximum(np.abs(r), 1e-6)
        for i in range(4):
            if rel[i] > peor[i]:
                peor[i] = rel[i]
                peor_caso[i] = (np.rad2deg(s[0]), s[1], np.rad2deg(s[2]))

    print(f"{N} estados aleatorios sobre toda la grilla "
          f"(V de 0.4 a 2.0 Vs, alpha de -40 a 20 deg)\n")
    print("  planta en ejes viento  vs  Apendice B rotando a ejes cuerpo:")
    for i, nom in enumerate(nombres):
        g0, v0, a0 = peor_caso[i]
        print(f"    {nom:10s} peor rel = {peor[i]:.2e}"
              f"   (gamma {g0:+6.1f} deg, V {v0:.2f} Vs, alpha {a0:+6.1f} deg)")
    ok = peor.max() < TOL
    print(f"\n{'IDENTICAS' if ok else 'DIFIEREN'} "
          f"(peor {peor.max():.2e}, tolerancia {TOL:.0e})")
    if ok:
        print("Con beta = 0 el eje de estabilidad ES el eje viento, asi que la "
              "rotacion de Riley\nya esta contenida en la forma en ejes viento. "
              "No falta ningun termino.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
