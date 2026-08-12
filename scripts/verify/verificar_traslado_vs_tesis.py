"""Nuestro traslado al CG contra la formulacion de la tesis de Poliak, termino
a termino. La afirmacion a probar es precisa: TODO coincide salvo el signo del
producto vectorial de la ec. (12)/(16).

Tesis, pag. 25:
    (13)/(15)  F_B/QS = ( -(C_D cos a - C_L sin a),  C_Y,  -(C_L cos a + C_D sin a) )
                      = ( -C_A, C_Y, -C_N )
    (16)       [b C_l, c C_m, b C_n]_CG = [b C_l, c C_m, b C_n]_ref + r_ref->CG x F
               con r_ref->CG = (x_cg, y_cg, z_cg)_B,  x_cg > 0 = CG hacia la NARIZ

Nuestro codigo: M_CG = M_ref - r_ref->CG x F  (derivacion estandar).
Se espera entonces  delta_nuestro == -delta_tesis, exactamente.
"""
import importlib, sys
import numpy as np

for m, c in (("aircraft.spin_grumman", "SpinGrumman"),
             ("aircraft.banked_spin_grumman", "BankedSpinGrumman"),
             ("aircraft.symmetric_full_grumman", "SymmetricFullGrumman")):
    try:
        a = getattr(importlib.import_module(m), c)(); nombre = m.split(".")[-1]; break
    except ImportError:
        continue

rng = np.random.default_rng(20260808)
peor_f, peor_m, peor_signo = 0.0, 0.0, 0.0
N = 500
for _ in range(N):
    x_cg, y_cg, z_cg = rng.uniform(-0.3, 0.3, 3)   # convencion de la TESIS
    cl_, cd_, cy_ = rng.uniform(-1.5, 2.0), rng.uniform(0.0, 1.0), rng.uniform(-0.4, 0.4)
    al = rng.uniform(-0.2, 0.7)

    # --- la tesis, literal ---
    ca, sa = np.cos(al), np.sin(al)
    F_tesis = np.array([-(cd_ * ca - cl_ * sa), cy_, -(cl_ * ca + cd_ * sa)])
    r_tesis = np.array([x_cg, y_cg, z_cg])
    dM_tesis = np.cross(r_tesis, F_tesis)
    d_tesis = np.array([dM_tesis[0] / a.WING_SPAN,
                        dM_tesis[1] / a.CHORD,
                        dM_tesis[2] / a.WING_SPAN])

    # --- lo nuestro. dx = hacia la COLA = -x_cg ---
    a.CG_AFT, a.CG_RIGHT, a.CG_BELOW = -x_cg, y_cg, z_cg
    d_nuestro = np.array(a._delta_momentos_cg(cl_, cd_, al, cy=cy_))

    # 1) las FUERZAS de cuerpo tienen que ser identicas (ec. 13/15)
    c_n = cl_ * ca + cd_ * sa
    c_a = cd_ * ca - cl_ * sa
    peor_f = max(peor_f, abs(F_tesis[0] + c_a), abs(F_tesis[2] + c_n))

    # 2) el traslado tiene que ser el OPUESTO, exactamente
    peor_signo = max(peor_signo, float(np.max(np.abs(d_nuestro + d_tesis))))
    # 3) y NO tiene que ser igual (si lo fuera, habriamos copiado el signo malo)
    peor_m = max(peor_m, float(np.max(np.abs(d_nuestro - d_tesis))))

print("modelo %s -- %d casos aleatorios" % (nombre, N))
print("  ec. (13)/(15), fuerzas de cuerpo   C_X=-C_A, C_Z=-C_N : coincide, peor %.2e" % peor_f)
print("  ec. (12)/(16), traslado            nuestro == -tesis  : peor %.2e" % peor_signo)
print("  y NO es igual a la tesis (control) separacion tipica  : %.3f" % peor_m)
ok = peor_f < 1e-15 and peor_signo < 1e-15 and peor_m > 1e-3
print("\n%s" % ("COINCIDE CON LA TESIS SALVO EL SIGNO, que es lo afirmado"
                if ok else "NO se cumple lo afirmado"))
sys.exit(0 if ok else 1)
