"""El modelo de empuje contra el ensayo en VUELO de Riley (Tabla VI).

Los otros chequeos comparan el codigo con el informe o con otra copia del
codigo. Este es el unico que lo compara con el avion: Riley volo el punto de
velocidad maxima y publico las cuatro cantidades que hacen falta para
reproducirlo -- peso, altitud, velocidad y posicion de palanca.

    TABLA VI, condicion de velocidad maxima:
        W = 1558 lb   h = 6090 ft   V = 165 ft/s   delta_t = 0.856
        alpha medido en vuelo 2.81 deg, su propia simulacion 1.84 deg

Si el modelo de empuje es correcto, a esa palanca y esa velocidad el avion
tiene que quedar en EQUILIBRIO: la fuerza axial neta debe dar cero. Un empuje
mal calibrado se delata ahi, porque el avion no puede sostener una velocidad
que el ensayo demuestra que sostuvo.

    THRUST_MODEL=riley PYTHONPATH=. python verificar_empuje_riley.py
"""
import os
import sys

import numpy as np

# Riley, Tabla VI, columna "Maximum speed"
W_LB, H_FT, V_FTS, DT = 1558.0, 6090.0, 165.0, 0.856
ALPHA_VUELO, ALPHA_SIM_RILEY = 2.81, 1.84
LBF, FTS = 4.4482216, 0.3048
TOL_ACC = 0.05          # m/s2; el ensayo esta en vuelo estabilizado


def carga():
    for mod, cls in (("aircraft.spin_grumman", "SpinGrumman"),
                     ("aircraft.banked_spin_grumman", "BankedSpinGrumman"),
                     ("aircraft.symmetric_full_grumman", "SymmetricFullGrumman")):
        try:
            m = __import__(mod, fromlist=[cls])
            return getattr(m, cls)()
        except ImportError:
            continue
    raise SystemExit("no encontre la clase del avion")


def main():
    a = carga()
    W = W_LB * 0.453592 * a.GRAVITY
    V = V_FTS * FTS
    sigma = (1.0 - 6.87535e-6 * H_FT) ** 4.2561      # atmosfera estandar
    rho = 1.225 * sigma
    qS = 0.5 * rho * a.WING_SURFACE_AREA * V * V

    print("Riley Tabla VI (vuelo real): W=%.0f lb  h=%.0f ft  V=%.0f ft/s  dt=%.3f"
          % (W_LB, H_FT, V_FTS, DT))
    print("  sigma=%.4f  rho=%.4f kg/m3  q_bar*S=%.0f N  C_L necesario=%.4f\n"
          % (sigma, rho, qS, W / qS))

    al = np.deg2rad(np.linspace(-6.0, 12.0, 18001))
    filas = []
    for modo in ("riley", "paper1"):
        os.environ["THRUST_MODEL"] = modo
        # C_T NO cambia con la altitud. El empuje escala con sigma (ec. A10,
        # T = T_sl*sigma) y la presion dinamica tambien, asi que el cociente
        # T/(q_bar S) es el mismo que a nivel del mar. Multiplicar por sigma
        # aca lo aplicaria dos veces: da 17% menos empuje y el avion "no
        # sostiene" una velocidad que sostuvo.
        ct = a._compute_ct(DT, V)
        cl = np.array([a._bilinear_interp(x, ct, a._CL_O_TABLE,
                                          a._CL_O_TABLE_CT05) for x in al])
        i = int(np.argmin(np.abs(cl - W / qS)))
        alpha = al[i]
        cd = float(a._bilinear_interp(alpha, ct, a._CD_O_TABLE,
                                      a._CD_O_TABLE_CT05))
        if hasattr(a, "_delta_cd_thrust"):
            cd += float(a._delta_cd_thrust(ct, alpha))
        acc = -cd * qS / (W_LB * 0.453592)
        filas.append((modo, ct, np.rad2deg(alpha), cd, acc))
        print("  %-7s C_T=%.4f  alpha_trim=%+5.2f deg  C_D_neto=%+.5f  "
              "aceleracion axial=%+.3f m/s2  %s"
              % (modo, ct, np.rad2deg(alpha), cd, acc,
                 "EQUILIBRIO" if abs(acc) < TOL_ACC else "no sostiene"))

    riley = filas[0]
    print("\n  alpha: vuelo %.2f deg, simulacion de Riley %.2f deg, "
          "este modelo %.2f deg" % (ALPHA_VUELO, ALPHA_SIM_RILEY, riley[2]))
    ok = abs(riley[4]) < TOL_ACC
    print("\n%s: con el empuje del Apendice A el avion %s la velocidad medida"
          % ("PASA" if ok else "FALLA", "sostiene" if ok else "NO sostiene"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
