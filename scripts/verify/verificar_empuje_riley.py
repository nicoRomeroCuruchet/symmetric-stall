"""The thrust model against Riley's FLIGHT test (Table VI).

The other checks compare the code with the report or with another copy of the
code. This is the only one that compares it with the aircraft: Riley flew the
maximum-speed point and published the four quantities needed to reproduce it --
weight, altitude, airspeed and throttle position.

    TABLE VI, maximum-speed condition:
        W = 1558 lb   h = 6090 ft   V = 165 ft/s   delta_t = 0.856
        alpha measured in flight 2.81 deg, his own simulation 1.84 deg

If the thrust model is right, at that throttle and that airspeed the aircraft
must be in EQUILIBRIUM: the net axial force must come out zero. A miscalibrated
thrust gives itself away there, because the aircraft cannot sustain an airspeed
the flight test proves it sustained.

    THRUST_MODEL=riley PYTHONPATH=. python verificar_empuje_riley.py
"""
import os
import sys

import numpy as np

# Riley, Table VI, "Maximum speed" column
W_LB, H_FT, V_FTS, DT = 1558.0, 6090.0, 165.0, 0.856
ALPHA_FLIGHT, ALPHA_SIM_RILEY = 2.81, 1.84
LBF, FTS = 4.4482216, 0.3048
TOL_ACC = 0.05          # m/s2; the test is in stabilised flight


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
    for mode in ("riley", "paper1"):
        os.environ["THRUST_MODEL"] = mode
        # C_T does NOT change with altitude. Thrust scales with sigma
        # (eq. A10, T = T_sl*sigma) and so does dynamic pressure, so the ratio
        # T/(q_bar S) is the same as at sea level. Multiplying by sigma here
        # would apply it twice: 17% less thrust, and the aircraft "fails to
        # sustain" an airspeed it did sustain.
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
        filas.append((mode, ct, np.rad2deg(alpha), cd, acc))
        print("  %-7s C_T=%.4f  alpha_trim=%+5.2f deg  C_D_neto=%+.5f  "
              "aceleracion axial=%+.3f m/s2  %s"
              % (mode, ct, np.rad2deg(alpha), cd, acc,
                 "EQUILIBRIO" if abs(acc) < TOL_ACC else "no sostiene"))

    riley = filas[0]
    print("\n  alpha: vuelo %.2f deg, simulacion de Riley %.2f deg, "
          "este modelo %.2f deg" % (ALPHA_FLIGHT, ALPHA_SIM_RILEY, riley[2]))
    ok = abs(riley[4]) < TOL_ACC
    print("\n%s: with the Appendix A thrust the aircraft %s the measured airspeed"
          % ("PASS" if ok else "FAIL", "sustains" if ok else "does NOT sustain"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
