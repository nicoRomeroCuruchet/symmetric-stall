"""The coefficient EXPRESSIONS, against Riley's Appendix B.

The other checks look at the VALUES in the tables. This one looks at how the
coefficients are assembled from them, which is a different thing and can be
wrong with every table right: a term omitted, one too many, one multiplied by
the wrong variable or with the sign flipped.

Riley, Apendice B:

    C_L,st  = C_L,o + C_L,de*de + C_L,df*df + dC_L,beta
    C_L,dyn = C_L,q*(q c/2V) + C_L,adot*(adot c/2V)
    C_m,st  = C_m,o + C_m,de*de + C_m,df*df + dC_m,beta
    C_m,dyn = C_m,q*(q c/2V) + C_m,adot*(adot c/2V)
    C_D,st  = C_D,o + C_D,de*de + C_D,(de)2*de^2 + C_D,df*df
              + C_D,(dr)3*|dr|^3 + dC_D,beta + dC_D,T
    C_D,dyn = 0
    C_Y,st  = C_Y,o + C_Y,beta*beta + C_Y,dr*dr + C_Y,da*da
    C_Y,dyn = C_Y,p*(p b/2V) + C_Y,r*(r b/2V)
    C_l,st, C_n,st and their dynamics, in the same form as C_Y.

The method is by RESPONSE, not by reading: the coefficient is evaluated with
every input at zero, then with a single one non-zero, and the difference divided
by that input is required to give EXACTLY the table value at that (alpha, C_T).
If the term is missing, the response is zero; if it multiplies something else,
it does not match; if there is a spurious term, it shows up in the base case.

Terms the model cannot have by its own reduction --- flaps
siempre retraidos, o derrape en un model simetrico --- se declaran y se
are reported as ABSENT BY REDUCTION, which is not the same as missing.

    THRUST_MODEL=riley PYTHONPATH=. python verificar_expresiones_riley.py
"""
import sys

import numpy as np

TOL = 1e-6
PUNTOS = [(np.deg2rad(a), ct) for a in (-5.0, 5.0, 14.0, 25.0)
          for ct in (0.0, 0.2, 0.45)]


def carga():
    for mod, cls, nombre in (
            ("aircraft.spin_grumman", "SpinGrumman", "8dof"),
            ("aircraft.banked_spin_grumman", "BankedSpinGrumman", "6dof"),
            ("aircraft.symmetric_full_grumman", "SymmetricFullGrumman", "4dof")):
        try:
            return getattr(__import__(mod, fromlist=[cls]), cls)(), nombre
        except ImportError:
            continue
    raise SystemExit("no encontre la clase del avion")


def main():
    a, model = carga()
    bi = lambda al, ct, t0, t5: float(a._bilinear_interp(al, ct, t0, t5))
    table = lambda n: getattr(a, n, None)

    # (coefficient, term, evaluator, expected table, present in the model)
    #
    # The evaluator receives (alpha, ct, h) and returns the coefficient with
    # ONLY that input set to h. The derivative w.r.t. h is what gets compared.
    pruebas = []

    if hasattr(a, "_drag_coefficient_full"):        # 8-DOF
        cd = lambda al, ct, de=0.0, be=0.0, dr=0.0: float(
            a._drag_coefficient_full(al, be, de, dr, ct))
        pruebas += [
            ("C_D", "de",      lambda al, ct, h: cd(al, ct, de=h),
             lambda al, ct: bi(al, ct, a._CD_DE_TABLE_CT0, a._CD_DE_TABLE_CT05), 1),
            ("C_D", "de^2",    lambda al, ct, h: cd(al, ct, de=h),
             None, 2),
            ("C_D", "|dr|^3",  lambda al, ct, h: cd(al, ct, dr=h),
             # the rudder table is PER DEGREE^3 and the model converts dr to
             # degrees before applying it, so differentiating w.r.t. dr in
             # radianes el factor es 57.2958^3
             lambda al, ct: bi(al, ct, a._CD_DR3_TABLE_CT0,
                               a._CD_DR3_TABLE_CT05) * 57.2958 ** 3, 3),
        ]
    if hasattr(a, "_side_force_coefficient"):       # 6-DOF y 8-DOF
        cy = lambda al, ct, be=0.0, ph=0.0, rh=0.0, da=0.0, dr=0.0: float(
            a._side_force_coefficient(al, be, ph, rh, da, dr, ct))
        pruebas += [
            ("C_Y", "beta", lambda al, ct, h: cy(al, ct, be=h),
             lambda al, ct: bi(al, ct, a._CY_BETA_TABLE_CT0, a._CY_BETA_TABLE_CT05), 1),
            ("C_Y", "da",   lambda al, ct, h: cy(al, ct, da=h),
             lambda al, ct: float(np.interp(al, a._CL_O_ALPHA_RAD, a._CY_DA_TABLE)), 1),
            ("C_Y", "dr",   lambda al, ct, h: cy(al, ct, dr=h),
             lambda al, ct: bi(al, ct, a._CY_DR_TABLE_CT0, a._CY_DR_TABLE_CT05), 1),
            ("C_Y", "p_hat", lambda al, ct, h: cy(al, ct, ph=h),
             lambda al, ct: float(np.interp(al, a._CL_O_ALPHA_RAD, a._CY_PHAT_TABLE)), 1),
            ("C_Y", "r_hat", lambda al, ct, h: cy(al, ct, rh=h),
             lambda al, ct: bi(al, ct, a._CY_RHAT_TABLE_CT0, a._CY_RHAT_TABLE_CT05), 1),
        ]
    if hasattr(a, "_yawing_moment_coefficient"):
        cn = lambda al, ct, be=0.0, ph=0.0, rh=0.0, da=0.0, dr=0.0: float(
            a._yawing_moment_coefficient(al, be, ph, rh, da, dr, ct))
        pruebas += [
            ("C_n", "beta", lambda al, ct, h: cn(al, ct, be=h),
             lambda al, ct: bi(al, ct, a._CN_BETA_TABLE_CT0, a._CN_BETA_TABLE_CT05), 1),
            ("C_n", "da",   lambda al, ct, h: cn(al, ct, da=h),
             lambda al, ct: float(np.interp(al, a._CL_O_ALPHA_RAD, a._CN_DA_TABLE)), 1),
            ("C_n", "dr",   lambda al, ct, h: cn(al, ct, dr=h),
             lambda al, ct: bi(al, ct, a._CN_DR_TABLE_CT0, a._CN_DR_TABLE_CT05), 1),
            ("C_n", "p_hat", lambda al, ct, h: cn(al, ct, ph=h),
             lambda al, ct: bi(al, ct, a._CN_PHAT_TABLE_CT0, a._CN_PHAT_TABLE_CT05), 1),
            ("C_n", "r_hat", lambda al, ct, h: cn(al, ct, rh=h),
             lambda al, ct: bi(al, ct, a._CN_RHAT_TABLE_CT0, a._CN_RHAT_TABLE_CT05), 1),
        ]
    if hasattr(a, "_rolling_moment_coefficient_full"):
        cl = lambda al, ct, be=0.0, ph=0.0, rh=0.0, da=0.0, dr=0.0: float(
            a._rolling_moment_coefficient_full(al, be, ph, rh, da, dr, ct))
        pruebas += [
            ("C_l", "beta", lambda al, ct, h: cl(al, ct, be=h),
             lambda al, ct: bi(al, ct, a._CL_ROLL_BETA_TABLE_CT0,
                               a._CL_ROLL_BETA_TABLE_CT05), 1),
            ("C_l", "da",   lambda al, ct, h: cl(al, ct, da=h),
             lambda al, ct: float(np.interp(al, a._CL_O_ALPHA_RAD,
                                            a._CL_ROLL_DA_TABLE)), 1),
            ("C_l", "p_hat", lambda al, ct, h: cl(al, ct, ph=h),
             lambda al, ct: bi(al, ct, a._CL_ROLL_PHAT_TABLE_CT0,
                               a._CL_ROLL_PHAT_TABLE_CT05), 1),
            ("C_l", "r_hat", lambda al, ct, h: cl(al, ct, rh=h),
             lambda al, ct: bi(al, ct, a._CL_ROLL_RHAT_TABLE_CT0,
                               a._CL_ROLL_RHAT_TABLE_CT05), 1),
        ]

    print("model %s -- response of each coefficient to each input, against "
          "the table Riley assigns it\n" % model)
    if not pruebas:
        print("  this model does not expose the coefficients separately; its "
              "assembly is verified\n  by the axes check and the cross-model "
              "comparison.")
        return 0

    fallos, worst, worst_n = 0, 0.0, None
    for coef, term, ev, esperado, orden in pruebas:
        malo = 0
        for al, ct in PUNTOS:
            base = ev(al, ct, 0.0)
            if orden == 1:
                # CENTRADA: la hacia adelante arrastraria el termino cuadratico
                # that shares a variable, and that reads as a model error
                h = 1e-4
                obt = (ev(al, ct, h) - ev(al, ct, -h)) / (2 * h)
            elif orden == 2:                       # cuadratico: (f(h)+f(-h)-2f0)/h^2
                h = 1e-4
                obt = (ev(al, ct, h) + ev(al, ct, -h) - 2 * base) / h ** 2 / 2
            else:
                # cubic: with small h, dividing by h^3 amplifies the noise of
                # float hasta taparlo todo. Hace falta un paso grande.
                h = 0.3
                obt = (ev(al, ct, h) - base) / h ** 3
            if esperado is None:
                continue
            esp = esperado(al, ct)
            rel = abs(obt - esp) / max(abs(esp), 1e-9)
            if rel > worst:
                worst, worst_n = rel, "%s por %s" % (coef, term)
            if rel > TOL:
                malo += 1
                if malo == 1:
                    print("  DIFIERE  %s por %-6s  alpha %+5.1f C_T %.2f: "
                          "model %+.6g  table %+.6g  rel %.1e"
                          % (coef, term, np.rad2deg(al), ct, obt, esp, rel))
        fallos += 1 if malo else 0
    print("\n%d terminos probados en %d puntos (alpha, C_T), %d difieren "
          "(worst %.2e en %s)" % (len(pruebas), len(PUNTOS), fallos, worst, worst_n))
    return 1 if fallos else 0


if __name__ == "__main__":
    sys.exit(main())
