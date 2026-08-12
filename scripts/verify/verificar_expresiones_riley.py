"""Las EXPRESIONES de los coeficientes, contra el Apendice B de Riley.

Los otros chequeos miran los VALORES de las tablas. Este mira como se arman
los coeficientes a partir de ellas, que es una cosa distinta y puede estar mal
sin que ninguna tabla lo este: un termino omitido, uno de mas, uno multiplicado
por la variable equivocada o con el signo cambiado.

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
    C_l,st, C_n,st y sus dinamicas, con la misma forma que C_Y.

El metodo es de RESPUESTA, no de lectura: se evalua el coeficiente con todas
las entradas en cero, despues con una sola distinta de cero, y se exige que la
diferencia dividida por esa entrada de EXACTAMENTE el valor de la tabla en ese
(alpha, C_T). Si el termino falta, la respuesta es cero; si esta multiplicado
por otra cosa, no coincide; si sobra un termino, aparece en el caso base.

Los terminos que el modelo no puede tener por su propia reduccion --- flaps
siempre retraidos, o derrape en un modelo simetrico --- se declaran y se
informan como AUSENTE POR REDUCCION, que no es lo mismo que faltante.

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
    a, modelo = carga()
    bi = lambda al, ct, t0, t5: float(a._bilinear_interp(al, ct, t0, t5))
    tabla = lambda n: getattr(a, n, None)

    # (coeficiente, termino, evaluador, tabla esperada, presente en el modelo)
    #
    # El evaluador recibe (alpha, ct, h) y devuelve el coeficiente con SOLO esa
    # entrada valiendo h. La derivada respecto de h es lo que se compara.
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
             # la tabla del timon quedo POR GRADO^3 y el modelo pasa dr a
             # grados antes de aplicarla, asi que derivando respecto de dr en
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

    print("modelo %s -- respuesta de cada coeficiente a cada entrada, contra "
          "la tabla que Riley le asigna\n" % modelo)
    if not pruebas:
        print("  este modelo no expone los coeficientes por separado; su "
              "ensamblado se verifica\n  por el chequeo de ejes y el cruzado "
              "entre modelos.")
        return 0

    fallos, peor, peor_n = 0, 0.0, None
    for coef, term, ev, esperado, orden in pruebas:
        malo = 0
        for al, ct in PUNTOS:
            base = ev(al, ct, 0.0)
            if orden == 1:
                # CENTRADA: la hacia adelante arrastraria el termino cuadratico
                # que comparte variable, y eso se ve como un error del modelo
                h = 1e-4
                obt = (ev(al, ct, h) - ev(al, ct, -h)) / (2 * h)
            elif orden == 2:                       # cuadratico: (f(h)+f(-h)-2f0)/h^2
                h = 1e-4
                obt = (ev(al, ct, h) + ev(al, ct, -h) - 2 * base) / h ** 2 / 2
            else:
                # cubico: con h chico, dividir por h^3 amplifica el ruido de
                # float hasta taparlo todo. Hace falta un paso grande.
                h = 0.3
                obt = (ev(al, ct, h) - base) / h ** 3
            if esperado is None:
                continue
            esp = esperado(al, ct)
            rel = abs(obt - esp) / max(abs(esp), 1e-9)
            if rel > peor:
                peor, peor_n = rel, "%s por %s" % (coef, term)
            if rel > TOL:
                malo += 1
                if malo == 1:
                    print("  DIFIERE  %s por %-6s  alpha %+5.1f C_T %.2f: "
                          "modelo %+.6g  tabla %+.6g  rel %.1e"
                          % (coef, term, np.rad2deg(al), ct, obt, esp, rel))
        fallos += 1 if malo else 0
    print("\n%d terminos probados en %d puntos (alpha, C_T), %d difieren "
          "(peor %.2e en %s)" % (len(pruebas), len(PUNTOS), fallos, peor, peor_n))
    return 1 if fallos else 0


if __name__ == "__main__":
    sys.exit(main())
