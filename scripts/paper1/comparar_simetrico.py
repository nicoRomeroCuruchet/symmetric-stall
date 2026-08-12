"""Compara los volcados de volcar_simetrico.py de las tres ramas.

Los tres modelos describen el MISMO avion. Puesto beta = mu = p = r = 0 y
aleron = timon = 0, el 6-DOF y el 8-DOF tienen que reducirse exactamente al
4-DOF. Cuando no lo hacen es porque una rama quedo atras de otra, y eso paso
mas de una vez: el 6-DOF no llevaba los terminos de alpha-punto y su q_dot
difería del 4-DOF por un factor 23 sin que ningun chequeo de esa rama lo
notara, porque todos los suyos comparaban la rama consigo misma.

    # en cada rama
    THRUST_MODEL=riley PYTHONPATH=. python volcar_simetrico.py v4.json
    # y despues
    python comparar_simetrico.py v4.json v6.json v8.json
"""
import json
import sys

import numpy as np

NOM = ["gamma_dot", "v_dot", "alpha_dot", "q_dot"]
TOL = 1e-4          # holgura de float32 en las tablas; muy por debajo de lo fisico


def main():
    rutas = sys.argv[1:] or ["v4.json", "v6.json", "v8.json"]
    V = {}
    for r in rutas:
        d = json.load(open(r))
        V[d["modelo"]] = d
    if "4dof" not in V:
        raise SystemExit("hace falta el volcado del 4-DOF: es la referencia")

    print("=== CONSTANTES DE LA CELULA ===")
    malas = 0
    claves = sorted(set().union(*[set(v["constantes"]) for v in V.values()]))
    orden = [k for k in ("4dof", "6dof", "8dof") if k in V]
    print("  %-26s %s" % ("", "".join("%15s" % k for k in orden)))
    for k in claves:
        vals = [V[n]["constantes"].get(k) for n in orden]
        igual = len({None if x is None else round(x, 9) for x in vals}) == 1
        malas += 0 if igual else 1
        print("  %-26s %s   %s"
              % (k, "".join("%15.6g" % (x if x is not None else float("nan"))
                            for x in vals),
                 "" if igual else "<-- DIFIERE"))

    print("\n=== C_T, con el mismo modelo de empuje ===")
    ref = np.array(V["4dof"]["ct"])
    for n in orden[1:]:
        d = np.abs(np.array(V[n]["ct"]) - ref).max()
        malas += 0 if d < 1e-9 else 1
        print("  %-6s peor diferencia contra el 4-DOF: %.2e %s"
              % (n, d, "" if d < 1e-9 else "<-- DIFIERE"))

    print("\n=== DERIVADAS EN EL LIMITE SIMETRICO, contra el 4-DOF ===")
    A = np.array(V["4dof"]["derivadas"])
    peor = 0.0
    for n in orden[1:]:
        B = np.array(V[n]["derivadas"])
        if B.shape != A.shape:
            print("  %-6s FORMA distinta: %s vs %s" % (n, B.shape, A.shape))
            malas += 1
            continue
        rel = np.abs(B - A) / np.maximum(np.abs(A), 1e-4)
        peor = max(peor, rel.max())
        print("  %-6s " % n + "  ".join("%s %.2e" % (NOM[i], rel[:, i].max())
                                        for i in range(4)))

    ok = malas == 0 and peor < TOL
    print("\n%s (peor %.2e sobre %d estados, tolerancia %.0e)"
          % ("LOS MODELOS SON CONSISTENTES" if ok else "LOS MODELOS DIFIEREN",
             peor, A.shape[0], TOL))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
