"""Compare the volcar_simetrico.py dumps from the three branches.

Los tres modelos describen el MISMO avion. Puesto beta = mu = p = r = 0 y
aileron = rudder = 0, the 6-DOF and the 8-DOF must reduce exactly to the
4-DOF. When they do not it is because one branch fell behind another, and that
happened more than once: the 6-DOF did not carry the alpha-dot terms and its
q_dot differed from the 4-DOF by a factor of 23 without any check on that
branch noticing, because all of its own checks compared the branch to itself.

    # on each branch
    THRUST_MODEL=riley PYTHONPATH=. python volcar_simetrico.py v4.json
    # y despues
    python comparar_simetrico.py v4.json v6.json v8.json
"""
import json
import sys

import numpy as np

NOM = ["gamma_dot", "v_dot", "alpha_dot", "q_dot"]
TOL = 1e-4          # float32 slack in the tables; far below anything physical


def main():
    rutas = sys.argv[1:] or ["v4.json", "v6.json", "v8.json"]
    V = {}
    for r in rutas:
        d = json.load(open(r))
        V[d["modelo"]] = d
    if "4dof" not in V:
        raise SystemExit("hace falta el volcado del 4-DOF: es la referencia")

    print("=== CONSTANTES DE LA CELULA ===")
    bad = 0
    claves = sorted(set().union(*[set(v["constantes"]) for v in V.values()]))
    orden = [k for k in ("4dof", "6dof", "8dof") if k in V]
    print("  %-26s %s" % ("", "".join("%15s" % k for k in orden)))
    for k in claves:
        vals = [V[n]["constantes"].get(k) for n in orden]
        igual = len({None if x is None else round(x, 9) for x in vals}) == 1
        bad += 0 if igual else 1
        print("  %-26s %s   %s"
              % (k, "".join("%15.6g" % (x if x is not None else float("nan"))
                            for x in vals),
                 "" if igual else "<-- DIFIERE"))

    print("\n=== C_T, with the same thrust model ===")
    ref = np.array(V["4dof"]["ct"])
    for n in orden[1:]:
        d = np.abs(np.array(V[n]["ct"]) - ref).max()
        bad += 0 if d < 1e-9 else 1
        print("  %-6s worst difference against the 4-DOF: %.2e %s"
              % (n, d, "" if d < 1e-9 else "<-- DIFIERE"))

    print("\n=== DERIVATIVES IN THE SYMMETRIC LIMIT, against the 4-DOF ===")
    A = np.array(V["4dof"]["derivadas"])
    worst = 0.0
    for n in orden[1:]:
        B = np.array(V[n]["derivadas"])
        if B.shape != A.shape:
            print("  %-6s FORMA distinta: %s vs %s" % (n, B.shape, A.shape))
            bad += 1
            continue
        rel = np.abs(B - A) / np.maximum(np.abs(A), 1e-4)
        worst = max(worst, rel.max())
        print("  %-6s " % n + "  ".join("%s %.2e" % (NOM[i], rel[:, i].max())
                                        for i in range(4)))

    ok = bad == 0 and worst < TOL
    print("\n%s (worst %.2e over %d states, tolerance %.0e)"
          % ("LOS MODELOS SON CONSISTENTES" if ok else "LOS MODELOS DIFIEREN",
             worst, A.shape[0], TOL))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
