"""Vuelca las derivadas de ESTE modelo en el limite simetrico, a JSON.

Se corre igual en las tres ramas y despues se comparan los volcados. La idea
es que los tres modelos describen el MISMO avion: puesto beta = mu = p = r = 0
y aleron = timon = 0, el 6-DOF y el 8-DOF tienen que reducirse exactamente al
4-DOF. Si no lo hacen, alguna rama quedo atras de otra, que es justo lo que
paso varias veces.

Se vuelcan tambien las constantes de la celula: una diferencia ahi explica
cualquier diferencia posterior y conviene verla primero.

    THRUST_MODEL=riley PYTHONPATH=. python volcar_simetrico.py salida.json
"""
import json
import sys

import numpy as np

# los mismos estados y controles en las tres ramas, sin azar
ESTADOS = [(g, v, a, q)
           for g in (0.0, -0.2, -0.6)
           for v in (0.6, 0.95, 1.4)
           for a in (np.deg2rad(-20), np.deg2rad(5), np.deg2rad(16),
                     np.deg2rad(30))
           for q in (-0.5, 0.0, 0.3)]
CONTROLES = [(np.deg2rad(-20), 0.0), (np.deg2rad(-5), 0.5),
             (np.deg2rad(10), 1.0)]


def carga():
    """Devuelve (nombre, avion, funcion que da (g_dot, v_dot, a_dot, q_dot)).

    Se prueba del modelo mas grande al mas chico: la clase del 4-DOF existe
    tambien en las otras ramas como remanente, asi que probarla primero
    devuelve el modelo equivocado en 2 de 3 casos.
    """
    try:
        from aircraft.spin_grumman import SpinGrumman
        ap = SpinGrumman()
        def f8(g, v, a, q, de, dt):
            d = ap._derivatives(g, v, a, 0.0, 0.0, 0.0, q, 0.0,
                                de, 0.0, dt, 0.0, ap.STALL_AIRSPEED)
            return (d[0], d[1], d[2], d[6])
        return ("8dof", ap, f8)
    except ImportError:
        pass
    try:
        from aircraft.banked_spin_grumman import BankedSpinGrumman
        ap = BankedSpinGrumman()
        def f(g, v, a, q, de, dt):
            d = ap.derivatives(g, v, a, 0.0, 0.0, q, de, 0.0, dt,
                               ap.STALL_AIRSPEED)
            return (d[0], d[1], d[2], d[5])
        return ("6dof", ap, f)
    except ImportError:
        pass
    from aircraft.symmetric_full_grumman import SymmetricFullGrumman
    ap = SymmetricFullGrumman()
    return ("4dof", ap,
            lambda g, v, a, q, de, dt: tuple(ap.derivatives(g, v, a, q, de, dt)))


nombre, ap, derivar = carga()
fuera = {
    "modelo": nombre,
    "constantes": {k: float(getattr(ap, k))
                   for k in ("MASS", "WING_SURFACE_AREA", "CHORD", "I_YY",
                             "STALL_AIRSPEED", "AIR_DENSITY", "GRAVITY",
                             "THROTTLE_LINEAR_MAPPING")
                   if hasattr(ap, k)},
    "ct": [[float(ap._compute_ct(dt, f * ap.STALL_AIRSPEED))
            for f in (0.5, 0.95, 1.5)] for _, dt in CONTROLES],
    "derivadas": [],
}
for g, v, a, q in ESTADOS:
    for de, dt in CONTROLES:
        fuera["derivadas"].append([float(x) for x in derivar(g, v, a, q, de, dt)])

salida = sys.argv[1] if len(sys.argv) > 1 else "volcado.json"
with open(salida, "w") as fh:
    json.dump(fuera, fh)
print("%s: %d estados x %d controles = %d filas -> %s"
      % (nombre, len(ESTADOS), len(CONTROLES), len(fuera["derivadas"]), salida))
