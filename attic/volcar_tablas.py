"""Vuelca TODAS las tablas y constantes de este modelo, para comparar ramas."""
import json, sys
import numpy as np
for mod, cls in (("aircraft.spin_grumman","SpinGrumman"),
                 ("aircraft.banked_spin_grumman","BankedSpinGrumman"),
                 ("aircraft.symmetric_full_grumman","SymmetricFullGrumman")):
    try:
        a = getattr(__import__(mod, fromlist=[cls]), cls)(); nombre = {"SpinGrumman":"8dof",
            "BankedSpinGrumman":"6dof","SymmetricFullGrumman":"4dof"}[cls]; break
    except ImportError:
        continue
fuera = {"modelo": nombre, "tablas": {}, "escalares": {}}
for n in dir(a):
    if n.startswith("__"): continue
    try: v = getattr(a, n)
    except Exception: continue
    if isinstance(v, np.ndarray):
        fuera["tablas"][n] = [float(x) for x in np.ravel(v)]
    elif isinstance(v, (int, float, np.floating)) and not isinstance(v, bool):
        fuera["escalares"][n] = float(v)
json.dump(fuera, open(sys.argv[1], "w"))
print("%s: %d tablas, %d escalares" % (nombre, len(fuera["tablas"]), len(fuera["escalares"])))
