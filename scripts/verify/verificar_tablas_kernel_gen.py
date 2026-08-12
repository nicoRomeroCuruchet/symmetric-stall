"""Compara literal por literal las tablas del kernel CUDA contra las del CPU.

El kernel lleva las tablas de Riley DUPLICADAS como literales dentro del
fuente CUDA, porque un __device__ const float no se puede alimentar desde
numpy sin pagar una indireccion por lectura. Esa duplicacion ya costo dos
errores en este proyecto: una conversion grado->radian robada de la tabla de
al lado, y cinco tablas escritas con cinco cifras cuando el CPU las calcula
con quince. Ninguno de los dos aparece en el chequeo de derivadas, que compara
resultados finales y promedia justo lo que hay que ver.

Esto no mide fisica: mide que las dos copias digan lo mismo.

El emparejamiento de nombres es automatico, con varias reglas candidatas, y
SE REPORTA: una tabla del kernel sin par, o una del CPU sin usar, se lista en
vez de ignorarse. Si una regla emparejara mal, los valores no coincidirian y
saldria como diferencia, que es el resultado que se quiere.

Usage:  PYTHONPATH=. python verificar_tablas_kernel.py <fuente_kernel.py>
"""
import re
import sys

import numpy as np

TOL = 1e-6      # float32 redondea a ~1e-7; por encima de esto es transcripcion

# Las tablas de empuje admiten DOS convenciones, y las ramas no coinciden:
# el 6-DOF las guarda crudas, en lbf y ft/s como el informe, y convierte
# dentro de compute_ct; el 8-DOF las guarda ya en N y N/(m/s) porque su
# compute_ct usa vt en m/s directamente. Las dos son correctas. El chequeo
# prueba las dos y REPORTA cual encontro, en vez de suponer una: suponerla
# marcaria como error una rama sana, y -- peor -- taparia el caso en que la
# tabla este de verdad mal por un factor parecido.
_LBF, _FTS = 4.4482216, 1.0 / 0.3048
ESCALA_ALTERNATIVA = {"THR_T0": _LBF, "THR_T1": _LBF * _FTS}


def carga_avion():
    for mod, cls in (("aircraft.spin_grumman", "SpinGrumman"),
                     ("aircraft.banked_spin_grumman", "BankedSpinGrumman"),
                     ("aircraft.symmetric_full_grumman", "SymmetricFullGrumman")):
        try:
            m = __import__(mod, fromlist=[cls])
            return getattr(m, cls)()
        except ImportError:
            continue
    raise SystemExit("no encontre la clase del avion en esta rama")


def tablas_del_kernel(fuente):
    # Los comentarios adentro de las llaves traen numeros -- factores de
    # conversion, rangos de alpha -- y si se cuelan al extraer, corren TODAS
    # las entradas de esa tabla y la diferencia que sale despues es del parser,
    # no del kernel. Se sacan antes de mirar nada.
    fuente = re.sub(r"/\*.*?\*/", " ", fuente, flags=re.S)
    fuente = re.sub(r"//[^\n]*", " ", fuente)
    patron = re.compile(
        r"__device__\s+const\s+float\s+([A-Z_0-9]+)\s*\[\s*(\d+)\s*\]\s*=\s*\{([^}]*)\}",
        re.S)
    fuera = {}
    for nombre, n, cuerpo in patron.findall(fuente):
        # Las entradas pueden ser productos: "-0.00116f*57.2958f". Hay que
        # evaluar cada elemento, no juntar los numeros sueltos: si no, el
        # factor de conversion entra como si fuera un valor de la tabla y
        # corre todo un lugar.
        vals = []
        for elem in cuerpo.split(","):
            nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", elem)
            if not nums:
                continue
            x = 1.0
            for s in nums:
                x *= float(s)
            vals.append(x)
        if len(vals) != int(n):
            print(f"  !! {nombre}: lei {len(vals)} entradas y declara {n}")
        fuera[nombre] = np.array(vals[:int(n)], dtype=np.float64)
    return fuera


def candidatos(kn):
    """Nombres de atributo plausibles para una tabla del kernel.

    Las ramas no usan la misma convencion -- TBL contra nada, ROLL contra R,
    ADOT contra AD -- asi que se prueban varias y se reporta cual pego.
    """
    # Abreviaturas del kernel -> nombre largo del CPU. Se declaran a mano
    # porque adivinarlas emparejaria mal en silencio; lo que no encuentre par
    # se reporta.
    ALIAS = {"CL_RB": "CL_ROLL_BETA", "CL_RP": "CL_ROLL_PHAT",
             "CL_RR": "CL_ROLL_RHAT", "CL_RDA": "CL_ROLL_DA",
             "CL_RDR": "CL_ROLL_DR", "CL_RO": "CL_ROLL_O",
             "CN_B": "CN_BETA", "CN_P": "CN_PHAT", "CN_R": "CN_RHAT",
             "CY_B": "CY_BETA", "CY_P": "CY_PHAT", "CY_R": "CY_RHAT",
             "CL_AD": "CL_ADOT", "CM_AD": "CM_ADOT",
             "ALPHA_TBL": "CL_O_ALPHA_RAD", "CL_A_TBL": "CL_O_ALPHA_RAD"}
    base = kn
    formas = {base}
    for corto, largo in ALIAS.items():
        if base == corto or base.startswith(corto + "_"):
            formas.add(base.replace(corto, largo, 1))
    for a, b in (("_TBL", "_TABLE"), ("_TBL", ""), ("TBL_", "TABLE_"),
                 ("_B10", "_TABLE_B10"), ("_B20", "_TABLE_B20"),
                 ("_CT0", "_TABLE_CT0"), ("_CT05", "_TABLE_CT05")):
        for f in list(formas):
            formas.add(f.replace(a, b))
    salida = set()
    for f in formas:
        salida |= {"_" + f, "_" + f + "_TABLE", "_" + f.replace("_CT0", "_TABLE_CT0")}
        if f.endswith("_CT0"):
            salida.add("_" + f[:-4] + "_TABLE")           # CT0 -> sin sufijo
        if f.endswith("_CT05"):
            salida.add("_" + f[:-5] + "_TABLE_CT05")
    salida.add("_" + base.replace("_TBL", "") + "_TABLE")
    return salida


def main():
    ruta = sys.argv[1] if len(sys.argv) > 1 else None
    if ruta is None:
        raise SystemExit("uso: verificar_tablas_kernel.py <fuente_kernel.py>")
    K = tablas_del_kernel(open(ruta).read())
    a = carga_avion()
    attrs = {n: np.asarray(getattr(a, n), dtype=np.float64)
             for n in dir(a) if n.startswith("_") and
             isinstance(getattr(a, n, None), np.ndarray)}

    print(f"{len(K)} tablas en {ruta}, {len(attrs)} arreglos en el modelo CPU\n")
    emparejadas, sin_par, fallos, peor, peor_n = 0, [], 0, 0.0, None
    convenciones = []
    usados = set()
    for kn in sorted(K):
        kv = K[kn]
        cn = next((c for c in candidatos(kn)
                   if c in attrs and attrs[c].shape == kv.shape), None)
        if cn is None:
            sin_par.append(kn)
            continue
        emparejadas += 1
        usados.add(cn)
        cv = attrs[cn]
        rel = np.abs(kv - cv) / np.maximum(np.abs(cv), 1e-9)
        if kn in ESCALA_ALTERNATIVA and rel.max() > TOL:
            cv2 = attrs[cn] * ESCALA_ALTERNATIVA[kn]
            rel2 = np.abs(kv - cv2) / np.maximum(np.abs(cv2), 1e-9)
            if rel2.max() <= rel.max():
                cv, rel = cv2, rel2
                convenciones.append("%s: el kernel la guarda ya convertida "
                                    "(x%.6g)" % (kn, ESCALA_ALTERNATIVA[kn]))
            else:
                convenciones.append("%s: el kernel la guarda cruda" % kn)
        i = int(rel.argmax())
        if rel[i] > peor:
            peor, peor_n = rel[i], kn
        if rel[i] > TOL:
            fallos += 1
            print(f"  DIFIERE  {kn:<22} <-> {cn:<26} entrada {i:2d}: "
                  f"kernel {kv[i]:+.7g}  CPU {cv[i]:+.7g}  rel {rel[i]:.2e}")

    for c in convenciones:
        print("  convencion detectada -- " + c)
    if sin_par:
        print(f"\n  SIN PAR en el CPU ({len(sin_par)}): {', '.join(sin_par)}")
    huerfanas = [n for n in attrs if n not in usados and attrs[n].size == 14]
    if huerfanas:
        print(f"  arreglos del CPU de 14 entradas sin usar ({len(huerfanas)}): "
              f"{', '.join(sorted(huerfanas))}")

    print(f"\n{emparejadas} tablas emparejadas, {fallos} difieren, "
          f"{len(sin_par)} sin par  (peor {peor:.2e} en {peor_n})")
    return 1 if (fallos or sin_par) else 0


if __name__ == "__main__":
    sys.exit(main())
