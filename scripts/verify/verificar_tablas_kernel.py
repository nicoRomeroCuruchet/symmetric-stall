"""Compara literal por literal las tablas del kernel CUDA contra las del CPU.

El kernel lleva las tablas de Riley DUPLICADAS como literales dentro del
fuente CUDA, porque un __device__ const float no se puede alimentar desde
numpy sin pagar una indireccion por lectura. Esa duplicacion ya nos costo dos
errores: una conversion grado->radian robada de la tabla de al lado, y un
CM_DE a C_T=0.5 que difiere en la cuarta cifra. Ninguno de los dos aparece en
verificar_cpu_vs_kernel.py, que compara las derivadas finales con la mediana
y por lo tanto promedia justo lo que hay que ver.

Esto compara los 26 arreglos entrada por entrada. No mide fisica: mide que
las dos copias digan lo mismo.

Usage:  PYTHONPATH=. python verificar_tablas_kernel.py
"""
import re
import sys

import numpy as np

import PolicyIteration as PI
from symmetric_stall.aircraft.symmetric_full_grumman import SymmetricFullGrumman

# nombre en el kernel -> atributo en el CPU. Se declara a mano a proposito:
# derivarlo por regla (TBL->TABLE) haria que un nombre mal escrito de un lado
# se empareje con el que no es, que es exactamente el error que se busca.
PARES = {
    "CL_O_TBL_CT0": "_CL_O_TABLE",       "CL_O_TBL_CT05": "_CL_O_TABLE_CT05",
    "CL_Q_TBL_CT0": "_CL_Q_TABLE",       "CL_Q_TBL_CT05": "_CL_Q_TABLE_CT05",
    "CD_O_TBL_CT0": "_CD_O_TABLE",       "CD_O_TBL_CT05": "_CD_O_TABLE_CT05",
    "CM_O_TBL_CT0": "_CM_O_TABLE",       "CM_O_TBL_CT05": "_CM_O_TABLE_CT05",
    "CM_Q_TBL_CT0": "_CM_Q_TABLE",       "CM_Q_TBL_CT05": "_CM_Q_TABLE_CT05",
    "CL_DE_TBL_CT0": "_CL_DE_TABLE_CT0", "CL_DE_TBL_CT05": "_CL_DE_TABLE_CT05",
    "CM_DE_TBL_CT0": "_CM_DE_TABLE_CT0", "CM_DE_TBL_CT05": "_CM_DE_TABLE_CT05",
    "CD_DE_TBL_CT0": "_CD_DE_TABLE_CT0", "CD_DE_TBL_CT05": "_CD_DE_TABLE_CT05",
    "CD_DE2_TBL_CT0": "_CD_DE2_TABLE_CT0",
    "CD_DE2_TBL_CT05": "_CD_DE2_TABLE_CT05",
    "CL_ADOT_TBL_CT0": "_CL_ADOT_TABLE_CT0",
    "CL_ADOT_TBL_CT05": "_CL_ADOT_TABLE_CT05",
    "CM_ADOT_TBL_CT0": "_CM_ADOT_TABLE_CT0",
    "CM_ADOT_TBL_CT05": "_CM_ADOT_TABLE_CT05",
    "THR_DTP": "_THR_DTP", "THR_T0": "_THR_T0", "THR_T1": "_THR_T1",
    # los quiebres de alpha: si estos se corren, se corren TODAS las tablas
    "CL_A_TBL": "_CL_O_ALPHA_RAD",
}


def tablas_del_kernel(fuente: str) -> dict:
    """Extrae los __device__ const float NOMBRE[N] = { ... } del fuente CUDA."""
    patron = re.compile(
        r"__device__\s+const\s+float\s+([A-Z_0-9]+)\s*\[\s*(\d+)\s*\]\s*=\s*\{([^}]*)\}",
        re.S)
    fuera = {}
    for nombre, n, cuerpo in patron.findall(fuente):
        vals = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?f?",
                                             cuerpo.replace("f", ""))]
        if len(vals) != int(n):
            print(f"  !! {nombre}: declara {n} entradas y tiene {len(vals)}")
        fuera[nombre] = np.array(vals, dtype=np.float64)
    return fuera


def main() -> int:
    fuente = PI.PolicyIterationStall.KERNEL_SOURCE if hasattr(
        PI.PolicyIterationStall, "KERNEL_SOURCE") else open(
            "PolicyIteration.py").read()
    K = tablas_del_kernel(fuente)
    a = SymmetricFullGrumman()

    print(f"tablas encontradas en el fuente CUDA: {len(K)}")
    faltan = [k for k in PARES if k not in K]
    if faltan:
        print(f"  !! declaradas en el mapeo pero ausentes del kernel: {faltan}")
    huerfanas = [k for k in K if k not in PARES]
    if huerfanas:
        print(f"  nota: en el kernel y sin par declarado: {huerfanas}")
    print()

    peor_nombre, peor = None, 0.0
    fallos = 0
    for kn, cn in sorted(PARES.items()):
        if kn not in K:
            continue
        kv = K[kn]
        cv = np.asarray(getattr(a, cn), dtype=np.float64)
        if kv.shape != cv.shape:
            print(f"  FORMA  {kn:<20} kernel {kv.shape} vs CPU {cv.shape}")
            fallos += 1
            continue
        denom = np.maximum(np.abs(cv), 1e-9)
        rel = np.abs(kv - cv) / denom
        i = int(rel.argmax())
        # float32 redondea a ~1e-7; por encima de 1e-6 es transcripcion, no formato
        estado = "ok" if rel[i] <= 1e-6 else "DIFIERE"
        if rel[i] > 1e-6:
            fallos += 1
            print(f"  {estado:<8}{kn:<20} entrada {i:2d}: kernel {kv[i]:+.6f} "
                  f"vs CPU {cv[i]:+.6f}  rel {rel[i]:.2e}")
        if rel[i] > peor:
            peor, peor_nombre = rel[i], kn

    print()
    if fallos == 0:
        print(f"LAS {len(PARES)} TABLAS COINCIDEN (peor {peor:.2e} en {peor_nombre})")
        return 0
    print(f"{fallos} TABLA(S) NO COINCIDEN (peor {peor:.2e} en {peor_nombre})")
    return 1


if __name__ == "__main__":
    sys.exit(main())
