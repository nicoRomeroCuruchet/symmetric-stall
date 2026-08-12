"""Compare the CUDA kernel's tables literal by literal against the CPU's.

The kernel carries Riley's tables DUPLICATED as literals inside the CUDA
source, because a __device__ const float cannot be fed from numpy without
paying an indirection per read. That duplication has already cost two bugs in
this project: a degree-to-radian conversion stolen from the neighbouring table,
and five tables written to five digits when the CPU computes them to fifteen.
Neither shows up in the derivatives check, which compares final results and
averages away exactly what needs to be seen.

This does not measure physics: it measures that the two copies say the same
thing.

Name pairing is automatic, with several candidate rules, and IS REPORTED: a
kernel table with no pair, or a CPU one left unused, gets listed rather than
ignored. If a rule paired things wrongly the values would not match and it
would come out as a difference, which is the desired outcome.

Usage:  PYTHONPATH=. python verificar_tablas_kernel.py <fuente_kernel.py>
"""
import re
import sys

import numpy as np

TOL = 1e-6      # float32 rounds to ~1e-7; above this it is transcription

# The thrust tables admit TWO conventions, and the branches disagree: the
# 6-DOF stores them raw, in lbf and ft/s as the report does, and converts
# inside compute_ct; the 8-DOF stores them already in N and N/(m/s) because its
# compute_ct usa vt en m/s directamente. Las dos son correctas. El chequeo
# tries both and REPORTS which one it found, instead of assuming: assuming
# would flag a healthy branch as broken and -- worse -- would mask the case
# where the table really is wrong by a similar factor.
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
    raise SystemExit("could not find the aircraft class on this branch")


def tablas_del_kernel(source):
    # The comments inside the braces carry numbers -- conversion factors
    # conversion, rangos de alpha -- y si se cuelan al extraer, corren TODAS
    # the entries of that table, and the difference that follows is the parser's,
    # no del kernel. Se sacan antes de mirar nada.
    source = re.sub(r"/\*.*?\*/", " ", source, flags=re.S)
    source = re.sub(r"//[^\n]*", " ", source)
    pattern = re.compile(
        r"__device__\s+const\s+float\s+([A-Z_0-9]+)\s*\[\s*(\d+)\s*\]\s*=\s*\{([^}]*)\}",
        re.S)
    out = {}
    for name, n, body in pattern.findall(source):
        # Entries can be products: "-0.00116f*57.2958f". Each element has to
        # be evaluated, not the loose numbers collected: otherwise the
        # conversion factor enters as if it were a table value and
        # corre todo un lugar.
        vals = []
        for elem in body.split(","):
            nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", elem)
            if not nums:
                continue
            x = 1.0
            for s in nums:
                x *= float(s)
            vals.append(x)
        if len(vals) != int(n):
            print(f"  !! {name}: lei {len(vals)} entradas y declara {n}")
        out[name] = np.array(vals[:int(n)], dtype=np.float64)
    return out


def candidatos(kn):
    """Plausible attribute names for a kernel table.

    The branches do not share a convention -- TBL vs nothing, ROLL vs R,
    ADOT vs AD -- so several are tried and the one that stuck is reported.
    """
    # Abreviaturas del kernel -> name largo del CPU. Se declaran a mano
    # because guessing them would pair wrongly in silence; whatever finds no pair
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
    out = set()
    for f in formas:
        out |= {"_" + f, "_" + f + "_TABLE", "_" + f.replace("_CT0", "_TABLE_CT0")}
        if f.endswith("_CT0"):
            out.add("_" + f[:-4] + "_TABLE")             # CT0 -> no suffix
        if f.endswith("_CT05"):
            out.add("_" + f[:-5] + "_TABLE_CT05")
    out.add("_" + base.replace("_TBL", "") + "_TABLE")
    return out


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
    paired, unpaired, failures, worst, worst_n = 0, [], 0, 0.0, None
    convenciones = []
    usados = set()
    for kn in sorted(K):
        kv = K[kn]
        cn = next((c for c in candidatos(kn)
                   if c in attrs and attrs[c].shape == kv.shape), None)
        if cn is None:
            unpaired.append(kn)
            continue
        paired += 1
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
        if rel[i] > worst:
            worst, worst_n = rel[i], kn
        if rel[i] > TOL:
            failures += 1
            print(f"  DIFIERE  {kn:<22} <-> {cn:<26} entrada {i:2d}: "
                  f"kernel {kv[i]:+.7g}  CPU {cv[i]:+.7g}  rel {rel[i]:.2e}")

    for c in convenciones:
        print("  convencion detectada -- " + c)
    if unpaired:
        print(f"\n  SIN PAR en el CPU ({len(unpaired)}): {', '.join(unpaired)}")
    huerfanas = [n for n in attrs if n not in usados and attrs[n].size == 14]
    if huerfanas:
        print(f"  arreglos del CPU de 14 entradas sin usar ({len(huerfanas)}): "
              f"{', '.join(sorted(huerfanas))}")

    print(f"\n{paired} tablas paired, {failures} difieren, "
          f"{len(unpaired)} unpaired  (worst {worst:.2e} in {worst_n})")
    return 1 if (failures or unpaired) else 0


if __name__ == "__main__":
    sys.exit(main())
