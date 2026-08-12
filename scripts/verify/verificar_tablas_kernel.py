"""Compare the CUDA kernel's tables literal by literal against the CPU's.

The kernel carries Riley's tables DUPLICATED as literals inside the CUDA
source, because a __device__ const float cannot be fed from numpy without
paying an indirection per read. That duplication has already cost two bugs: a
degree-to-radian conversion stolen from the neighbouring table, and a CM_DE at
C_T=0.5 differing in the fourth digit. Neither shows up in
verificar_cpu_vs_kernel.py, which compares the final derivatives using the
median and therefore averages away exactly what needs to be seen.

This compares the 26 arrays entry by entry. It does not measure physics: it
measures that the two copies say the same thing.

Usage:  PYTHONPATH=. python verificar_tablas_kernel.py
"""
import re
import sys

import numpy as np

from symmetric_stall import policy_iteration as PI
from symmetric_stall.aircraft.symmetric_full_grumman import SymmetricFullGrumman

# kernel name -> CPU attribute. Declared by hand on purpose: deriving it by a
# rule (TBL->TABLE) would make a misspelt name on one side pair up with the
# wrong one, which is exactly the bug being hunted.
PAIRS = {
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
    # the alpha breakpoints: if these shift, ALL the tables shift
    "CL_A_TBL": "_CL_O_ALPHA_RAD",
}


def kernel_tables(source: str) -> dict:
    """Extract the __device__ const float NAME[N] = { ... } from the CUDA source."""
    pattern = re.compile(
        r"__device__\s+const\s+float\s+([A-Z_0-9]+)\s*\[\s*(\d+)\s*\]\s*=\s*\{([^}]*)\}",
        re.S)
    out = {}
    for name, n, body in pattern.findall(source):
        vals = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?f?",
                                             body.replace("f", ""))]
        if len(vals) != int(n):
            print(f"  !! {name}: declara {n} entradas y tiene {len(vals)}")
        out[name] = np.array(vals, dtype=np.float64)
    return out


def main() -> int:
    source = PI.PolicyIterationStall.KERNEL_SOURCE if hasattr(
        PI.PolicyIterationStall, "KERNEL_SOURCE") else open(
            "PolicyIteration.py").read()
    K = kernel_tables(source)
    a = SymmetricFullGrumman()

    print(f"tables found in the CUDA source: {len(K)}")
    missing = [k for k in PAIRS if k not in K]
    if missing:
        print(f"  !! declared in the map but absent from the kernel: {missing}")
    orphans = [k for k in K if k not in PAIRS]
    if orphans:
        print(f"  note: in the kernel with no declared pair: {orphans}")
    print()

    worst_name, worst = None, 0.0
    failures = 0
    for kn, cn in sorted(PAIRS.items()):
        if kn not in K:
            continue
        kv = K[kn]
        cv = np.asarray(getattr(a, cn), dtype=np.float64)
        if kv.shape != cv.shape:
            print(f"  FORMA  {kn:<20} kernel {kv.shape} vs CPU {cv.shape}")
            failures += 1
            continue
        denom = np.maximum(np.abs(cv), 1e-9)
        rel = np.abs(kv - cv) / denom
        i = int(rel.argmax())
        # float32 rounds to ~1e-7; above 1e-6 it is transcription, not format
        status = "ok" if rel[i] <= 1e-6 else "DIFIERE"
        if rel[i] > 1e-6:
            failures += 1
            print(f"  {status:<8}{kn:<20} entrada {i:2d}: kernel {kv[i]:+.6f} "
                  f"vs CPU {cv[i]:+.6f}  rel {rel[i]:.2e}")
        if rel[i] > worst:
            worst, worst_name = rel[i], kn

    print()
    if failures == 0:
        print(f"ALL {len(PAIRS)} TABLES MATCH (worst {worst:.2e} in {worst_name})")
        return 0
    print(f"{failures} TABLE(S) DO NOT MATCH (worst {worst:.2e} in {worst_name})")
    return 1


if __name__ == "__main__":
    sys.exit(main())
