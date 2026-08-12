# data/policies

Trained policy `.npz` files go here. They are **out of git** (119 MB each) —
this file exists so the directory is created on clone.

They are regenerated with `symstall-train` (see the root README). The ones that
produced the current results stayed on the 3090 machine:

    nromero@udesa:/home/nromero/stall-spin-recovery-dp/results/politicas/

| file | grid | what it is |
|---|---|---|
| `SymmetricStall_alpha_m10_40_FILLED.npz` | 56×81×80×41 | the one behind every figure in the paper |
| `SymmetricStall_alpha_m10_40_RAW.npz` | same | the same policy before the terminal fill |
| `SymmetricStall_policy.npz` | 56×41×60×41 | paper-1 grid with Riley thrust, still unanalysed |

All three are **anonymous**: they were trained before the `.npz` recorded its
configuration, so loading them makes the solver warn that it does not know
which thrust model or which CG produced them. From `results/README.md` we know
it was `THRUST_MODEL=riley` with no CG offset.
