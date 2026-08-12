# data/policies

Acá van los `.npz` de políticas entrenadas. **Están fuera de git** (119 MB cada
una) — este archivo existe para que el directorio se cree al clonar.

Se regeneran con `symstall-train` (ver el README de la raíz). Las que
produjeron los resultados actuales quedaron en la máquina de la 3090:

    nromero@udesa:/home/nromero/stall-spin-recovery-dp/results/politicas/

| archivo | grilla | qué es |
|---|---|---|
| `SymmetricStall_alpha_m10_40_FILLED.npz` | 56×81×80×41 | la de todas las figuras del paper |
| `SymmetricStall_alpha_m10_40_RAW.npz` | ídem | la misma, antes del relleno terminal |
| `SymmetricStall_policy.npz` | 56×41×60×41 | grilla del paper 1 con empuje Riley, sin analizar |

Las tres son **anónimas**: se entrenaron antes de que el `.npz` registrara su
configuración, así que al cargarlas el solver avisa que no sabe con qué modelo
de empuje ni con qué CG se hicieron. Por el `LEEME.md` sabemos que fue
`THRUST_MODEL=riley` y CG sin corrimiento.
