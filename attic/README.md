# attic — código muerto o de un solo uso

Nada de acá se importa desde `src/` ni desde `scripts/`, y **los imports no se
reescribieron** cuando el proyecto pasó a ser un paquete: estos archivos quedan
tal como llegaron de la máquina `udesa`. Si algo de acá se rescata, hay que
arreglarle los imports primero.

## Roto: depende de módulos que no existen

Restos de la rama de barrena/6-DOF, que vivía en otro árbol.

| archivo | importa | estado |
|---|---|---|
| `diag_eval_6dof.py` | `PolicyIterationBankedSpin` | el módulo no existe |
| `volcar_simetrico.py` | `aircraft.banked_spin_grumman`, `aircraft.spin_grumman` | no existen |

## Obsoleto: reemplazado por el CLI

| archivo | reemplazado por |
|---|---|
| `set_grilla_paper.py` | `symstall-train --grid` — editaba `main.py` en el lugar |
| `main.py.grilla_nueva` | preset `riley` de `train.py:GRIDS` |
| `main.py.bak_umbral2` | ídem, con el umbral `has_dived` viejo (−2°, hoy −1°) |

Los tres `main.py.*` son la evidencia de cómo se elegía la grilla antes: la
copia con la grilla nueva es la que entrenó la política principal, aunque el
`main.py` que quedó en disco tenía la del paper 1.

## Parches de un solo uso

`patch_alpha_final.py`, `patch_det.py`, `patch_esc.py`, `fix_familia.py`,
`fix_fig.py` — scripts que reescribían *otros* scripts con reemplazo de texto.
Ya aplicados; se guardan solo como registro de qué se cambió.

`completar_arm_pd.py`, `regen_orphans.py` — utilidades puntuales de limpieza de
datos.

## Otra línea de trabajo

`PPO-SymmetricStall.py`, `PPO-SymmetricStall-baseline.py` — los baselines de
PPO del paper 1. Necesitan `stable-baselines3`, que no está en las
dependencias.

`exportar_godot_8dof.py`, `volcar_tablas.py` — exportadores para el visualizador
en Godot y volcado de tablas.
