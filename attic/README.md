# attic — dead or single-use code

Nothing here is imported from `src/` or `scripts/`, and **the imports were not
rewritten** when the project became a package: these files are exactly as they
arrived from the `udesa` machine. If anything here is to be rescued, its
imports have to be fixed first.

## Broken: depends on modules that do not exist

Leftovers from the spin / 6-DOF branch, which lived in a different tree.

| file | imports | state |
|---|---|---|
| `diag_eval_6dof.py` | `PolicyIterationBankedSpin` | module does not exist |
| `volcar_simetrico.py` | `aircraft.banked_spin_grumman`, `aircraft.spin_grumman` | do not exist |

## Obsolete: superseded by the CLI

| file | superseded by |
|---|---|
| `set_grilla_paper.py` | `symstall-train --grid` — it edited `main.py` in place |
| `main.py.grilla_nueva` | the `riley` preset in `train.py:GRIDS` |
| `main.py.bak_umbral2` | the same, with the old `has_dived` threshold (−2°, now −1°) |

The three `main.py.*` files are the evidence of how the grid used to be
selected: the copy carrying the new grid is the one that trained the main
policy, even though the `main.py` left on disk carried the paper-1 grid.

## Single-use patches

`patch_alpha_final.py`, `patch_det.py`, `patch_esc.py`, `fix_familia.py`,
`fix_fig.py` — scripts that rewrote *other* scripts by text substitution.
Already applied; kept only as a record of what was changed.

`completar_arm_pd.py`, `regen_orphans.py` — one-off data-cleaning utilities.

## A different line of work

`PPO-SymmetricStall.py`, `PPO-SymmetricStall-baseline.py` — the PPO baselines
from paper 1. They need `stable-baselines3`, which is not among the
dependencies.

`exportar_godot_8dof.py`, `volcar_tablas.py` — exporters for the Godot
visualiser and for dumping tables.
