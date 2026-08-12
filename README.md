# Symmetric stall recovery — DP óptimo sobre la aerodinámica de Riley

Recuperación óptima de pérdida simétrica en un Grumman AA-1, resuelta por
iteración de políticas sobre GPU en un modelo 4-DOF con las tablas
aerodinámicas de Riley (1985). El resultado central: la política óptima es una
**cota**, no un procedimiento volable, y la penalización de los procedimientos
CAA/FAA se explica casi enteramente por la rampa de potencia, no por el
gatillo del elevador.

**Estado:** el manuscrito todavía no está escrito. Los resultados, con sus
números y su interpretación, están en [`results/LEEME.md`](results/LEEME.md) —
es el material del que sale el paper.

## Instalación

Requiere una GPU NVIDIA (el solver es un kernel CUDA compilado con CuPy) y
Python ≥ 3.10.

```bash
uv venv --python 3.14 .venv
uv pip install -e .
```

El extra `[ctk]` de CuPy viene incluido a propósito: sin los headers de CUDA,
el kernel falla en tiempo de ejecución con `Failed to find CUDA headers`.

## Entrenar

```bash
symstall-train                    # grilla riley + empuje riley = lo del paper
symstall-train --grid paper1      # grilla del paper 1, para aislar efectos
symstall-train --help
```

| grilla | dimensiones (γ, V, α, q) | estados | tiempo (RTX 3090) |
|---|---|---|---|
| `riley` (default) | 56 × 81 × 80 × 41 | 14.878.080 | 47 iter, **4 h 18** |
| `paper1` | 56 × 41 × 60 × 41 | 5.648.160 | 23 iter, **43 min** |

La política sale en `data/policies/`, con la configuración codificada en el
nombre y estampada adentro del `.npz` (clave `run_metadata`).

**La grilla `riley` es el default en todo el software**, no sólo en el CLI:
`setup_symmetric_stall_experiment()` sin argumentos la construye, así que los
scripts de figuras que reconstruyen el espacio de estados también la usan. Cada
corrida loguea la grilla que construyó, y `assert_grid_matches()` aborta si una
política cargada no fue entrenada con la grilla que se está usando — antes ese
emparejamiento era silencioso.

**Memoria de GPU:** ~430 MB para la grilla `riley` (todos los arrays son
O(estados), ninguno O(estados × acciones)), así que entra en cualquier placa
moderna. Verificado en una RTX 3070 Laptop de 8 GB. Lo que escala mal es el
**tiempo**: en una 3070 esperá del orden de 11–15 h para la grilla `riley`.

Después de entrenar hay que rellenar la política terminal:

```bash
python scripts/figures/fill_terminal_policy.py data/policies/<archivo>.npz
```

Sobrescribe el `.npz` y deja el original como `.npz.raw`. El relleno vale entre
10 cm y 74 cm de altura perdida según la velocidad inicial (ver `LEEME.md`).

## Reproducir las figuras

Todos los scripts se corren **desde la raíz del repo** y toman la política como
primer argumento. Las figuras salen en `results/`.

```bash
python scripts/figures/maniobras_086.py data/policies/<archivo>.npz 0.86
```

| script | qué hace |
|---|---|
| `canonica_filled_vs_raw.py <raw> <filled> V0...` | trayectorias rellenada vs cruda |
| `barrido_v0.py <raw> <filled>` | la curva `h_min(V0)` |
| `familia2.py <npz> <salida.png> V0...` | trayectorias superpuestas |
| `maniobras_086.py <npz> <V0> [alpha0]` | óptimo vs CAA vs FAA |
| `ablacion.py <npz> <V0>` | factorial gatillo × potencia |
| `potencia_pura.py <npz> <V0>` | mismo elevador, distinta rampa |
| `potencia_riley.py <npz> <V0> <tau_fig> <taus...>` | con el motor de Riley |
| `piloto_realista.py <npz> <V0> <tau_h> <taus_m...>` | retardo humano + motor |

## La configuración que antes viajaba escondida

Este es el arreglo principal de la migración. La configuración de una corrida
entraba por tres canales invisibles, y el `.npz` no registraba ninguno:

| antes | ahora |
|---|---|
| `THRUST_MODEL=riley` como variable de entorno, default del código `paper1` — si te olvidabas, la canónica daba −13.3 m en vez de −6.8 | flag `--thrust`, default `riley`, y avisa si corrés con el default del código |
| la grilla se elegía **editando `main.py`** con `set_grilla_paper.py`, dejando los comentarios describiendo una grilla y el código corriendo otra | flag `--grid`, dos presets nombrados en `train.py:GRIDS` |
| `CG_AFT_M`/`CG_RIGHT_M`/`CG_BELOW_M`, leídas **al importar** la planta | flags `--cg-aft`/`--cg-right`/`--cg-below`, aplicadas antes del import (ver `runconfig.py`) |
| todas las políticas se guardaban como `SymmetricStall_policy.npz` y se pisaban entre sí | el nombre codifica grilla + empuje + CG, y la config va adentro del `.npz` |

Las políticas entrenadas antes de este cambio son anónimas: al cargarlas, el
solver avisa que no registran su configuración.

## Estructura

```
src/symmetric_stall/     el paquete instalable
  policy_iteration.py    el solver: kernel CUDA + iteración de políticas
  train.py               grillas, entrenamiento, simulación, heatmaps
  cli.py                 el CLI (parsea y fija el entorno ANTES de importar la planta)
  runconfig.py           empuje y CG: aplicar, describir, estampar
  aircraft/              la planta: Grumman AA-1 y tablas de Riley
  utils/                 interpolación baricéntrica, monitor de recuperación
  analysis/              métricas de política, ablación de dt
scripts/
  figures/               las figuras del paper 4-DOF
  verify/                verificaciones contra las tablas de Riley
  paper1/                scripts del paper anterior (PPO vs PI, barridos de CG)
results/                 figuras generadas + LEEME.md con los números
data/policies/           los .npz (fuera de git: 119 MB c/u, regenerables)
refs/                    PDFs de referencia (fuera de git)
logs/                    logs de las corridas de la 3090
attic/                   código muerto o de un solo uso — ver attic/README.md
```

## Pendientes

1. **Escribir el manuscrito.** El material está en `results/LEEME.md`.
2. **Analizar la corrida de la grilla del paper 1**, que es la que separa el
   efecto de la grilla del efecto del empuje. Todavía sin analizar.
3. **El valor de `tau_e` de Riley**: no está en el Apéndice A ni en la lista de
   símbolos con cifra. Habría que buscarlo en las referencias del informe. La
   pérdida absoluta depende fuerte de él; la comparación entre brazos no (8 %).
4. **La caja no absorbente del 6-DOF**: el 4,99 % de sus celdas vale menos que
   estrellarse. No afecta al 4-DOF de este repo (techo de α en +40).

## Procedencia

Migrado desde `nromero@udesa:/home/nromero/stall-spin-recovery-dp` (la máquina
con la 3090, donde se corrieron todos los resultados). Se trajo el código, las
figuras y los logs; no las políticas entrenadas ni los entornos virtuales.
