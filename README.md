# Symmetric stall recovery — optimal DP over Riley's aerodynamics

Optimal recovery from a symmetric stall in a Grumman AA-1, solved by GPU policy
iteration on a 4-DOF model with Riley's (1985) aerodynamic tables. The central
result: the optimal policy is a **bound**, not a flyable procedure, and the
penalty paid by the CAA/FAA procedures is explained almost entirely by the
power ramp, not by the elevator trigger.

**Status:** the manuscript is not written yet. The results, with their numbers
and their interpretation, live in [`results/README.md`](results/README.md) —
that is the material the paper is written from.

## Installation

Requires an NVIDIA GPU (the solver is a CUDA kernel compiled through CuPy) and
Python ≥ 3.10.

```bash
uv venv --python 3.14 .venv
uv pip install -e .
```

CuPy's `[ctk]` extra is included on purpose: without the CUDA headers the
kernel fails at runtime with `Failed to find CUDA headers`.

## Training

```bash
symstall-train                    # riley grid + riley thrust = the paper's setup
symstall-train --grid paper1      # paper-1 grid, to separate effects
symstall-train --help
```

| grid | dimensions (γ, V, α, q) | states | time (RTX 3090) |
|---|---|---|---|
| `riley` (default) | 56 × 81 × 80 × 41 | 14,878,080 | 47 iter, **4 h 18** |
| `paper1` | 56 × 41 × 60 × 41 | 5,648,160 | 23 iter, **43 min** |

The policy is written to `data/policies/`, with its configuration encoded in
the filename and stamped inside the `.npz` (the `run_metadata` key).

**The `riley` grid is the default throughout the software**, not just in the
CLI: `setup_symmetric_stall_experiment()` with no arguments builds it, so the
figure scripts that rebuild the state space use it too. Every run logs the grid
it built, and `assert_grid_matches()` aborts if a loaded policy was not trained
on the grid in use — that pairing used to happen silently.

**GPU memory:** ~430 MB for the `riley` grid (every array is O(states), none is
O(states × actions)), so it fits on any modern card. Verified on an 8 GB RTX
3070 Laptop. What scales badly is **time**: on a 3070 expect on the order of
11–15 h for the `riley` grid.

After training, the terminal policy has to be filled in:

```bash
python scripts/figures/fill_terminal_policy.py data/policies/<file>.npz
```

This overwrites the `.npz` and leaves the original as `.npz.raw`. The fill is
worth between 10 cm and 74 cm of lost altitude depending on the initial
airspeed (see `results/README.md`).

## Reproducing the figures

Every script runs **from the repository root** and takes the policy as its
first argument. Figures are written to `results/`.

```bash
python scripts/figures/maniobras_086.py data/policies/<file>.npz 0.86
```

| script | what it does |
|---|---|
| `canonica_filled_vs_raw.py <raw> <filled> V0...` | filled vs raw trajectories |
| `barrido_v0.py <raw> <filled>` | the `h_min(V0)` curve |
| `familia2.py <npz> <out.png> V0...` | overlaid trajectories |
| `maniobras_086.py <npz> <V0> [alpha0]` | optimum vs CAA vs FAA |
| `ablacion.py <npz> <V0>` | trigger × power factorial |
| `potencia_pura.py <npz> <V0>` | same elevator, different ramp |
| `potencia_riley.py <npz> <V0> <tau_fig> <taus...>` | with Riley's engine lag |
| `piloto_realista.py <npz> <V0> <tau_h> <taus_m...>` | human delay + engine |

## The configuration that used to travel hidden

This is the main fix from the migration. A run's configuration arrived through
three invisible channels, and the `.npz` recorded none of them:

| before | now |
|---|---|
| `THRUST_MODEL=riley` as an environment variable, with the code defaulting to `paper1` — forget it and the canonical recovery gave −13.3 m instead of −6.8 | `--thrust`, defaulting to `riley`, and it warns when running on the code default |
| the grid was selected by **editing `main.py`** with `set_grilla_paper.py`, leaving the comments describing one grid while the code ran another | `--grid`, two named presets in `train.py:GRIDS` |
| `CG_AFT_M`/`CG_RIGHT_M`/`CG_BELOW_M`, read **at import time** by the plant | `--cg-aft`/`--cg-right`/`--cg-below`, applied before the import (see `runconfig.py`) |
| every policy was saved as `SymmetricStall_policy.npz`, overwriting the previous one | the filename encodes grid + thrust + CG, and the configuration goes inside the `.npz` |

Policies trained before this change are anonymous: loading one makes the solver
warn that it does not record its configuration.

## Layout

```
src/symmetric_stall/     the installable package
  policy_iteration.py    the solver: CUDA kernel + policy iteration
  train.py               grids, training, simulation, heatmaps
  cli.py                 the CLI (sets the environment BEFORE importing the plant)
  runconfig.py           thrust and CG: apply, describe, stamp
  aircraft/              the plant: Grumman AA-1 and Riley's tables
  utils/                 barycentric interpolation, recovery monitor
  analysis/              policy metrics, dt ablation
scripts/
  figures/               the 4-DOF paper figures
  verify/                checks against Riley's tables
  paper1/                scripts from the previous paper (PPO vs PI, CG sweeps)
results/                 generated figures + README.md with the numbers
data/policies/           the .npz files (out of git: 119 MB each, regenerable)
refs/                    reference PDFs (out of git)
logs/                    logs from the 3090 runs
attic/                   dead or single-use code — see attic/README.md
```

## Open items

1. **Write the manuscript.** The material is in `results/README.md`.
2. **Analyse the paper-1 grid run**, the one that separates the effect of the
   grid from the effect of the thrust model. Still unanalysed — and note that
   the `paper1` grid carries two known pathologies of its own (half its alpha
   axis runs on clamped coefficients, and its +20° ceiling is non-absorbing),
   so it is not a clean grid-only comparison.
3. **Riley's `tau_e`**: it appears neither in Appendix A nor in the symbol list
   with a figure. It would have to be tracked down in the report's references.
   The absolute altitude loss depends strongly on it; the comparison between
   arms barely does (8%).
4. **The 6-DOF non-absorbing box**: 4.99% of its cells are worth less than
   crashing. It does not affect the 4-DOF model in this repo (alpha ceiling at
   +40).

### How a run reports itself

`run()` saves only when given a path; the single write happens in
`train_or_load_policy`, together with the metadata. It used to always call
`save()`, which falls back to the current working directory when given `None`,
so every policy was written twice — once anonymously into the CWD. That is
where the stray `SymmetricStall_policy.npz` at the root of the udesa tree came
from, byte-identical to the one under `results/politicas/`.

Policy iteration stops when the number of states still changing their optimal
action falls under a chattering tolerance of `n_states*1e-4` (1,487 on the
`riley` grid). That is a declared approximation, not exact convergence, so the
`.npz` records what actually happened:

| key | meaning |
|---|---|
| `n_policy_steps` | policy-iteration steps taken |
| `final_residual` | last evaluation residual |
| `n_states_chattering` | states still changing action when it stopped |
| `chattering_tolerance` | the threshold that allowed it to stop |

A run that stops with `n_states_chattering > 0` says *"converged with N states
still changing action"*, not *"converged optimally"*.

## Physical constants

Every constant traces to Riley's Table I (PDF pp. 29-30, printed 25-26), with
exact conversions: 1 lb = 0.45359237 kg, 1 ft = 0.3048 m, 1 slug-ft2 =
1.35581795 kg-m2.

| code | Table I | value |
|---|---|---|
| `MASS` | Nominal test weight, lb | 1577 → 715.3152 kg |
| `WING_SURFACE_AREA` | Wing: Area, ft² | 98.11 → 9.114717 m² |
| `CHORD` | Wing: Mean aerodynamic chord, ft | 4.00 → 1.2192 m |
| `WING_SPAN` | Overall dimensions: Span, ft | 24.46 → 7.455408 m |
| `I_YY` | Moments of inertia: Iy, slug-ft² | 738 → 1000.5936 kg·m² |

`MASS` and `CHORD` were mistranscribed as 715.21 and 1.22 until 2026-08-12;
the values above were confirmed against the page. They are duplicated in the
CUDA kernel, so both copies move together — `verificar_cpu_vs_kernel.py` is
what catches it if they ever do not.

Riley's Table I contradicts itself on span: *Overall dimensions* says 24.46 ft
and *Wing* says 26.46 ft. The aspect ratio settles it — Riley states AR = 6.10,
and b²/S gives 6.098 for 24.46 against 7.136 for 26.46. The span does not enter
the 4-DOF longitudinal equations at all; it only appears in the lateral moment
increments of the CG transfer, which this model discards.

Table I also gives `Ixz = 0` and the CG at 0.25 c̄ — the first confirms by data
that the inertia cross-terms the 4-DOF reduction drops were already zero, the
second is the reference every aerodynamic coefficient is measured about.

## Provenance

Migrated from `nromero@udesa:/home/nromero/stall-spin-recovery-dp` (the machine
with the 3090, where every result was produced). Code, figures and logs were
brought over; trained policies and virtualenvs were not.
