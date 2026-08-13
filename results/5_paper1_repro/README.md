# Paper 1's experiments, re-run against the corrected thrust model

Everything here is produced by `scripts/paper1/run.py` from the reference
policy `SymmetricStall_riley_56x81x80x41_thrust-riley.npz` (Riley grid,
50 policy steps, 4 h 19 on the 3090), with `THRUST_MODEL=riley` and the CG at
Riley's reference quarter-chord.

The experiments are paper 1's, unchanged in what they ask. What changed is the
plant they ask it of.

    python scripts/paper1/run.py --list          # the suite and what blocks what
    python scripts/paper1/run.py trajectories    # one experiment
    python scripts/paper1/run.py procedures held_pull procedure_figs

## Why the numbers moved

Paper 1's propeller was a constant `Kt`, calibrated so that full throttle
sustains level flight at 2 Vs. Riley supplies the real map in Appendix A, and
it **decays with airspeed**. The two agree to 9 % at the calibration point and
differ by **43 % at 0.95 Vs** — which is where a stall recovery happens. The
old model starved the engine exactly where it mattered, and every published
altitude loss is inflated by it.

The consequence is not a shift, it is a relocation of the problem:

| α₀ = 20° | Δh | |
|---|---|---|
| 0.85 Vs | **−9.35 m** | recovers |
| 0.90 Vs | −2.70 m | recovers |
| 0.95 Vs | +11.78 m | **climbs**, no manoeuvre |
| 1.00 Vs | +17.41 m | climbs |

Paper 1 evaluated on `V₀/Vs ∈ {0.90, 0.95, 1.00}` and called **(20°, 0.95)**
canonical, reporting −9.09 m there. With Riley's thrust that entry does not
descend at all: two of its three columns measured nothing. The evaluation band
therefore moves to **{0.80, 0.85, 0.90}** with the canonical entry at
**(20°, 0.85)**, which is the live region — and, at −9.35 m, numerically almost
the same case the paper thought it was reporting.

The band is not hardcoded: `STALL_VNORM_GRID`, `STALL_ALPHA_GRID`,
`STALL_CANON_VNORM` and the `STALL_IC_*` variables set it, and
`procedures.py` refuses to import if the canonical entry is not one of the
grid's own cells.

## What each experiment says

### The IC plane (`fig_optimal_ic_heatmap`, `ic_heatmap_dense.json`)

3315 rollouts over 13 α₀ × 51 V₀ × 5 power strategies. The isolines are almost
**vertical**: the altitude cost is set by entry airspeed and is nearly
independent of how deep the stall is. The recoverable region ends abruptly near
**0.92 Vs**; above it the aircraft flies away instead of stalling, and those
cells are masked rather than reported (a climb integrated over the horizon
would otherwise print as the largest "altitude loss" on the map).

### The canonical recovery (`fig_trajectories_procedures`)

DP optimum −9.35 m, CAA −23.45 m, FAA −27.80 m. The optimum unloads to
α ≈ 12° and then **rides the stall boundary at 14°** — the sliding mode. The
scripted pilots push to 7°, throwing away lift for a full second. Their loss
comes from over-unloading, not from applying power late.

### Procedures and pilot error (`fig_procedures`, `fig_pilot_sensitivity`)

- **Full throttle is optimal essentially everywhere**: the argmax commands
  δt = 1 in **99.0 %** of the state space and **99.7 %** inside the recovery
  window. This is the CAA-structure claim, and it survives the correction.
- **Power delay costs ≈ 10.7 m per second.**
- **The 2-second ramp alone costs 12.7 m** (−22.02 vs −9.35 at τ = 0) — more
  than a full second of complete delay. *How fast* the throttle is advanced
  matters more than *when* it starts moving.
- **The pull-up switch is the sharp axis**: late by 0.25 s → −11.93 m; by
  0.5 s → −22.75 m (×2.4); by 1 s → −50.79 m (×5.4).
- **The pull authority is the forgiving axis — if flown closed loop.** Clipping
  the policy's pull anywhere from −25° to −10° gives the same −9.35 m, because
  the optimum never asks for more than −12.8°. Flown **open loop** the same
  deflection has a narrow optimum at **−5°** (the equivalent control) and
  anything past −12.5° drives a **secondary stall** the aircraft does not
  recover from: −25° held costs −92 m.

### CAA vs FAA (`table_maneuvers.tex`, `table_caa_vs_faa.tex`)

At the canonical entry the optimum is **2.5× better than CAA**, and CAA is
**1.19× better than FAA**. The ratio worsens with entry airspeed (×1.86 at
0.80 Vs, ×5.76 at 0.90 Vs): the less altitude there is to lose, the more of it
the scripted procedure wastes. Flown with a full-deflection open-loop pull
instead of an α-hold, every entry re-stalls and none recovers (−92 to −105 m).

## Status of the suite

| experiment | state |
|---|---|
| `ic_heatmap`, `caa_ramp`, `switch_heatmap` | reproduced |
| `trajectories`, `maneuvers`, `caa_vs_faa` | reproduced |
| `procedures`, `held_pull` | reproduced |
| `robustness`, `rob_steady`, `rob_feasible` | reproduced |
| `rob_thrust` | **rewritten** — see below |
| `cg_sweep` | needs 7 GPU re-solves on the Riley grid |
| `dp_vs_ppo`, `q_values` | need a PPO baseline retrained on the corrected model |
| `mca` | needs an MCA-timestep policy on the Riley grid |
| `gamma_alpha`, `montecarlo` | generators still in `attic/regen_orphans.py` |

### Two traps removed on the way

**`rob_thrust` measured nothing.** It perturbed `THROTTLE_LINEAR_MAPPING` —
the constant Riley's model does not read. Under `THRUST_MODEL=riley` it would
have produced a flat table, to be read as "the model is insensitive to
thrust". The perturbation now goes through `THRUST_SCALE`, which multiplies
delivered thrust under either model, and the question becomes the one that
survives the correction: what an engine that underdelivers costs.

**`maneuvers` depended on a policy it does not need.** Its DP reference came
out of `mca_comparison.json`, so the manoeuvre table could not be built without
first training a second policy. It now computes the reference itself (9
rollouts, 15 s) when the cache is absent.

Both are the same failure as the thrust model itself: something that produces
plausible numbers instead of failing. `paths.load_policy()` is the guard
against the general case — it refuses to load any policy until the thrust model
has been chosen explicitly, and cross-checks it against the `.npz` metadata.
