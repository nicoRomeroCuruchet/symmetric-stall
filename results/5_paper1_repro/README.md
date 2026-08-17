# Paper 1's experiments, re-run against the corrected thrust model

Everything here is produced by `scripts/paper1/run.py` from the reference
policy `SymmetricStall_riley_56x81x80x41_thrust-riley.npz` (Riley grid,
50 policy steps, 4 h 19 on the 3090), with `THRUST_MODEL=riley` and the CG at
Riley's reference quarter-chord.

The experiments are paper 1's, unchanged in what they ask. What changed is the
plant they ask it of.

> **The engine here is ideal: throttle commands take effect instantly.** Riley's
> aircraft has a first-order lag, eq. (A4), with τ_e ≈ 0.85 s — a constant he
> defines and never tabulates, recovered from his own figures 16 and 18. The
> same suite flown on the real engine is in **`../6_riley_engine/`**, and that is
> the directory to cite for any ABSOLUTE altitude loss: the canonical entry
> reads −9.35 m here and −19.79 m there. Differences between arms move less than
> 12 % and can be cited from either. This directory is the τ_e = 0 end of the
> sensitivity analysis, not superseded work.
>
> Three figure pairs here are **not** τ_e = 0 results and are kept only as the
> working record of how the constant was pinned down:
> `fig_trajectories_procedures_tau050` (an intermediate guess at τ_e, made
> before the identification), `fig_trajectories_procedures_tau085` (an early
> render, produced before figures carried the engine stamp) and
> `fig_engine_tau_id` (the identification itself). For all three the
> authoritative copies live in `../6_riley_engine/`. Do not cite these.

    python scripts/paper1/run.py --list          # the suite and what blocks what
    python scripts/paper1/run.py trajectories    # one experiment
    python scripts/paper1/run.py procedures held_pull procedure_figs
    python scripts/paper1/run.py ic_figs         # redraw maps from the cache

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

Nothing is hardcoded: `STALL_VNORM_GRID`, `STALL_ALPHA_GRID`,
`STALL_CANON_VNORM` and the `STALL_IC_*` variables set the bands, and
`procedures.py` refuses to import if the canonical entry is not one of the
grid's own cells.

## What each experiment says

### The IC plane — `fig_optimal_ic_heatmap`, `ic_heatmap_dense.json`

The sweep runs the **whole airspeed axis the policy was solved on, 0.40 to
0.90 Vs**: 13 α₀ × 101 V₀ × 5 power strategies, 6565 rollouts, and 11,817
across the three IC sweeps. Every one recovers — inside this band even the
worst power timing gets the aircraft back, and what separates the strategies
is only how much altitude they spend.

The isolines are **vertical**: the cost is set by entry airspeed and is nearly
independent of how deep the stall is. It is close to linear, about **1.75 m
per hundredth of Vs**, from −85.75 m at 0.40 to −2.70 m at 0.90, easing off
only above 0.85. The recoverable region ends near **0.92 Vs**; above it the
aircraft flies away instead of stalling, and those cells are masked rather
than reported — a climb integrated over the horizon would otherwise print as
the largest "altitude loss" on the map.

*Caveat below 0.58 Vs.* Riley fits thrust as a linear T(V) and declares it
appropriate above 60 ft/s = 0.579 Vs, saying it **under**estimates thrust
below that. The bottom third of the sweep is therefore conservative: those
losses are upper bounds.

### The canonical recovery — `fig_trajectories_procedures`

DP optimum −9.35 m, CAA −23.45 m, FAA −27.80 m. The optimum unloads to
α ≈ 12° and then **rides the stall boundary at 14°** — the sliding mode. The
scripted pilots push to 7°, throwing away lift for a full second. Their loss
comes from over-unloading, not from applying power late.

### Procedures and pilot error — `fig_procedures`, `fig_pilot_sensitivity`

- **Full throttle is optimal essentially everywhere**: δt = 1 in **99.0 %** of
  the state space and **99.7 %** inside the recovery window. The CAA structure
  survives the correction.
- **Power delay costs ≈ 10.7 m per second.**
- **The 2-second ramp alone costs 12.7 m** (−22.02 vs −9.35 at τ = 0) — more
  than a full second of complete delay. *How fast* the throttle is advanced
  matters more than *when* it starts moving.
- **The pull-up switch is the sharp axis.** Late by 0.25 s → −11.93 m; by
  0.5 s → −22.75 m (×2.4); by 1 s → −50.79 m (×5.4). Over the IC plane
  (`fig_switch_delay_ic`) the penalty **grows with entry airspeed** — +30 m at
  0.45 Vs against +40 m at 0.85 Vs — and at 0.25 s the map is blank across the
  whole band. The cliff is between 0.5 and 1 s, not before.
- **The pull authority is the forgiving axis — if flown closed loop.** Clipping
  the policy's pull anywhere from −25° to −10° gives the same −9.35 m, because
  the optimum never asks for more than −12.8°. Flown **open loop** the same
  deflection has a narrow optimum at **−5°** (the equivalent control) and
  anything past −12.5° drives a **secondary stall** the aircraft does not
  recover from: −25° held costs −92 m and reaches α = 36°.

### CAA vs FAA — `table_maneuvers.tex`, `fig_procedures_ic_heatmap`

At the canonical entry the optimum is **2.5× better than CAA**, and CAA is
**1.19× better than FAA**.

Over the full airspeed axis the penalty is **additive, not multiplicative**:
CAA +10 to +12 m, FAA +15 to +25 m, power-delayed +30 to +34 m, nearly flat
across the band. The ×5.76 ratio that appears at 0.90 Vs is an artefact of
dividing by an optimum whose own loss tends to zero there. *A procedure costs
a fixed number of metres* — which is also the form a pilot can use.

### Mass and CG — `fig_robustness_matrix`, `fig_cg_reach`

The matrix is **13 × 14 cells on a uniform 2.5 % step**, mass 0.85–1.15 and
x_cg from 0.15 to 0.475 c̄, at the canonical IC. It reaches past the point
where the airframe itself breaks, and draws both regime edges.

- **Mass dominates**: 17.0 m across its range against 3.8 m for CG. A 15 %
  mass change is 107 kg — one adult plus baggage, or 149 L of avgas.
- **Mass sensitivity is not constant.** It climbs from 0.62 m to 2.54 m per
  2.5 % step, light end to heavy end: a factor of 4. Overloading is much worse
  than extrapolating linearly from nominal would suggest.
- **Aft CG helps**, monotonically, ~0.6 m per step — until the heavy corner.
- **Separability breaks in exactly one place.** `excess = f(mass) + g(CG)` fits
  to **0.49 m** over the operational window, degrades to **2.34 m** over the
  extended plane, and recovers to 1.18 m once the divergent columns are
  dropped. At +15 % mass the CG effect reverses past +17.5 % chord, from −0.24
  to +1.54 m per step; at −15 % it stays flat and favourable throughout.
  **Heavy and aft is the one corner where the two axes stop being separable.**

`fig_cg_reach` pushes the CG to 0.55 c̄ and separates the airframe from the
recovery. Linearising d(α, q)/dt with the controls frozen, the short-period
mode is a stable complex pair to ~0.40 c̄, splits into real roots, and turns
**divergent at 0.453 c̄** (+0.85 /s at 0.50 c̄ — a doubling time of 0.8 s). The
closed-loop recovery goes straight through it, improving all the way, and the
elevator works *less*: total travel falls from 51° to 12°. A policy stepping
at 100 Hz with full authority holds a mildly unstable airframe easily — this
is relaxed static stability arrived at by accident. The intuition that an aft
CG should ruin the aeroplane is right about the airframe and wrong about the
recovery.

Two caveats. `Cm_q` and `Cm_α̇` are **not** re-referenced to the displaced CG,
so the model over-damps at aft CG by roughly `dx/l_t` — about 2.5 % at 7 % of
chord, growing to ~10 % at the far end, where it flatters stability. And an
open-loop pilot is not a 100 Hz controller: this shows the instability is
inside the optimum's authority, not that it is harmless.

## What "recovered" means, precisely

The stopping rule has **two clauses** (`utils/recovery.py`), and they are not
interchangeable. Over the matrix's 1638 rollouts:

| | |
|---|---|
| crossed γ ≥ 0 — back to level | **739** |
| settled below level, γ̇ → 0 — convergence clause | **713** |
| never closed | **186** |

The 713 approach level **asymptotically from below** and stay there, in a
quasi-steady descent of centimetres per second (γ ≈ −0.04°, 2.4 cm/s in the
case checked). A crossing test alone would never detect them and their
altitude loss would be undefined, growing with the integration horizon. The
two clauses cannot be confused: a genuine crossing happens at |γ̇| ≈ 0.5–1.2
°/s, the asymptotic ones sit at ≤ 5×10⁻⁴ °/s.

So "every configuration recovers" is true operationally but hides that **fewer
than half reach γ ≥ 0**. That is a property of the measurement, not of the
physics — 2.4 cm/s is not an aircraft falling — but it has to be stated.

The 186 that never close are **not failures either**: all sit aft of +5 % chord
and rise monotonically with it (2 at +5 %, 50 at +22.5 %), and the light-
aircraft ones climb away at +36 m rather than diving. There are **no crashes**
anywhere in the suite.

### The marks on the matrix

`*` = did not return to level; `†` = closed still stalled. At the canonical IC
8 cells carry `†` and 4 carry `*`, **all in the aft corner, none in the
certified band**. The threshold for `†` is one degree above the higher of the
stall boundary and the reference cell's own close, which makes it structurally
impossible to flag the reference — a bare α_s = 14° flagged 26 of 49 cells
including the nominal one, because the optimum *ends* pinned to the boundary
at 13.7–14.5° by design. What the mark is for reaches 35°.

## Status of the suite

| experiment | state |
|---|---|
| `ic_heatmap`, `caa_ramp`, `switch_heatmap`, `ic_figs` | reproduced, 0.40–0.90 Vs |
| `trajectories`, `maneuvers`, `caa_vs_faa` | reproduced |
| `procedures`, `held_pull`, `procedure_figs` | reproduced |
| `robustness`, `rob_steady`, `rob_feasible` | reproduced, 13 × 14 |
| `cg_reach` | new — the airframe's own limit |
| `rob_thrust` | **rewritten**, see below |
| `cg_sweep` | needs 7 GPU re-solves on the Riley grid |
| `dp_vs_ppo`, `q_values` | need a PPO baseline retrained on the corrected model |
| `mca` | needs an MCA-timestep policy on the Riley grid |
| `gamma_alpha`, `montecarlo` | generators still in `attic/regen_orphans.py` |

### Traps removed on the way

**`rob_thrust` measured nothing.** It perturbed `THROTTLE_LINEAR_MAPPING` —
the constant Riley's model does not read. Under `THRUST_MODEL=riley` it would
have produced a flat table, to be read as "the model is insensitive to
thrust". The perturbation now goes through `THRUST_SCALE`, which multiplies
delivered thrust under either model.

**`maneuvers` depended on a policy it does not need.** Its DP reference came
out of `mca_comparison.json`, so the manoeuvre table could not be built without
first training a second policy. It now computes the reference itself.

**Bands and keys were duplicated.** `gen_table_caa_vs_faa.py` and
`paper_robustness.py` each carried their own copy of paper 1's evaluation
band, so moving it left them indexing cells that no longer existed. The
`gamma_alpha` and `montecarlo` generators still sampled 0.90–1.00 Vs, mostly
above the recoverable edge; both now sample the live band. Matrix cell keys go
through one `cell_key()` helper at three decimals — at 2.5 % steps a 2-decimal
key wrote 0.875 as `m0.88`, misnaming the aircraft it held.

All of these are the same failure as the thrust model itself: code that
produces plausible numbers instead of failing. `paths.load_policy()` is the
guard against the general case — it refuses to load any policy until the
thrust model has been chosen explicitly, and cross-checks it against the
configuration stamped in the `.npz`.
