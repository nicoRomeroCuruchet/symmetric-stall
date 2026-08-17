# The same experiments, flown on Riley's actual engine

Everything in `5_paper1_repro/` was produced on an aircraft whose throttle
takes effect instantly. Riley's does not. This directory is the whole suite
re-run with his first-order engine lag, eq. (A4), in the loop:

    python scripts/paper1/run.py --engine-tau 0.85 --out results/6_riley_engine \
        procedures maneuvers trajectories held_pull procedure_figs caa_vs_faa \
        ic_heatmap caa_ramp ic_figs switch_heatmap \
        robustness rob_steady rob_feasible cg_reach riley_coeffs

Same policy, same grids, same experiments. Only the engine changed, and only in
the evaluation — the policy is still the one solved against an ideal engine.
Every figure carries the engine in its lower-right corner and every JSON carries
it in a `run_config` block, so no artifact here can be confused with one from
the directory next door.

## Where τ_e = 0.85 s comes from

Riley defines the constant and never tabulates it. It is the **only** constant
of appendix A he leaves out: (A3) prints 0.65 and 0.35, (A9) prints the twelve
entries of the T₀/T₁ table, (A12) prints eighteen more, (A4) prints nothing.

It is recoverable anyway, because he publishes throttle chops with the engine
response beside them. Appendix A closes the loop between the two panels:

    (A4)   δ_t  = 1/(τ_e s + 1) δ_t,c          ← the unknown
    (A3)   δ_t' = 0.65 δ_t + 0.35
    (A12)  N    = N₀(δ_t') + N₁(δ_t') V + N₂(δ_t') V²

so `N(t)` is an observation of `δ_t(t)` and τ_e is the only free parameter.
Digitising the traces and fitting gives **0.79 s** on fig. 16(a), **0.82 s** on
16(b) and **0.97 s** on 18(b); 0.85 s is the centre. The fit is well
conditioned — on 16(a) the residual is 77 rpm at the optimum against 139 rpm at
τ_e = 0.1 s and 270 rpm at 3 s.

Four checks that the digitiser reads the page rather than itself, each against a
source that is not the plot: it recovers a **1.83 s** chop where Riley's *text*
says the chops "occurred in 2 sec"; it reads **120.0 ft/s** at t = 0 where the
*caption* states V = 120 ft/sec; the (A3)+(A12) chain predicts **2387 rpm** at
the trim throttle where the figure starts at **2400**; and **1135 rpm** at
closed throttle where the figure floors at **1100**.

### How good is it

The time histories are published **as plots only**. Tables I–VIII carry
aerodynamic coefficients, airfoil coordinates, hinge moments and stability
comparisons; none of them is a time history. So the trace is a reconstruction of
a drawing, and the question worth answering is how much of one.

Not the raster: at 400 dpi a pixel is worth 48 ms and 16 rpm. What limits the
reading is the **printed line, 7 px thick = 113 rpm** — there is no value of N,
there is a band. Perturbing each thing the reconstruction could get wrong
(`identificar_tau_motor.py --uncertainty`):

| perturbation | τ_e |
|---|---|
| none (as fitted) | 0.79 |
| **N up half a line width (+57 rpm)** | **1.00** |
| **N down half a line width (−57 rpm)** | **0.62** |
| V misread by ±5 ft/s | 0.68 / 0.91 |
| time axis ±1 % | 0.79 / 0.80 |
| fit window 8 s / 20 s | 0.80 / 0.79 |

The geometry is solid; where the centre of a thick line is placed is not. So the
honest number is **τ_e = 0.8 ± 0.2 s**, with 0.85 inside it — and the three
independent figures scatter over 0.79–0.97, about what that noise predicts.

Downstream that is worth roughly **±2 m** on the canonical −19.8 m. Quote it to
one decimal at most; the third digit is decoration.

**`fig_engine_tau_id.png`** is that identification, and it is the evidence for
every number in this directory. Panel (a) is the fit: rms residual in engine
speed against τ_e, one curve per readable figure, each with a clean minimum near
0.8 s rather than a plateau — which is what says the constant is determined and
not merely assumed. Panel (b) is the same result in the time domain: Riley's
digitised engine-speed trace with eq. (A4) replayed through it at three
constants, where τ_e = 0.1 s falls too early, 2 s too late, and 0.85 s tracks
the published curve through the whole transient.

## What moved, and what did not

The split is sharp and it is the reason both directories exist.

**Absolute altitude losses roughly double.** The canonical entry goes from
−9.35 m to **−19.79 m**. The whole IC plane shifts with it: at 0.90 Vs from
−2.70 to −12.19 m, at 0.80 Vs from −16.87 to −27.63 m, and the dense sweep's
range moves from [−85.75, −1.85] to [−92.80, −10.50] m.

**Comparisons between arms barely move.** Every margin the paper actually
argues from survives, losing 9 to 12 %:

| arm, canonical IC | ideal | τ_e = 0.85 | margin vs. DP |
|---|---|---|---|
| DP optimum | −9.35 | **−19.79** | — |
| CAA (α-hold) | −23.45 | −32.17 | −14.10 → **−12.38** |
| FAA (α-hold) | −27.80 | −36.12 | −18.45 → **−16.33** |
| CAA (full pull) | −97.30 | −105.50 | −87.95 → −85.71 |
| FAA (full pull) | −100.23 | −108.92 | −90.88 → −89.14 |

Note which way the handicap runs. The DP arm is the only one penalised for not
knowing about the lag — the scripted procedures do not optimise against any
engine — so flying the real engine **understates** the optimum's advantage.
Every margin here is a lower bound.

The pilot-error sensitivities keep their shape and compress, because they are
all differences:

| delaying the nose-down → pull switch | ideal | τ_e = 0.85 |
|---|---|---|
| 0 s | −9.35 | −19.79 |
| 0.25 s | −11.93 | −21.25 |
| 0.5 s | −22.75 | −28.59 |
| 1 s | −50.79 | −51.54 |

The 1-second delay costs 41.4 m on an ideal engine and 31.8 m on the real one:
still the difference between a recovery and an accident.

## The robustness matrix comes out cleaner

This is the one place where the real engine does not merely shift the numbers —
it improves the figure.

|  | ideal | τ_e = 0.85 |
|---|---|---|
| cells that never return to level (`*`) | 4 | **0** |
| cells closing still stalled (`†`) | 5 | **1** |
| range of the map | [−8.37, +15.03] m | [−9.78, +13.21] m |
| α at close, across the matrix | 11.85–16.02° | **13.50–15.04°** |

With an ideal engine, four heavy-and-aft cells never crossed γ = 0 inside the
15 s horizon, and their numbers were the loss up to an arbitrary cut rather than
the cost of a manoeuvre. On the real engine all 182 cells close. The reason is
physical and slightly counter-intuitive: with the CG past the divergence
boundary, slamming in full thrust is part of the problem, and a slow engine acts
as a filter that calms exactly those cells.

Inside the day-to-day loading envelope the map is otherwise unchanged — the
per-cell shift is under 0.55 m against numbers of 5 to 15 m — so the
conclusions drawn from `5_paper1_repro/fig_robustness_matrix.png` stand as
written. The single remaining `†` is (+15 % mass, +22.5 % CG), closing at
15.04° against a nominal 14.01°.

`cg_reach` is unaffected where it matters: the divergence boundary is at
**x_cg/c̄ = 0.4529** on both, because it comes from the open-loop eigenvalues of
the airframe and the engine does not enter them.

## Which directory to cite

- **Anything stated as a difference** — procedure penalties, pilot-error costs,
  the robustness map, CAA vs FAA — is robust to τ_e and can be cited from
  either. Prefer this one; it is the real aircraft.
- **Anything stated as an absolute altitude loss** must come from here, and must
  name the engine. On an ideal engine the same manoeuvre reads −9.4 m instead of
  −19.8 m, and a figure that does not say which is a number without units.
- `5_paper1_repro/` is kept as the τ_e = 0 end of the sensitivity analysis, not
  as superseded work. The pair is the analysis.

## Files

Same set as `5_paper1_repro/`, with two differences:

- `fig_trajectories_procedures_tau085.{png,pdf}` carries τ in its name, because
  the trajectory figure is the one most likely to travel alone. Panel (f) plots
  the commanded throttle dotted against the engine's response solid.
- `fig_riley_coefficients.{png,pdf}` is new here: it used to be written to a
  hardcoded `stall-paper/img/` outside the run's own directory, which is what
  aborted the first attempt at this suite.

`riley_coeffs` is model-only and identical in both directories — the coefficient
tables do not involve the engine.
