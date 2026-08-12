# 4-DOF results — 2026-08-11

All of this was run on the **3090** (`nromero@100.68.2.122`), branch
`4dof-riley-thrust`, and **entirely with `THRUST_MODEL=riley`**. The code
default is `paper1`: run anything without that variable and the numbers will
not match (the canonical recovery gives −13.3 m instead of −6.8).

> Since the migration this is handled by `symstall-train --thrust riley`, which
> is the default, and the configuration is stamped inside the `.npz`. The
> policies below predate that and are anonymous.

## The two policies

| file | grid | states | run |
|---|---|---|---|
| `politicas/SymmetricStall_alpha_m10_40_FILLED.npz` | 56×81×80×41 | 14,878,080 | 47 iter, **4 h 18** |
| `politicas/SymmetricStall_alpha_m10_40_RAW.npz` | same, **unfilled** | | (the same run, before the fill) |
| `politicas/SymmetricStall_policy.npz` | 56×41×60×41 | 5,648,160 | 23 iter, **43 min** |

**The first** is the new grid: `alpha` relocated to **[−10, +40]** (exactly the
domain Riley tabulates) and `V` lowered to **[0.4, 2.0]**. The second is the
**paper-1 grid** with Riley's thrust — it was run to separate the effect of the
grid from the effect of the thrust model, and **has not been analysed yet**.

Configuration identical in both, and to the paper's: `gamma=1.0`, `theta=5e-6`,
`maximum_iterations=20,000`, `n_micro=10`, fixed dt, all shaping terms at zero.

## The directories

    0_heatmaps/        the heatmaps and the trajectory that main.py generates on its own
    1_canonica_fill/   80 files: comparacion_fill_vXXX.png (16 airspeeds, filled
                       vs unfilled) plus the individual ones and the PDF
    2_barrido_IC/      barrido_v0.png (the h_min(V0) curve) and the two families
                       of overlaid trajectories
    3_maniobras/       the CAA/FAA material and the ablations  <- WHAT MATTERS MOST
    politicas/         the .npz files

## The numbers

### Canonical trajectory (gamma0=0, alpha0=20, q0=0)

| V0/Vs | h_min filled | h_min unfilled | the fill is worth |
|---|---|---|---|
| 0.80 | −16.859 | −16.963 | 10.4 cm |
| 0.85 | **−9.344** | −9.589 | 24.5 cm |
| 0.86 | −7.904 | −8.199 | 29.5 cm |
| 0.89 | −3.897 | −4.369 | 47.2 cm |
| 0.90 | −2.697 | −3.233 | 53.6 cm |
| 0.93 | −0.036 | −0.774 | 73.8 cm |
| 0.95 | **+0.000** | −0.390 | — |

- **At 0.95 Vs there is no manoeuvre**: with Riley's thrust the aircraft does
  not sink. That is why the canonical initial condition drops to 0.85/0.86.
- The loss is almost linear between 0.80 and 0.90: **1.42 m per hundredth of
  Vs**.
- **`h_min` is the robust metric**, not the final `dh`: it does not depend on
  the cut-off criterion. The `has_dived` threshold was lowered from −2 to
  **−1 degree** because at 0.90 the filled policy only reaches −1.82 and would
  not trigger the cut.

### Invariants (they hold across all 16 entries)

- Mean `delta_t` is **exactly 1.000** in the filled policy, always.
- Final `alpha` between **13.66 and 14.11 degrees**: the manoeuvre ends pinned
  to the stall boundary. This is the sliding mode from paper 1.
- `alpha_min` varies by only **0.26 degrees** between 0.85 and 0.90 while the
  altitude loss changes by a factor of 3.5: **the manoeuvre is the same, what
  changes is how long it lasts**.

### CAA / FAA manoeuvres (`3_maniobras/maniobras_v086.png`), at 0.86 Vs

| | dh | alpha_max | state |
|---|---|---|---|
| optimum | −7.904 | 20.00 | recovered |
| CAA alpha-hold | −21.850 | 20.07 | recovered |
| FAA alpha-hold | −26.055 | 20.07 | recovered |
| CAA full-pull | −96.321 | **35.82** | TIMEOUT |
| FAA full-pull | −99.156 | 35.24 | TIMEOUT |

The `full-pull` **re-stalls the wing** and does not recover: it backs the
footnote to the paper's table with a number.

### Ablation: trigger vs power (`3_maniobras/ablacion_v086.png`)

Of the 13.95 m the CAA loses against the optimum:

    the 2 s power ramp explains    12.90 m   (92.5 %)
    the late elevator trigger       0.59 m   ( 4.2 %)

`trigger 14 + step` gives **−8.948 m**, within a metre of the optimum: the
procedure, **without touching the elevator at all** and merely applying power
as a step, almost reaches the optimum.

### Power only, same elevator (`3_maniobras/potencia_pura_v089.png`), 0.89 Vs

`delta_e` comes from the policy in all three arms; the only thing that changes
is the power ramp:

| | h_min | penalty |
|---|---|---|
| DP, 0.6 s ramp | −7.907 | — |
| CAA, 2 s ramp from t=0 | −15.857 | −7.95 m |
| FAA, 2 s ramp after the nose-down | −20.050 | −12.14 m |

The **0.38 s** the FAA waits for the nose-down cost **4.19 m**, i.e. **11
metres per second of delay in applying power**.

### Riley's engine dynamics (`3_maniobras/potencia_riley_v089_tau050.png`)

Riley, Appendix A eq. (A4), includes a **first-order lag** between the throttle
lever and thrust: `delta_t = 1/(tau_e s + 1) delta_t,c`. **The model does NOT
have it** (the throttle feeds `_compute_ct` directly, with no state for it), and
**Riley does not publish a value for `tau_e`**.

| tau_e | DP | CAA | FAA | CAA−DP | FAA−CAA |
|---|---|---|---|---|---|
| 0.00 | −7.907 | −15.857 | −20.050 | −7.95 | −4.19 |
| 0.25 | −10.747 | −18.501 | −22.612 | −7.75 | −4.11 |
| 0.50 | −13.498 | −21.099 | −25.136 | −7.60 | −4.04 |
| 1.00 | −18.495 | −25.788 | −29.644 | −7.29 | −3.86 |

**The absolute loss depends strongly on `tau_e`; the comparison between arms
barely does (8 %).** So "waiting for the nose-down costs 4 m" can be reported
without committing to a value of `tau_e`; what canNOT be reported without
fixing it is the optimum's absolute altitude loss.

### Pilot with delay (`3_maniobras/piloto_realista_v086.png`)

With **1 s of human delay** following the DP's `delta_e`, it **crashes every
time**, even with an ideal engine: it holds the dive one second too long,
`alpha` falls to **−19 degrees**, `gamma` to −30, then it overcorrects with
`q = +95 deg/s` and re-stalls. It is a **PIO**.

Conclusion: **the optimal policy is a bound, not a flyable procedure.** Its
elevator pulse lasts 0.2 s and tolerates no human delay. The CAA/FAA procedures
are robust precisely because they use state triggers (`alpha < 14`) rather than
a time profile.

## How to reproduce

    cd ~/stall-spin-recovery-dp
    THRUST_MODEL=riley .venv/bin/python <script> results/politicas/<npz> <args>

> Since the migration: run from the repo root against `data/policies/`, and the
> thrust model is a flag rather than an environment variable. See the root
> README.

| script | what it does |
|---|---|
| `canonica_filled_vs_raw.py <raw> <filled> V0...` | filled vs raw trajectories |
| `barrido_v0.py <raw> <filled>` | the `h_min(V0)` curve |
| `familia2.py <npz> <out.png> V0...` | overlaid trajectories |
| `maniobras_086.py <npz> <V0> [alpha0]` | optimum vs CAA vs FAA |
| `ablacion.py <npz> <V0>` | trigger × power factorial |
| `potencia_pura.py <npz> <V0>` | same elevator, different ramp |
| `potencia_riley.py <npz> <V0> <tau_fig> <taus...>` | with Riley's engine |
| `piloto_realista.py <npz> <V0> <tau_h> <taus_m...>` | human delay + engine |
| `fill_terminal_policy.py <npz>` | the terminal fill (overwrites, leaves `.npz.raw`) |

## Open items

1. **Analyse the paper-1 grid run** (`SymmetricStall_policy.npz`), the one that
   separates the effect of the grid from the effect of the thrust model.
2. **`main.py` records neither `THRUST_MODEL` nor the CG, in the log or in the
   `.npz`.** There is no way to tell, looking at a policy, which model trained
   it. *(Fixed in the migration: see `runconfig.py` and the `run_metadata` key.)*
3. Riley's `tau_e`: it is in neither Appendix A nor the symbol list with a
   figure. It would have to be tracked down in the report's references.
4. The 6-DOF grid still has the non-absorbing box problem (4.99 % of its cells
   are worth less than crashing).
