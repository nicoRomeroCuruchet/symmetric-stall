# `img/` — where every figure of the manuscript comes from

Assembled 2026-08-31. This directory is what `main.tex` reads; nothing here
is authored, every file is a copy of a generated artefact under `results/`.
The point of this file is that a reader can check that, because getting it
wrong is easy and silent — see **The L1/L4 trap** below.

`img/` existed when `main.pdf` was built (2026-08-17 20:38) and was then
lost. It was never committed, is not in `stall-spin-recovery-dp`, and is not
on any of the three GPU nodes. It was rebuilt from `results/`, with two files
recovered from the old PDF itself.

## Copied under the same name (12)

| file | source |
|---|---|
| `banked_glider_L1_mca_policy_{CL,MuDot}_V_{1.2,4.0}.png` (4) | `results/6_riley_engine/case1_3dof/` |
| `banked_glider_L4_mca_policy_{CL,MuDot}_V_{1.2,4.0}.png` (4) | `results/6_riley_engine/case1_3dof/` |
| `fig_pilot_sensitivity.png` | `results/6_riley_engine/` |
| `fig_procedures.pdf` | `results/6_riley_engine/` |
| `fig_robustness_matrix.png` | `results/6_riley_engine/` |
| `fig_trajectories_procedures.pdf` | `results/5_paper1_repro/` |

## Renamed on copy (4)

| file | source |
|---|---|
| `fig_ic_optimum.pdf` | `results/6_riley_engine/fig_optimal_ic_heatmap.pdf` |
| `fig_ic_procedures.pdf` | `results/6_riley_engine/fig_procedures_ic_heatmap.pdf` |
| `riley_coefficients.png` | `results/6_riley_engine/fig_riley_coefficients.png` |
| `riley_symmetric_stall_heatmaps.png` | `results/0_heatmaps/symmetric_stall_heatmaps.png` |

## Regenerated (1)

`combined_alt_loss_contours.png` — the file and the script that drew it were
both lost; only page 9 of the old PDF survived. It is recoverable because it
is not a rollout but the converged value function itself: on the
shortest-path formulation `dh_min ~= -V*`, so the four panels are four slices
of `policy_L1_53280.npz` (430 KB, in the repository, no GPU). Regenerate with

    python3 scripts/paper1/fig_alt_loss_contours.py

The four airspeeds the panels ask for land exactly on grid nodes, so nothing
is interpolated. The result reproduces the archived page: same topology, same
asymmetry about the knife-edge condition, same contour levels.

## Recovered from the old PDF (2)

`profiling_table_L1.tex` — transcribed from the text layer of Table 2 of
`main.pdf`. The generated `.tex` did not survive anywhere.

`profiling_table_L4.tex` — copied from
`results/6_riley_engine/case1_3dof/`, **with its label changed** from
`tab:profiling` to `tab:profiling_L4`. The generator in
`stall-spin-recovery-dp` emits `tab:profiling` for both grids, so the two
tables collided and `\ref{tab:profiling_L4}` in the manuscript resolved to
nothing. That repository is read-only for this work, so the fix lives here.

Note the L4 table gives a total of 894 563 ms while the old PDF's Table 3
says 896 659 ms. They are two runs of the same solve; the reproduced one,
894.6 s = 14.9 min, is the number the manuscript states.

## The L1/L4 trap

`results/6_riley_engine/case1_3dof/` holds two generations of the Case I
material side by side. The `L1` files predate commit `606f609` of the other
repository, which fixed three defects in the DP-vs-CasADi validation, none of
them in the DP. The largest: the NLP was solved with `CL_REF = 1.64`,
extrapolating the linear lift slope through stall, giving a stall speed of
27.33 m/s against the DP's 31.95 m/s — 16.9 % off. From the same
*normalised* initial condition the two methods therefore started at different
true airspeeds, 32.8 against 38.3 m/s, and the NLP entered the pullout with
37 % less kinetic energy. It lost more altitude at all ten entries, and the
figure showed the DP winning by 10–25 m. That advantage was an artefact.

The manuscript no longer includes those trajectory figures — they were
replaced by `tables/table_casadi_benchmark.tex` — but the policy plots for
both grids are still cited, deliberately, as a baseline-vs-refined
comparison. Do not "fix" an L1 filename by pointing it at an L4 file, or the
other way round.

## Three things left to decide

1. **`riley_symmetric_stall_heatmaps.png` is stale.** Its source is dated
   2026-08-12, before all of the engine-lag work. The manuscript shows it
   beside results that do carry Riley's `tau_e = 0.85 s`. Regenerating it
   means re-running the heatmap extraction on the Case II policy.
2. **`fig_trajectories_procedures.pdf` has a newer sibling.**
   `results/5_paper1_repro/fig_trajectories_procedures_tau085-010.pdf` is the
   same figure flown with `tau_e = 0.85 s` and `tau_de = 0.10 s`. The
   manuscript currently asks for the name without the suffix, so the copy
   here is the one WITHOUT actuator dynamics. Swapping it changes what the
   figure means, so it was left alone.
3. **`fig_robustness_matrix.png`** is the reworked version (CG forward
   positive, cropped at −15 %, no regime lines, no secondary axis). Check it
   against whatever the surrounding text still says about the CG sign — as of
   this writing `main.tex` lines 912, 948 and 1697 still describe the offset
   as aft-positive and contradict the figure.
