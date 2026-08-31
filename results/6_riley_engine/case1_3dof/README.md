# Case I — 3-DOF banked pullout, the figures the paper needs

Case I of the paper validates the solver against Bunge et al. (2018) on a
3-DOF banked-pullout glider. It shares nothing with Case II but the solver:
different aircraft, different state space, **no engine at all**. It lives here
only because this is where the paper's figures are assembled; nothing in this
folder is affected by `engine_tau_s`, and nothing in it should be re-run when
the engine changes.

The work is not in this repository. It comes from `stall-spin-recovery-dp`,
branch `3dof-reduced-banked-pullout`, commit `606f609` ("Fix three defects in
the DP-vs-CasADi validation, none of them in the DP").

## What each file is

| file | grid | provenance |
|---|---|---|
| `banked_glider_L1_mca_policy_*.png` (4) | L1, 53,280 states | copied from the other repo, pre-existing |
| `banked_glider_L1_mca_validation_guided_Fig{3,4}.png` | L1 | idem |
| `policy_L1_53280.npz` | L1 | idem, 430 KB |
| `banked_glider_L4_mca_policy_*.png` (4) | **L4, 31,948,500 states** | **re-solved 2026-08-17** |
| `banked_glider_L4_mca_validation_guided_Fig{3,4}.png` | L4 | idem |
| `profiling_report_L4.txt`, `profiling_table_L4.tex` | L4 | idem |

The L4 policy itself is **not** in this folder: it is 255 MB. It lives at
`data/policies/ReducedBankedGliderPullout_L4_mca_full.npz`, which `.gitignore`
excludes, so it is on this laptop but not in the repository. A second copy
stays on the machine that produced it,
`a1554-ubu:~/trabajo-L4/repo/results/`. The laptop copy is covered by the
restic backup (snapshot `4f8b7cd0`, 2026-08-18); the remote one is not.

## The L4 re-solve

The L4 material had been lost and no policy for it existed anywhere — not in
this repo, not in the other one on any of its sixteen branches, and not on any
of the three GPU nodes. It was re-solved on `a1554-ubu` (RTX 4090):

    31,948,500 states (361 x 250 x 354) x 91 actions
    use_mca_timestep=True, mca_include_mudot=True, dt_max=0.10   -> mca_full
    31 minutes wall clock

`main.py --mca` selects `mca_state` (`mca_include_mudot=False`), which is NOT
what the paper's figures use; `mca_full` has no CLI flag, so the solve was
driven by a small runner setting the field directly.

**On convergence.** The run stopped at the 1000-iteration cap rather than at
zero residual, which the `--mca` help text warns about ("does NOT converge at
L3+ under gamma=1"). The warning overstates it for `mca_full`: the number of
states changing their optimal action falls from 4.77 M in the first twenty
iterations to a flat **26 states out of 31,948,500** — eight ten-millionths of
a percent, two equally-good actions alternating forever. That is chattering,
not non-convergence, and it is the same phenomenon the Case II policy reports
as `n_states_chattering = 1228` of 14.9 M. Say "26 states chattering", not
"did not converge".

## Two things to check before using these

**The validation has one broken point.** `Fig4` (gamma_0 = -60 deg) is clean
except at mu_0 = 150 deg:

    mu0 = 150 deg   DP = -1002.92 m   NLP = -500.01 m   gap = 100.6 %
    mu0 = 120 deg   DP =  -194.89 m   NLP = -193.14 m   gap =   0.9 %

Every other point across both figures agrees to 0.7-3.6 %, with the DP always
slightly below the NLP optimum, which is the correct direction. The outlier
sits at the extreme corner of the envelope, the NLP returns a suspiciously
round -500.01 m, and Ipopt reports success -- which is worse than a failure
because it does not announce itself. Unresolved; do not publish that point
without looking at it.

**The profiling timing was reproduced, the stored one was not.** This run gives
**894.6 s = 14.9 min**, which is what the paper states. The
`profiling_report.txt` sitting in the other repo says 1019.4 s = 17.0 min, a
slower run of the same solve. The paper's number is right.

Note also that `profiling_table_L4.tex` carries `\label{tab:profiling}` as
generated, while the manuscript cites `\ref{tab:profiling_L4}`. That mismatch
is one of the paper's dangling references.

## Still missing

`combined_alt_loss_contours.png` — the four-panel altitude-loss contour figure
of Sec. 4.4, swept over `V_0/V_s` in {1.2, 2.0, 3.0, 4.0}. It is not in either
repository, on any branch, in any of the three GPU nodes, or in any trash
folder. From the text it is a baseline-grid (53,280-state) figure, so it should
be regenerable from `policy_L1_53280.npz` without a GPU -- but the script that
drew it has not been located.

`plot_value_function_contours` at L4 is not usable: it materialises the whole
grid and was killed by the OOM killer at 29.6 GB of host RAM on a 31 GB
machine. The paper does not use that figure at L4. Worth a sentence in the
scaling section, since the solve is compute-bound and scales while the
post-processing does not.

## The benchmark, as a number

`scripts/paper1/casadi_benchmark.py` parses `diagnostics/casadi.log` and
writes `casadi_benchmark.json` and `table_casadi_benchmark.tex`. On the
agreed metric — the disagreement as a fraction of the deepest altitude loss
of each sweep, `|h_DP - h_NLP| / max_j |h_NLP,j|` — the nine valid points
give:

| | worst | mean |
|---|---|---|
| gamma_0 = -30 deg (5 points) | 1.37 % | 0.87 % |
| gamma_0 = -60 deg (4 points) | 0.91 % | 0.88 % |
| **overall** | **1.37 %** | 0.87 % |

Pointwise (dividing by each point's own loss) the worst is 3.6 %, which is
the shallowest entry of the shallowest sweep and is why that normalisation
was not the one chosen.

**The gap is not CasADi's tuning.** It is the rollout accumulating altitude
with forward Euler at dt = 0.10 s while the NLP integrates with RK4 at
T/150. Over a pullout gamma rises monotonically to zero, so `|h_dot|` falls
monotonically and the rectangle rule always takes the larger endpoint: the
error is one-signed — the DP appears to lose more at all nine points — and
telescopes to

    bias = -(dt/2) * V_0 * sin(gamma_0)

which depends on gamma_0 and on nothing else, not even mu_0. That predicts
one number per sweep before looking at the data:

| gamma_0 | predicted | smallest observed gap | ratio |
|---|---|---|---|
| -30 deg | 0.96 m | 0.96 m | 1.002 |
| -60 deg | 1.66 m | 1.66 m | 1.000 |

So the floor of the disagreement is an artefact of the evaluation script,
not of either optimiser, and re-integrating the same policy with a
trapezoid would move the DP DOWN towards the NLP. Tightening Ipopt would
move the NLP down as well and make the gap slightly worse, not better.

What remains above that floor is real: 0.06-1.18 m, all of it at
gamma_0 = -30 deg, and that part is the genuine price of holding each
control for 0.10 s against an NLP free to switch at every node.
