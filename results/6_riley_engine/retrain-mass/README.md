# Does the optimum only hold at the point it was solved for?

The reviewer's objection to a globally optimal policy computed for one
aeroplane is that it is optimal for THAT aeroplane. This directory answers it
by re-solving the policy for a perturbed aircraft and flying the two against
each other.

Produced by `scripts/paper1/retrain_mass_study.py`.

## The result

Retraining the policy for a perturbed aeroplane buys **0.1 to 0.4 m**, on
recoveries costing 9 to 46 m, across ±15 % weight and a CG 10 % of chord
forward. Nearly all of the degradation a perturbed aircraft causes is removed
instead by rescaling one pre-flight scalar, the stall-speed normalisation,
which is known from the loading sheet before take-off.

## Files

| file | what it is |
|---|---|
| `fig_retrain_mass.{png,pdf}` | the figure the paper uses, at the paper's plant |
| `table_retrain_mass.tex` | the same, per cell, with the absolute losses |
| `retrain_mass_study.json` | every number, with the run configuration |
| `fig_retrain_mass_de{000,010,015}.png` | elevator-lag sensitivity, see below |

`fig_retrain_mass_de000.png` is byte-identical to `fig_retrain_mass.png`. It
is kept so that the sensitivity set is complete and can be read without
knowing which of the three the baseline happens to be.

## Why the entries are deep

The paper's default evaluation grid starts at `alpha_0 = 16` deg, two degrees
past Riley's 14 deg boundary. That band was calibrated on the nominal
aircraft, and on one 10 % lighter it stops being a recovery at all: the value
function puts the optimum at 1.04 m from the shallowest entry, and on an ideal
engine the aeroplane climbs 17.9 m and times out rather than descending.
Almost the whole loss reported there was the engine lag, not the policy.

This study therefore enters at `alpha_0` in {20, 26, 32} deg, 6 to 18 past the
boundary. Every cell is a genuine recovery at every mass: `gamma` reaches
−5.6 to −16.9 deg and the trajectory is never above its entry altitude.

## Why the axis is in metres

It was in per cent, and the axis was doing damage. Per cent divides by the
retrained loss, which runs from 9 to 46 m here, so a fixed disagreement of a
few centimetres reads as 0.2 % at one end of a panel and 3.4 % at the other.
Two cells came out negative, which is impossible against a policy optimal for
that aeroplane, and the figure had to carry a shaded band and a paragraph
explaining that the impossibility was an artefact.

In metres there is nothing to explain, and the claim stops depending on what
it is divided by. Each row's tick label names the loss its excess is measured
against, because moving to metres removes the denominator the per-cent axis
carried implicitly.

## The elevator-lag sensitivity

`tau_de` is the one constant of the plant with no source behind it: Riley
models no elevator dynamics, and 0.1 s is an order of magnitude taken from
quasi-linear pilot models rather than from a measurement. The three figures
sweep it.

| panel | `tau_de` = 0 | 0.10 | 0.15 |
|---|---|---|---|
| m = 0.90 | one cell at −0.23 m | mostly positive | all nine positive |
| m = 1.10 | all nine positive | all nine positive | four negative |
| m = 1.00, CG fwd | all nine negative, 2–6 cm | one negative | one negative |
| m = 1.15, CG fwd | all nine positive | one negative | three negative |

**No value leaves all four panels clean, and which panel goes negative moves
with `tau_de`.** A defect belonging to a policy would stay in its panel when
the plant changes. These do not, which is what establishes that the
sub-decimetre signs carry no information rather than merely asserting it.

The paper uses `tau_de = 0`. Not because it gives the best-looking figure,
which at 0.15 it plainly does not, but because it is the plant every other
artefact in the result set is computed on: `procedures.json`, the four tables,
both initial-condition heat maps and the robustness matrix. The elevator lag
enters the paper as a sensitivity study, not as the baseline.

Regenerate any of them with

    STALL_ELEVATOR_TAU=0.15 python3 scripts/paper1/retrain_mass_study.py

## Still open

At `tau_de = 0` and m = 1.00 with the CG forward, all nine cells come out 2 to
6 cm negative, systematically rather than scattered. One correlate, not a
demonstrated cause: of the five policies, the CG-only one stopped closest to
its chattering tolerance, 1477 states against a limit of 1487, where the
others stopped at 1228 to 1404. Re-solving it with a tighter tolerance would
settle whether that is the reason. The conclusion of that panel does not
depend on it: nothing in the observation carries the CG, so there is no scalar
to rescale and nothing for retraining to buy, and cells sitting on zero in
either direction is what that looks like.
