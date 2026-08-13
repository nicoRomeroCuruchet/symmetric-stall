# Figures from the corrected-constants run — 2026-08-12

Produced from the reference policy trained on udesa's 3090 after the audit:

    data/policies/SymmetricStall_riley_56x81x80x41_thrust-riley.npz

Run: riley grid (56x81x80x41, 14,878,080 states), riley thrust, nominal CG,
50 policy-iteration steps in 4 h 19, stopped with 1228 states still changing
action under the 1487 chattering tolerance. Final residual 0.0.

This is the first policy trained with MASS and CHORD corrected against Riley's
Table I (715.3152 kg and 1.2192 m; they were 715.21 and 1.22). Vs moved by
+0.0073%, and since the grid is expressed in V/Vs that rescales the whole
airspeed axis, so these figures are not bit-comparable with the ones under
1_canonica_fill/ and 3_maniobras/, which come from the earlier run.

They do reproduce it to within 6 mm on h_min:

| V0/Vs | h_min filled | earlier run |
|---|---|---|
| 0.85 | -9.350 | -9.344 |
| 0.86 | -7.910 | -7.904 |
| 0.89 | -3.901 | -3.897 |
| 0.90 | -2.703 | -2.697 |
| 0.93 | -0.039 | -0.036 |
| 0.95 | +0.000 | +0.000 |

CAA/FAA at 0.86 Vs: optimum -7.910, CAA alpha-hold -21.877, FAA alpha-hold
-26.083, and both full-pull arms re-stall (alpha_max 35.8 deg) and time out.
