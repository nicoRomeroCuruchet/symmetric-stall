"""engine.py — Riley's first-order engine lag, eq. (A4).

Riley's simulation puts a first-order lag between the throttle handle in the
cockpit and the throttle the aerodynamic model sees (Appendix A, p. 13):

    delta_t = 1/(tau_e s + 1) delta_t,c                                   (A4)

He defines tau_e in the symbol list -- "engine-response time constant, sec" --
and never TABULATES a value for it. It is the only constant of appendix A he
leaves out: (A3) prints 0.65 and 0.35, (A9) prints the twelve entries of the
T0/T1 table, (A12) prints eighteen more, and (A4) prints nothing.

RECOVERING IT ANYWAY. The value is not lost, only unlabelled: Riley publishes
throttle chops with the engine-speed response beside them (figs. 16 and 18),
and appendix A closes the loop between the two,

    (A4)   delta_t  = 1/(tau_e s + 1) delta_t,c        <- the unknown
    (A3)   delta_t' = 0.65 delta_t + 0.35
    (A12)  N        = N0(delta_t') + N1(delta_t') V + N2(delta_t') V^2

so N(t) is an observation of delta_t(t) and tau_e is the only free parameter.
Digitising the published traces and fitting gives 0.79 s on fig. 16(a), 0.82 s
on 16(b) and 0.97 s on 18(b) -- hence RILEY_TAU_E below. The fit is well
conditioned: on 16(a) the residual is 77 rpm at the optimum against 139 rpm at
tau_e = 0.1 s and 270 rpm at 3 s.

Three checks that the digitiser is reading the page and not itself: it recovers
a 1.83 s chop where Riley's text says the chops "occurred in 2 sec"; the
(A3)+(A12) chain predicts 2387 rpm at the trim throttle where the figure starts
at 2400; and 1135 rpm at closed throttle where the figure floors at 1100.

It is an identification off a 1985 scan, not a number Riley signs, and the
figure is flown at 5000 ft while (A12) is stated at sea level. Treat 0.85 s as
good to about +/- 0.15 s and say where it comes from.
See scripts/verify/identificar_tau_motor.py, which reproduces all of it.

WHERE THIS LIVES, AND WHY NOT IN THE DP. The lag is applied when a trajectory
is rolled out, not while the policy is solved. Carrying it into the dynamic
program would mean promoting the effective throttle to a fifth state variable,
which multiplies the grid by the number of bins on that axis: 11 bins takes the
riley grid from 14.9 M states to 164 M, from 430 MB of VRAM to 5.0 GB, and from
4 h 18 on a 3090 to roughly six days on a 3070.

So the policy is optimal for an ideal engine and is then flown on the real one.
That is the conservative reading: the true optimum for the lagged plant would
do at least as well as this one, so the gap measured against the CAA/FAA
procedures is a LOWER bound on what optimal control could achieve. Note which
way the handicap runs -- the DP arm is the only one penalised for not knowing
about the lag, since the scripted CAA/FAA procedures do not optimise against
any engine at all -- so flying the lag understates the optimum's advantage.

WHAT THE LAG COSTS. At the canonical entry it is worth about 6 m of altitude
to every arm, which is most of the absolute number and almost none of the
comparison: over tau_e from 0 to 2 s the optimum's own loss grows 222 % while
its margin over the CAA procedure moves 23 %. The same split holds across the
mass x CG matrix, where inside the day-to-day loading envelope no cell of the
robustness map moves by more than 0.55 m. So results quoted as DIFFERENCES are
robust to the value; results quoted as ABSOLUTE altitude loss are not, and have
to name the engine they were flown on.
"""
from __future__ import annotations

import numpy as np

#: Riley's engine-response time constant, seconds. Not tabulated in TM-86309;
#: identified from the throttle chops of his figures 16 and 18 (0.79, 0.82 and
#: 0.97 s on the three legible traces). See the module docstring for the
#: derivation and scripts/verify/identificar_tau_motor.py for the code.
RILEY_TAU_E = 0.85


class EngineLag:
    """First-order lag on the throttle, per Riley eq. (A4).

    `tau` is the time constant in seconds; `tau <= 0` disables the lag and the
    commanded throttle passes straight through, which reproduces the plant the
    policy was solved against.

    The update is the EXACT discretisation of the first-order response over a
    step of `dt`,

        delta_t <- delta_t + (cmd - delta_t) (1 - exp(-dt/tau))

    rather than the forward-Euler form (cmd - delta_t) (dt/tau) the figure
    scripts each carried a copy of. The two agree to first order in dt/tau --
    at the 0.01 s control step and the tau >= 0.25 s of the published sweep the
    difference is under 2% of the increment -- but Euler is only conditionally
    stable, oscillating for dt > 2 tau, while this form is exact for a step
    input at any tau and cannot be driven unstable by a small tau.
    """

    __slots__ = ("tau", "value")

    def __init__(self, tau: float = 0.0, initial: float = 0.0) -> None:
        self.tau = float(tau)
        self.value = float(initial)

    def reset(self, initial: float = 0.0) -> None:
        self.value = float(initial)

    def step(self, commanded: float, dt: float) -> float:
        """Advance by `dt` towards `commanded` and return the effective throttle."""
        if self.tau <= 0.0:
            self.value = float(commanded)
        else:
            self.value += (float(commanded) - self.value) * (
                1.0 - np.exp(-float(dt) / self.tau)
            )
        return self.value
