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

Four checks that the digitiser is reading the page and not itself, each against
a source that is not the plot: it recovers a 1.83 s chop where Riley's TEXT says
the chops "occurred in 2 sec"; it reads 120.0 ft/s at t = 0 where the CAPTION
states V = 120 ft/sec; the (A3)+(A12) chain predicts 2387 rpm at the trim
throttle where the figure starts at 2400; and 1135 rpm at closed throttle where
the figure floors at 1100.

HOW GOOD IS IT. The time histories are published as PLOTS ONLY -- tables I to
VIII carry aerodynamic coefficients, airfoil coordinates and stability
comparisons, and none of them is a time history -- so the trace is a
reconstruction of a drawing and the honest question is how much of one.

At 400 dpi a pixel is worth 48 ms and 16 rpm, far finer than needed. What limits
the reading is the printed line, whose median thickness is 7 px = 113 rpm: there
is no value of N, there is a band. Displacing the trace by half a line width
moves tau_e from 0.62 to 1.00, and misreading V by 5 ft/s moves it from 0.68 to
0.91. What does NOT matter is the geometry -- a 1 % error in the time axis, or
a fit window anywhere from 8 to 20 s, moves the last digit only.

So the number is tau_e = 0.8 +/- 0.2 s, and RILEY_TAU_E = 0.85 sits inside it.
Reassuringly the three independent figures span 0.79 to 0.97, i.e. they scatter
by about what the digitisation noise predicts. Downstream this is worth roughly
+/- 2 m on the canonical altitude loss of -19.8 m -- so quote that number to one
decimal at most, and never to three. `identificar_tau_motor.py --uncertainty`
regenerates the perturbation table; the script reproduces everything else.

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


#: Time constant of the elevator channel, seconds. UNLIKE RILEY_TAU_E this is
#: not a property of Riley's aeroplane recovered from his data: Riley models no
#: elevator dynamics at all (his only first-order lag is the engine's, eq. A4),
#: and neither Gratton's flight tests nor either thesis in refs/ report one.
#:
#: The AA-1 has a REVERSIBLE, directly linked control system -- stick to cable
#: to elevator, no servo -- so what limits the elevator is not an actuator but
#: the pilot's neuromuscular system plus control-system compliance. The
#: quasi-linear pilot models put that lag near 0.1 s (omega ~ 10 rad/s).
#:
#: TREAT THIS NUMBER AS UNVERIFIED. It is an order of magnitude taken from the
#: pilot-dynamics literature, not a measurement, and the primary result should
#: be a sweep over tau rather than a single value: if the loss is flat across
#: 0.05-0.2 s the choice does not matter, and if it is not, the value has to be
#: argued from a source that has been read rather than cited from memory.
DEFAULT_ELEVATOR_TAU = 0.10


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


class ActuatorLag(EngineLag):
    """First-order lag on the elevator command. Evaluation only.

    Same mathematics as EngineLag -- the exact discretisation of a first-order
    response -- but a different physical claim, and it is worth keeping the two
    classes apart so a reader is not misled into thinking Riley published this
    one too. See DEFAULT_ELEVATOR_TAU.

    Like the engine lag, this belongs to the EVALUATION and never to the solve:
    the policy is computed against an instantaneous elevator and then flown on
    a lagged one. That is a real approximation and the paper has to say so, but
    it errs in the safe direction. A policy optimal for the ideal plant, flown
    on the lagged plant, can only do WORSE than the policy that is optimal for
    the lagged plant, so every altitude loss reported this way is an upper
    bound on what internalising the lag would achieve. Conclusions of the form
    "the optimum still beats the certified procedures" therefore survive; a
    conclusion of the form "the optimum loses exactly X metres" does not.

    Internalising it would mean carrying delta_e as a fifth state. The obstacle
    is NOT memory -- 21 elevator bins take the grid to 312 M states, 1.2 GB for
    V in float32, comfortable on a 24 GB card, and saying otherwise would
    contradict this work's own O(N_s) claim. The obstacle is time: 21x the
    states is ~37 h on an RTX 4090 at the measured rate of the 4-DOF solve.
    """

    __slots__ = ()
