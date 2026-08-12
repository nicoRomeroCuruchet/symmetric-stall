"""engine.py — Riley's first-order engine lag, eq. (A4).

Riley's simulation puts a first-order lag between the throttle handle in the
cockpit and the throttle the aerodynamic model sees (Appendix A, p. 13):

    delta_t = 1/(tau_e s + 1) delta_t,c                                   (A4)

He defines tau_e in the symbol list -- "engine-response time constant, sec" --
and never publishes a value for it, anywhere in the report.

WHERE THIS LIVES, AND WHY NOT IN THE DP. The lag is applied when a trajectory
is rolled out, not while the policy is solved. Carrying it into the dynamic
program would mean promoting the effective throttle to a fifth state variable,
which multiplies the grid by the number of bins on that axis: 11 bins takes the
riley grid from 14.9 M states to 164 M, from 430 MB of VRAM to 5.0 GB, and from
4 h 18 on a 3090 to roughly six days on a 3070. It would also produce a policy
valid for one invented value of tau_e, since Riley supplies none.

So the policy is optimal for an ideal engine and is then flown on a lagged one.
That is the conservative reading: the true optimum for the lagged plant would
do at least as well as this one, so the gap measured against the CAA/FAA
procedures is a lower bound on what optimal control could achieve. It also
sharpens rather than weakens the paper's headline -- the optimal policy is a
bound, and the engine lag is one more reason it is not a flyable procedure.

What IS reported is the sensitivity: scripts/figures/potencia_riley.py sweeps
tau_e and finds that the absolute altitude loss depends strongly on it while
the comparison between arms barely does (8% over tau_e from 0 to 1 s). The
conclusion "waiting for the nose-down costs 4 m" therefore survives without
committing to a value of tau_e; the absolute loss of the optimum does not.
"""
from __future__ import annotations

import numpy as np


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
