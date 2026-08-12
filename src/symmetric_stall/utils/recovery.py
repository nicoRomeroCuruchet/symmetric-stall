"""Single definition of the stall-recovery stopping rule.

Every paper experiment measures altitude loss over the same interval: from the
stalled entry until the recovery is complete. That rule used to be copied by
hand into each experiment script, which is how the two clauses below can drift
apart without anyone noticing. It lives here instead, with one owner.

Recovery is declared when the flight path returns to level after the dive has
developed. Two clauses implement that:

  crossing     gamma >= 0, the direct reading of "back to level flight".

  convergence  gamma sits within LEVEL_BAND_DEG of level and gamma-dot has
               fallen below GAMMA_DOT_TOL_DEG_S for DWELL_S. This closes the
               marginal entries near V0 = Vs, whose flight path approaches
               level asymptotically from below and never crosses: they settle
               into a quasi-steady 0.11 m/s descent (gamma ~ -0.2 deg,
               alpha ~ 11.4 deg) that no crossing test can ever detect, so the
               crossing clause alone leaves their altitude loss undefined --
               it just keeps growing with the integration horizon.

The clauses cannot be confused for one another: a trajectory that genuinely
crosses does so with |gamma-dot| ~ 0.5-1.2 deg/s inside the band, while the
asymptotic ones sit at <= 5e-4 deg/s. Three orders of magnitude separate them,
so the convergence clause never fires on a crossing trajectory and every
altitude loss measured before it existed is unchanged.
"""

DIVE_THRESHOLD_DEG = -0.5     # gamma below which the dive counts as developed
LEVEL_BAND_DEG = -0.5         # band below level where convergence may be read
GAMMA_DOT_TOL_DEG_S = 0.05    # gamma-dot below which the path has converged
DWELL_S = 0.5                 # how long that must hold before declaring it


class RecoveryMonitor:
    """Step-by-step evaluation of the stopping rule.

    Feed it gamma (deg) once per integration step; `update` returns True on the
    step the recovery completes. The monitor keeps the dive latch and the
    gamma-dot history, so callers hold no stopping-rule state of their own.
    """

    def __init__(self, dt):
        self.dt = dt
        self.has_dived = False
        self._prev_gamma = None
        self._dwell = 0.0

    def update(self, gamma_deg):
        if gamma_deg < DIVE_THRESHOLD_DEG:
            self.has_dived = True

        gamma_dot = (None if self._prev_gamma is None
                     else (gamma_deg - self._prev_gamma) / self.dt)
        self._prev_gamma = gamma_deg

        if not self.has_dived:
            return False
        if gamma_deg >= 0.0:
            return True

        # Convergence clause: inside the band, with the flight path no longer
        # moving. The dwell keeps a momentary inflection from passing as one.
        if (gamma_deg > LEVEL_BAND_DEG and gamma_dot is not None
                and abs(gamma_dot) < GAMMA_DOT_TOL_DEG_S):
            self._dwell += self.dt
            if self._dwell >= DWELL_S:
                return True
        else:
            self._dwell = 0.0
        return False
