"""The 4-DOF symmetric reduction of Riley's 6-DOF model.

Riley (1985), Appendix B, writes nonlinear six-degree-of-freedom rigid-body
equations in body axes. This module carries the longitudinal reduction used
throughout the paper, with state

    x = (gamma, V/Vs, alpha, q)

and control u = (delta_e, delta_t). The reduction is EXACT, not approximate,
under the following declared constraints:

    v = p = r = 0,  phi = 0,  beta = 0,  delta_a = delta_r = delta_f = 0

Every term Riley carries that this model does not is killed by those
constraints, not dropped by hand:

  - the lateral-directional equations (v-dot, p-dot, r-dot) decouple entirely;
  - the inertia cross terms (Iz-Ix)pr and Ixz(r^2 - p^2) in q-dot vanish, so
    Riley's pitch equation collapses to q-dot = M_b / Iyy;
  - the propeller gyroscopic term in q-dot is proportional to r and vanishes
    too (Riley keeps it "for completeness"; it is small even when it does not);
  - the sideslip increments dC_L,beta, dC_m,beta and dC_D,beta are zero by
    construction, since Riley tabulates them as zero at beta = 0;
  - C_D,dyn is identically zero in Riley's model, so drag carries no rate
    terms.

With beta = 0 the stability axis coincides with the wind axis, so Riley's
rotation R(alpha) from stability to body axes is already contained in the
wind-axes form used here. That is not assumed: scripts/verify/
verificar_ejes_riley.py implements Riley's chain independently and compares,
agreeing to 2e-12 over 4000 random states.

What this model deliberately does NOT reproduce from Riley:

  - the first-order engine lag of eq. (A4), delta_t = 1/(tau_e s + 1)
    delta_t,c. Riley never publishes a value for tau_e, and the sensitivity to
    it is reported separately (scripts/figures/potencia_riley.py). Thrust is
    otherwise commanded directly.
  - altitude: rho is fixed at sea level. This is consistent rather than
    contradictory, because C_T = T/(q_bar S) is independent of density --
    thrust scales with sigma by eq. (A10) and so does q_bar.

Equations, as implemented in `derivatives`:

    gamma_dot = [A C_L + A C_Lalpha_dot k q - (g/V) cos(gamma)]
                / (1 + A C_Lalpha_dot k)          A = q_bar S/(m V), k = c/2V
    alpha_dot = q - gamma_dot
    V_dot     = (-g sin(gamma) - D/m) / Vs
    q_dot     = M_y / Iyy

There is no explicit thrust term in the force equations, and that is Riley's
formulation, not an omission -- Appendix B, p. 17: "Note that in this model
there are no thrust terms directly input into the equations. Thrust effects
are contained in the basic aerodynamic terms and are input through the tables
using the parameter CT."

The same four equations appear as eq. (4.9) of Robbie's thesis, with the state
written as (V, alpha, theta, q) instead of (gamma, V, alpha, q); the two are
related by theta = gamma + alpha. This model additionally carries Riley's
alpha-dot aerodynamic terms, which that reference omits.
"""
import numpy as np

from symmetric_stall.aircraft.extended_grumman import ExtendedGrumman


class SymmetricFullGrumman(ExtendedGrumman):
    # The CG offset lives in the base class (CG_AFT / CG_RIGHT / CG_BELOW, in
    # metres, see Grumman._delta_momentos_cg). It is NOT re-initialised here:
    # doing so overwrote the value set through the environment and left the
    # plant on Riley's CG while the kernel ran with the shifted one.
    # DXCG_OVER_CHORD still exists as a property, in chord fractions, for
    # paper_cg_sweep_solve.py.

    def derivatives(self, g, v, a, qr, elevator, throttle):
        """The four derivatives, in a single place.

        This used to be a closure inside command_airplane, so verifying the
        kernel against "the CPU" meant hand-copying the expression into the
        verifier -- and a hand copy goes stale silently: it was missing dCD_T,
        and the verifier kept reporting green while comparing the kernel
        against a plant that was no longer this one. Now there is only one.
        """
        v_stall = self.STALL_AIRSPEED
        vt = max(v * v_stall, 0.1)
        q_hat = qr * self.CHORD / (2.0 * vt)

        ct = self._compute_ct(throttle, vt)

        # Bilinear interp between CT=0 and CT=0.5 tables (Riley 1985, Table III)
        cl_o = float(self._bilinear_interp(
            a, ct, self._CL_O_TABLE, self._CL_O_TABLE_CT05))
        cl_q = float(self._bilinear_interp(
            a, ct, self._CL_Q_TABLE, self._CL_Q_TABLE_CT05))
        cl_de = float(self._bilinear_interp(
            a, ct, self._CL_DE_TABLE_CT0, self._CL_DE_TABLE_CT05))
        cl = cl_o + cl_de * elevator + cl_q * q_hat

        cd = float(self._bilinear_interp(
            a, ct, self._CD_O_TABLE, self._CD_O_TABLE_CT05))
        # Riley's drag carries two elevator terms besides the base one, and
        # they are not optional: the elevator is deflected throughout a
        # recovery, so omitting them overstates drag.
        cd += float(self._bilinear_interp(
            a, ct, self._CD_DE_TABLE_CT0,
            self._CD_DE_TABLE_CT05)) * elevator
        cd += float(self._bilinear_interp(
            a, ct, self._CD_DE2_TABLE_CT0,
            self._CD_DE2_TABLE_CT05)) * elevator * elevator
        cd += float(self._delta_cd_thrust(ct, a))
        cm_o = float(self._bilinear_interp(
            a, ct, self._CM_O_TABLE, self._CM_O_TABLE_CT05))
        cm_q = float(self._bilinear_interp(
            a, ct, self._CM_Q_TABLE, self._CM_Q_TABLE_CT05))
        cm_de = float(self._bilinear_interp(
            a, ct, self._CM_DE_TABLE_CT0, self._CM_DE_TABLE_CT05))
        cm = cm_o + cm_de * elevator + cm_q * q_hat

        cl_adot = float(self._bilinear_interp(
            a, ct, self._CL_ADOT_TABLE_CT0, self._CL_ADOT_TABLE_CT05))
        cm_adot = float(self._bilinear_interp(
            a, ct, self._CM_ADOT_TABLE_CT0, self._CM_ADOT_TABLE_CT05))

        qS = 0.5 * self.AIR_DENSITY * self.WING_SURFACE_AREA * vt * vt

        # The alpha-dot terms close an implicit loop: alpha_dot comes from
        # q - gamma_dot, gamma_dot depends on lift, and lift is precisely what
        # these terms modify. It is solved exactly rather than with a
        # fixed-point step, so that the CUDA kernel can evaluate the SAME
        # expression and the two do not drift apart through the choice of
        # scheme.
        #
        #   cl      = cl_base + cl_adot*k*alpha_dot,  k = c/(2V)
        #   gamma_d = A*cl - (g/V)cos(gamma),         A = qS/(m V)
        #   alpha_d = q - gamma_d
        #
        # substituting and solving for gamma_d:
        #
        #   gamma_d = [A*cl_base + A*cl_adot*k*q - (g/V)cos(gamma)]
        #             / (1 + A*cl_adot*k)
        #
        # The denominator stays within 1.000 +/- 0.005 over the grid's whole
        # airspeed range, so there is no risk of it vanishing.
        k = self.CHORD / (2.0 * vt)
        A = qS / (self.MASS * vt)
        g_over_v_cos = (self.GRAVITY / vt) * np.cos(g)

        g_dot = ((A * cl + A * cl_adot * k * qr - g_over_v_cos)
                 / (1.0 + A * cl_adot * k))
        a_dot = qr - g_dot

        # full cl and cm, now with alpha_dot resolved
        cl_full = cl + cl_adot * k * a_dot
        cm_full = cm + cm_adot * k * a_dot

        # transfer to the CG: needs cl and cd ALREADY complete, which is why
        # it goes here and not above. With the CG at Riley's reference all
        # three terms are zero and this is the identity.
        _, dcm, _ = self._delta_momentos_cg(cl_full, cd, a)
        cm_full = cm_full + dcm

        D = qS * cd
        My = qS * self.CHORD * cm_full

        # Thrust is embedded in CT tables — no separate thrust term
        v_dot = (-self.GRAVITY * np.sin(g) - D / self.MASS) / v_stall
        q_dot = My / self.I_YY

        return g_dot, v_dot, a_dot, q_dot

    def command_airplane(self, elevator, throttle):
        """
        RK4 integration matching the CUDA PolicyIteration kernel exactly.
        Uses 2 micro-steps of TIME_STEP/2 each for finer numerical capture
        of the bang-bang singular control transitions. Total control step
        remains TIME_STEP (100 Hz) — only the inner RK4 is finer.

        Operates on normalized velocity (V/Vs) to match the GPU kernel.
        """
        dt_macro = self.TIME_STEP
        n_micro = 10
        dt = dt_macro / n_micro   # 1 ms inner RK4 step
        v_stall = self.STALL_AIRSPEED

        self.last_elevator = elevator
        self.last_throttle = throttle

        def _derivs(g, v, a, qr):
            return self.derivatives(g, v, a, qr, elevator, throttle)

        for _ in range(n_micro):
            gamma = self.flight_path_angle
            vn = self.airspeed_norm
            alpha = self.alpha
            q = self.pitch_rate

            # --- RK4 ---
            k1_g, k1_v, k1_a, k1_q = _derivs(gamma, vn, alpha, q)

            k2_g, k2_v, k2_a, k2_q = _derivs(
                gamma + 0.5 * dt * k1_g, vn + 0.5 * dt * k1_v,
                alpha + 0.5 * dt * k1_a, q + 0.5 * dt * k1_q,
            )

            k3_g, k3_v, k3_a, k3_q = _derivs(
                gamma + 0.5 * dt * k2_g, vn + 0.5 * dt * k2_v,
                alpha + 0.5 * dt * k2_a, q + 0.5 * dt * k2_q,
            )

            k4_g, k4_v, k4_a, k4_q = _derivs(
                gamma + dt * k3_g, vn + dt * k3_v,
                alpha + dt * k3_a, q + dt * k3_q,
            )

            self.flight_path_angle += (dt / 6.0) * (k1_g + 2 * k2_g + 2 * k3_g + k4_g)
            self.airspeed_norm += (dt / 6.0) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)
            self.alpha += (dt / 6.0) * (k1_a + 2 * k2_a + 2 * k3_a + k4_a)
            self.pitch_rate += (dt / 6.0) * (k1_q + 2 * k2_q + 2 * k3_q + k4_q)

        self.airspeed = self.airspeed_norm * v_stall

    def reset(self, flight_path_angle, airspeed_norm, alpha, pitch_rate):
        super().reset(flight_path_angle, airspeed_norm, alpha, 0, 0, 0, pitch_rate, 0)
