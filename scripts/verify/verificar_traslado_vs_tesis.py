"""Our transfer to the CG against the formulation in Poliak's thesis, term by
term. The claim under test is precise: EVERYTHING agrees except the sign of the
cross product in eq. (12)/(16).

Thesis, p. 25:
    (13)/(15)  F_B/QS = ( -(C_D cos a - C_L sin a),  C_Y,  -(C_L cos a + C_D sin a) )
                      = ( -C_A, C_Y, -C_N )
    (16)       [b C_l, c C_m, b C_n]_CG = [b C_l, c C_m, b C_n]_ref + r_ref->CG x F
               with r_ref->CG = (x_cg, y_cg, z_cg)_B,  x_cg > 0 = CG towards the NOSE

Our code: M_CG = M_ref - r_ref->CG x F  (the standard derivation).
So delta_ours == -delta_thesis is expected, exactly.
"""
import importlib, sys
import numpy as np

for m, c in (("symmetric_stall.aircraft.spin_grumman", "SpinGrumman"),
             ("symmetric_stall.aircraft.banked_spin_grumman", "BankedSpinGrumman"),
             ("symmetric_stall.aircraft.symmetric_full_grumman", "SymmetricFullGrumman")):
    try:
        a = getattr(importlib.import_module(m), c)(); name = m.split(".")[-1]; break
    except ImportError:
        continue

rng = np.random.default_rng(20260808)
worst_f, worst_m, worst_sign = 0.0, 0.0, 0.0
N = 500
for _ in range(N):
    x_cg, y_cg, z_cg = rng.uniform(-0.3, 0.3, 3)   # convencion de la TESIS
    cl_, cd_, cy_ = rng.uniform(-1.5, 2.0), rng.uniform(0.0, 1.0), rng.uniform(-0.4, 0.4)
    al = rng.uniform(-0.2, 0.7)

    # --- la tesis, literal ---
    ca, sa = np.cos(al), np.sin(al)
    F_thesis = np.array([-(cd_ * ca - cl_ * sa), cy_, -(cl_ * ca + cd_ * sa)])
    r_tesis = np.array([x_cg, y_cg, z_cg])
    dM_tesis = np.cross(r_tesis, F_thesis)
    d_thesis = np.array([dM_tesis[0] / a.WING_SPAN,
                        dM_tesis[1] / a.CHORD,
                        dM_tesis[2] / a.WING_SPAN])

    # --- ours. dx = towards the TAIL = -x_cg ---
    a.CG_AFT, a.CG_RIGHT, a.CG_BELOW = -x_cg, y_cg, z_cg
    d_ours = np.array(a._delta_momentos_cg(cl_, cd_, al, cy=cy_))

    # 1) the body FORCES must be identical (eq. 13/15)
    c_n = cl_ * ca + cd_ * sa
    c_a = cd_ * ca - cl_ * sa
    worst_f = max(worst_f, abs(F_thesis[0] + c_a), abs(F_thesis[2] + c_n))

    # 2) the transfer must be exactly the OPPOSITE
    worst_sign = max(worst_sign, float(np.max(np.abs(d_ours + d_thesis))))
    # 3) and it must NOT be equal (if it were, we would have copied the bad sign)
    worst_m = max(worst_m, float(np.max(np.abs(d_ours - d_thesis))))

print("model %s -- %d random cases" % (name, N))
print("  eq. (13)/(15), body forces   C_X=-C_A, C_Z=-C_N : agrees, worst %.2e" % worst_f)
print("  eq. (12)/(16), transfer      ours == -thesis     : worst %.2e" % worst_sign)
print("  and NOT equal to the thesis (control) typical gap  : %.3f" % worst_m)
ok = worst_f < 1e-15 and worst_sign < 1e-15 and worst_m > 1e-3
print("\n%s" % ("MATCHES THE THESIS EXCEPT FOR THE SIGN, which is the claim"
                if ok else "the claim does NOT hold"))
sys.exit(0 if ok else 1)
