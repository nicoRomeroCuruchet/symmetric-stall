"""The transfer of moments to the CG, against physics and against itself.

Cuatro cosas, todas repetibles:

  1. With the CG where Riley takes it (0.25 c on the fuselage centreline,
     report p. 5 and Table I) the transfer must be the BIT-FOR-BIT IDENTITY
     on the plant derivatives. If it is not,
     adding the transfer moved published results.
  2. With the CG shifted, the three deltas must agree with a cross
     product computed SEPARATELY in this same file -- not by calling the
     plant, which is what is under test:
         M_CG = M_ref + (r_ref - r_CG) x F,  F/qS = (-C_A, C_Y, -C_N)
  3. The sign, against physics: an aft CG with positive lift must give
     NOSE-UP (less stable). Poliak's thesis (2025) p. 25 eqs. (12) and (16)
     writes the transfer with the opposite sign; this check is what makes it
     impossible to copy that without noticing.
  4. DXCG_OVER_CHORD is still the chord fraction, which is what
     paper_cg_sweep_solve.py y paper_fig_cg_sweep.py.

The CG can be set through the environment (CG_AFT_M / CG_RIGHT_M /
CG_BELOW_M, in metres), just like THRUST_MODEL, and from there it is also
picked up by the config of
PolicyIteration y verificar_cpu_vs_kernel.py.

    THRUST_MODEL=riley PYTHONPATH=. python verificar_traslado_cg.py
"""

import importlib, itertools, sys
import numpy as np

MOD = (("8dof", "aircraft.spin_grumman", "SpinGrumman"),
       ("6dof", "aircraft.banked_spin_grumman", "BankedSpinGrumman"),
       ("4dof", "aircraft.symmetric_full_grumman", "SymmetricFullGrumman"))

def load():
    for n, m, c in MOD:
        try:
            return getattr(importlib.import_module(m), c)(), n
        except ImportError:
            pass
    raise SystemExit("sin modelo")

a, name = load()

# --- test states ---
if name == "4dof":
    STATES = [(g, v, al, q, de, dt)
               for g in (-0.3, 0.0) for v in (0.6, 1.0, 1.6)
               for al in np.deg2rad((-5., 5., 14., 25., 35.))
               for q in (-0.5, 0.0, 0.5) for de in (-0.4, 0.0, 0.4)
               for dt in (0.0, 1.0)]
    call_plant = lambda: [a.derivatives(*s) for s in STATES]
elif name == "6dof":
    STATES = [(g, v, al, q, mu, p, de, da, dt)
               for g in (-0.3, 0.0) for v in (0.6, 1.0, 1.6)
               for al in np.deg2rad((-5., 5., 14., 25., 35.))
               for q in (-0.5, 0.5) for mu in (-0.5, 0.0, 0.5) for p in (-0.3, 0.3)
               for de in (-0.4, 0.4) for da in (-0.2, 0.2) for dt in (0.0, 1.0)]
    call_plant = lambda: [a.derivatives(*s) for s in STATES]
else:
    STATES = [(g, v, al, be, q, mu, p, r, de, da, dr, dt)
               for g in (-0.3, 0.0) for v in (0.6, 1.0, 1.6)
               for al in np.deg2rad((-5., 5., 14., 25., 35.))
               for be in np.deg2rad((-10., 0., 10.))
               for q in (-0.5, 0.5) for mu in (-0.5, 0.5) for p in (-0.3, 0.3)
               for r in (-0.2, 0.2) for de in (-0.4, 0.4) for da in (-0.2, 0.2)
               for dr in (-0.2, 0.2) for dt in (0.0, 1.0)][:4000]
    call_plant = lambda: [a._derivatives(s[0], s[1], s[2], s[3], s[5], s[6], s[4],
                                     s[7], s[8], s[9], s[11], s[10],
                                     a.STALL_AIRSPEED) for s in STATES]

print("model %s -- %d test states" % (name, len(STATES)))

# ---- (1) identity with the CG at Riley's reference ----
base = np.array(call_plant(), dtype=np.float64)
a.CG_AFT = 0.0; a.CG_RIGHT = 0.0; a.CG_BELOW = 0.0
other = np.array(call_plant(), dtype=np.float64)
ident = np.array_equal(base, other)
print("  (1) CG en la referencia -> identidad bit a bit: %s" % ("SI" if ident else "NO"))

# ---- (2) the transfer against an explicit cross product ----
# M_CG = M_ref + (r_ref - r_CG) x F, in body axes (x forward, y right,
# z down). F_x = -C_A, F_y = C_Y, F_z = -C_N (non-dimensionalised by qS).
rng = np.random.default_rng(0)
worst = 0.0
for _ in range(200):
    dx, dy, dz = rng.uniform(-0.3, 0.3, 3)
    cl_, cd_, cy_ = rng.uniform(-1.5, 2.0), rng.uniform(0.0, 1.0), rng.uniform(-0.4, 0.4)
    al = rng.uniform(-0.2, 0.7)
    a.CG_AFT, a.CG_RIGHT, a.CG_BELOW = dx, dy, dz
    obt = np.array(a._delta_momentos_cg(cl_, cd_, al, cy=cy_))

    c_n = cl_ * np.cos(al) + cd_ * np.sin(al)
    c_a = cd_ * np.cos(al) - cl_ * np.sin(al)
    r_cg = np.array([-dx, dy, dz])          # CG position in body axes
    F = np.array([-c_a, cy_, -c_n])
    dM = np.cross(-r_cg, F)                 # (r_ref - r_CG) x F, with r_ref = 0
    esp = np.array([dM[0] / a.WING_SPAN, dM[1] / a.CHORD, dM[2] / a.WING_SPAN])
    worst = max(worst, float(np.max(np.abs(obt - esp))))
print("  (2) against the explicit cross product: worst %.2e" % worst)

# ---- (3) an aft CG must destabilise ----
a.CG_AFT, a.CG_RIGHT, a.CG_BELOW = 0.0, 0.0, 0.0
al14 = np.deg2rad(14.0)
_, dcm0, _ = a._delta_momentos_cg(1.26, 0.24, al14)
a.CG_AFT = 0.05 * a.CHORD
_, dcm_aft, _ = a._delta_momentos_cg(1.26, 0.24, al14)
a.CG_AFT = -0.05 * a.CHORD
_, dcm_fwd, _ = a._delta_momentos_cg(1.26, 0.24, al14)
print("  (3) alpha=14, C_L=1.26: CG 5%% aft dCm=%+.5f (nose-up, less stable); "
      "5%% forward dCm=%+.5f" % (dcm_aft, dcm_fwd))
ok3 = dcm_aft > 0 > dcm_fwd

# ---- (4) DXCG_OVER_CHORD still works ----
a.CG_AFT = a.CG_RIGHT = a.CG_BELOW = 0.0
a.DXCG_OVER_CHORD = 0.05
ok4 = abs(a.CG_AFT - 0.05 * a.CHORD) < 1e-12 and abs(a.DXCG_OVER_CHORD - 0.05) < 1e-12
print("  (4) DXCG_OVER_CHORD is still the chord fraction: %s" % ("YES" if ok4 else "NO"))

sys.exit(0 if (ident and worst < 1e-15 and ok3 and ok4) else 1)
