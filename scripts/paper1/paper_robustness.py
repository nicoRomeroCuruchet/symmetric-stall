"""
paper_robustness.py — F2a: robustness of the NOMINAL optimal policy to
flight-to-flight variations of mass and longitudinal CG position (4-DOF
symmetric stall recovery, Riley model).

Scenario (operational): the policy was trained for the nominal aircraft
(m = 715.21 kg, CG at the Riley reference 0.25 c-bar). The actual aircraft
of the day differs. The policy is NOT retrained or recalibrated:

  * dynamics use the perturbed MASS and DXCG_OVER_CHORD;
  * the observation normalization keeps the NOMINAL stall speed (the
    policy "believes" the aircraft is nominal);
  * initial conditions are placed relative to the REAL stall speed of the
    perturbed aircraft, Vs_real = Vs_nom * sqrt(m/m_nom) (the physics of
    the stall break belongs to the real aircraft; the error lives only in
    the policy's head);
  * I_yy is NOT perturbed (day-to-day load sits near the CG; its inertia
    contribution is second order) — declared modeling choice.

Reference for the CG axis: the nominal-CG cell of this same matrix. What is
reported is the DEGRADATION of the one policy an operator actually flies, not
its distance to an oracle that re-solves for the CG of the day — that oracle
does not exist in the cockpit, and comparing against it measures something
nobody can act on.

Outputs:
  results/paper/robustness.json
  results/paper/fig_robustness_matrix.{png,pdf}
  results/paper/table_robustness_cg_gap.tex
CPU-only (~10 min).
"""
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from symmetric_stall.aircraft.symmetric_stall import SymmetricStall
from symmetric_stall import paths
from symmetric_stall.procedures import rollout, ctrl_optimal, stamp_engine, dump_json
from symmetric_stall.procedures import ALPHA_GRID_DEG, CANONICAL, VNORM_GRID
from symmetric_stall.policy_iteration import PolicyIterationStall
from symmetric_stall.utils.utils import get_optimal_action

logger = logging.getLogger(__name__)

OUT_DIR = paths.out_dir()

# Both axes on a uniform 2.5 % step. The mass axis used to move in 5 % jumps,
# which is coarse for a quantity whose whole range is one passenger's worth of
# load; the CG axis used to step 2, 2, 3, 3, 2, 2 % of chord, so its contours
# carried a spacing artefact that had nothing to do with the aircraft. The CG
# range moves from +-0.07 to +-0.075 to land on the same grid, which is the
# only thing that changed about what is covered.
MASS_FACTORS = [round(0.85 + 0.025 * i, 4) for i in range(13)]     # 0.85 .. 1.15

# The CG axis runs out to the point where the AIRFRAME breaks, not just over
# the loading variation of a given day. cg_reach.py puts the open-loop
# short-period divergence at 0.453 c-bar; ending the sweep at +0.20 (x_cg =
# 0.45) reaches it, so the matrix can say whether mass and CG stay separable
# all the way there -- something a +-7.5 % window cannot resolve.
#
# Past about +0.075 this is no longer a loading variation but a different
# aeroplane, well outside any certified envelope. CERT_DXCG marks where that
# stops being true, and the figure draws it.
DXCG_LIST = [round(-0.100 + 0.025 * i, 4) for i in range(14)]      # aft-positive
CERT_DXCG = 0.075

#: Aft edge of the plotted matrix, in chord fractions (aft-positive, as the
#: data are stored). 0.150 is 15 % of chord, which the axis labels show as
#: -15 % under the forward-positive convention.
DX_PLOT_MAX_AFT = 0.150
X_CG_REF = 0.25                      # Riley's tables are referenced here
X_CG_DIVERGENT = 0.453               # from cg_reach.py
# CANONICAL comes from procedures.py: it used to be redeclared here as
# (20, 0.95), so moving the evaluation band would have left this file
# indexing a cell its own grid no longer contains.

# WITHDRAWN 2026-07-29. It was a hardcoded dict of GPU re-solves from an old
# campaign (one per CG), used to tabulate the "suboptimality gap" against the
# nominal policy. Two problems:
#
#   1. Nothing regenerated it. When the aerodynamic model was corrected, the
#      nominal column was updated and this one was not, so the table compared
#      two different aircraft. It showed up as the dxcg=0 row giving a gap of
#      -1.29 m where by definition it must give ZERO: same aircraft, same
#      policy.
#
#   2. The question was badly posed. A pilot does not fly a policy
#      re-optimised for that day's actual CG; he flies the nominal one. The
#      distance to an oracle that re-solves is not the quantity of interest.
#
# What does matter -- the degradation of the NOMINAL policy as the CG shifts --
# is already measured by run_matrix, and that is where the table comes from now.

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "stix", "font.size": 10,
    "axes.labelsize": 11, "legend.fontsize": 9,
})


def cell_key(mass_factor: float, dxcg: float) -> str:
    """Key of a (mass, CG) cell in robustness.json.

    Three decimals, not two: at 2.5 % steps a 2-decimal key prints 0.875 as
    0.88 and 0.075 as 0.07, which is both lossy and a lie about which aircraft
    the cell is. Everything that writes or reads the matrix goes through here.
    """
    return f"m{mass_factor:.3f}_dx{dxcg:+.3f}"


NOMINAL_KEY = "m1.000_dx+0.000"


def perturbed_env(mass_factor: float, dxcg: float) -> SymmetricStall:
    """Perturb dynamics only; observation normalization (STALL_AIRSPEED)
    and throttle map (THROTTLE_LINEAR_MAPPING) stay nominal."""
    env = SymmetricStall()
    env.airplane.MASS = env.airplane.MASS * mass_factor
    env.airplane.DXCG_OVER_CHORD = dxcg
    return env


_MATRIX_STATE = {}


def _matrix_init():
    """One policy per worker. The .npz is 119 MB; loading it 91 times would
    cost more than the rollouts."""
    _MATRIX_STATE["pi"] = paths.load_policy(env=SymmetricStall())


def _matrix_cell(job):
    """One (mass, CG) cell: the nine ICs of the shared evaluation grid.

    The unit of work is the CELL, not the rollout, so the perturbed plant is
    built exactly once per cell as in the serial version -- and the nine
    rollouts inside it stay in the order that produced the published numbers.
    """
    mf, dx = job
    pi = _MATRIX_STATE["pi"]
    env = perturbed_env(mf, dx)
    vs_ratio = float(np.sqrt(mf))              # Vs_real / Vs_nominal
    cell = {}
    for a0 in ALPHA_GRID_DEG:
        for v0f in VNORM_GRID:
            # IC at v0f of the REAL stall speed, expressed in the
            # nominal-Vs units the policy (and env state) use.
            vnorm0 = v0f * vs_ratio
            r = rollout(env, pi, ctrl_optimal, a0, vnorm0, record=True)
            amax = float(np.rad2deg(np.max(r["hist"]["alpha"])))
            gmax = float(np.rad2deg(np.max(r["hist"]["gamma"])))
            # alpha AT EPISODE CLOSE. The stopping rule cuts at the first
            # return to gamma = 0 after a dive, and the DP's terminal set is
            # {gamma >= 0} with no condition on alpha: an oscillating
            # trajectory can cross level while still stalled and be declared
            # recovered. Recording it allows those cells to be flagged
            # instead of read as fast recoveries.
            afin = float(np.rad2deg(r["hist"]["alpha"][-1]))
            cell[f"a{a0:.0f}_v{v0f:.2f}"] = {
                "h": r["h"], "t": r["t"], "status": r["status"],
                "alpha_max_deg": amax, "gamma_max_deg": gmax,
                "alpha_final_deg": afin,
            }
    return cell


def run_matrix(pi=None, workers=None):
    """Mass x CG matrix, one process per worker.

    The cells are independent and deterministic, so this is a pure speedup:
    at 2.5 % steps the serial loop was 819 rollouts back to back, near an
    hour, which made every iteration on the figure expensive.
    """
    import multiprocessing as mp
    import os

    data = {"mass_factors": MASS_FACTORS, "dxcg": DXCG_LIST,
            "alpha0_deg": list(ALPHA_GRID_DEG), "v0_frac": list(VNORM_GRID),
            "cells": {}}
    jobs = [(float(mf), float(dx)) for mf in MASS_FACTORS for dx in DXCG_LIST]
    n = workers or max(1, min(12, (os.cpu_count() or 2) - 2))
    logger.info(f"[matrix] {len(jobs)} cells x {len(ALPHA_GRID_DEG)*len(VNORM_GRID)}"
                f" ICs over {n} workers")
    with mp.get_context("spawn").Pool(n, initializer=_matrix_init) as pool:
        out = pool.map(_matrix_cell, jobs, chunksize=1)

    ck = f"a{CANONICAL[0]:.0f}_v{CANONICAL[1]:.2f}"
    for (mf, dx), cell in zip(jobs, out):
        data["cells"][cell_key(mf, dx)] = cell
        logger.info(f"m×{mf:.3f} dx{dx:+.3f}: canonical {cell[ck]['h']:.2f} m "
                    f"({cell[ck]['status']})")
    dump_json(OUT_DIR / "robustness.json", data, indent=1)
    logger.info("[+] robustness.json written")
    return data


def normalization_gap(long_horizon_s: float = 60.0):
    """Isolate the normalisation mismatch on the mass axis.

    `perturbed_env` changes MASS but NOT STALL_AIRSPEED, which grumman.py
    computes in __init__ from the mass. The vnorm observation the policy sees is
    therefore divided by the NOMINAL Vs, not the real one. The "corrected" arm
    recomputes Vs for the real mass and touches nothing else.

    Besides the altitude the metric measures, it integrates `long_horizon_s`
    with no stopping rule, because the metric cuts at the first crossing of
    gamma >= 0 after a dive and that does not distinguish a recovery that
    settles from one that keeps oscillating. In the badly normalised arm the
    light aircraft enters a sustained phugoid: the mis-scaled state falls below
    the grid's vnorm = 0.9 floor, utils.get_barycentric_weights_and_indices
    CLAMPS (it does not extrapolate), and the policy stops responding to
    airspeed. With the normalisation corrected, the same policy settles and
    climbs.

    The three monotone-saving numbers quoted in the robustness section come from
    here. They used to be computed by hand, outside any script.
    """
    pi = paths.load_policy(env=SymmetricStall())
    a0, v0f = CANONICAL
    floor = float(np.load(paths.policy_path(), allow_pickle=True)["bounds_low"][1])

    def free_run(env, vnorm0):
        """Rollout with no stopping rule: h(T) and the gamma swing in the tail."""
        vs = env.airplane.STALL_AIRSPEED
        dt = env.airplane.TIME_STEP
        obs, _ = env.specific_reset(0.0, vnorm0, np.deg2rad(a0), 0.0)
        t = h = 0.0
        ts, hs, gs = [], [], []
        while t < long_horizon_s:
            act = get_optimal_action(obs, pi)[0]
            ts.append(t); hs.append(h); gs.append(np.rad2deg(obs[0]))
            obs, _, _, _, _ = env.step(np.asarray(act, dtype=np.float32))
            h += obs[1] * vs * np.sin(obs[0]) * dt
            t += dt
        ts = np.asarray(ts); tail = ts >= 0.5 * long_horizon_s
        return h, float(np.min(np.asarray(gs)[tail])), \
            float(np.max(np.asarray(gs)[tail]))

    h_nom = rollout(perturbed_env(1.0, 0.0), pi, ctrl_optimal, a0, v0f)["h"]
    rows = []
    for mf in [0.95, 0.90, 0.85]:
        # badly normalised: nominal Vs, IC at 0.95 of the REAL Vs -> vnorm off scale
        env_bad = perturbed_env(mf, 0.0)
        v_bad = v0f * float(np.sqrt(mf))
        r_bad = rollout(env_bad, pi, ctrl_optimal, a0, v_bad)
        h60_bad, gmin_bad, gmax_bad = free_run(perturbed_env(mf, 0.0), v_bad)

        # corregido: Vs de la masa real, misma IC en unidades de su propio Vs
        env_ok = perturbed_env(mf, 0.0)
        env_ok.airplane.STALL_AIRSPEED *= float(np.sqrt(mf))
        r_ok = rollout(env_ok, pi, ctrl_optimal, a0, v0f)
        env_ok2 = perturbed_env(mf, 0.0)
        env_ok2.airplane.STALL_AIRSPEED *= float(np.sqrt(mf))
        h60_ok, gmin_ok, gmax_ok = free_run(env_ok2, v0f)

        rows.append({
            "mass_factor": mf,
            "vnorm0_nominal_norm": round(v_bad, 4),
            "below_grid_floor": bool(v_bad < floor),
            "h_nominal_norm": round(r_bad["h"], 3),
            "h_corrected_norm": round(r_ok["h"], 3),
            "saved_nominal_norm": round(h_nom - r_bad["h"], 3),
            "saved_corrected_norm": round(h_nom - r_ok["h"], 3),
            "h_at_T_nominal_norm": round(h60_bad, 2),
            "h_at_T_corrected_norm": round(h60_ok, 2),
            "gamma_tail_nominal_norm": [round(gmin_bad, 2), round(gmax_bad, 2)],
            "gamma_tail_corrected_norm": [round(gmin_ok, 2), round(gmax_ok, 2)],
        })
        logger.info(
            f"m×{mf:.2f}: mal-norm {r_bad['h']:+.2f} m (ahorro "
            f"{h_nom - r_bad['h']:+.2f}), corregido {r_ok['h']:+.2f} m (ahorro "
            f"{h_nom - r_ok['h']:+.2f}); a {long_horizon_s:.0f} s "
            f"{h60_bad:+.1f} vs {h60_ok:+.1f} m, gamma cola "
            f"[{gmin_bad:.1f},{gmax_bad:.1f}] vs [{gmin_ok:.1f},{gmax_ok:.1f}] deg")

    out = {"h_nominal": round(h_nom, 3), "grid_vnorm_floor": floor,
           "long_horizon_s": long_horizon_s, "canonical": list(CANONICAL),
           "rows": rows}
    dump_json(OUT_DIR / "normalization_gap.json", out, indent=1)
    logger.info("[+] normalization_gap.json written")
    return out


def make_matrix_figure(data=None):
    """Annotated mass x CG matrix at the canonical IC: excess loss of the
    nominal policy on the perturbed aircraft, relative to nominal-on-
    nominal. Diverging scale: purple = worse than nominal, red = better
    (legitimate here: aft CG genuinely loses less)."""
    if data is None:
        data = json.loads((OUT_DIR / "robustness.json").read_text())
    ck = f"a{CANONICAL[0]:.0f}_v{CANONICAL[1]:.2f}"
    M = data["mass_factors"]
    # The stored sweep runs out to +0.225 aft to locate the divergence
    # boundary, but the figure stops at 15 % of chord either side. Past that
    # the aeroplane is outside anything a loading sheet can produce, and the
    # extra columns spent width arguing about an aircraft nobody flies. The
    # DATA keeps them: this is a display crop, not a re-run.
    DX = [dx for dx in data["dxcg"] if dx <= DX_PLOT_MAX_AFT]
    H = np.array([[data["cells"][cell_key(mf, dx)][ck]["h"]
                   for dx in DX] for mf in M])
    h_nom = data["cells"][NOMINAL_KEY][ck]["h"]
    excess = h_nom - H          # positive = worse than nominal-on-nominal

    from matplotlib.colors import TwoSlopeNorm
    # Intuitive polarity for a robustness map: red = loses MORE than
    # nominal (bad), blue = loses less (good), white = equal.
    cmap = plt.cm.RdBu_r
    lim = max(abs(excess.min()), abs(excess.max()), 1.0)
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)

    fig, ax = plt.subplots(figsize=(11.0, 6.2))
    ax.imshow(excess, cmap=cmap, norm=norm, aspect="auto", origin="lower")

    # Threshold for "returned to level STILL STALLED", referred to the nominal
    # cell rather than to alpha_s.
    #
    # A bare alpha_s = 14 deg marked 26 of the 49 cells, the nominal one among
    # them -- and the nominal cell is the reference the whole map is measured
    # against, its excess is +0.0 by construction, so flagging it as a
    # non-comparable recovery contradicts the figure. The cause is that the
    # optimum ENDS pinned to the stall boundary: that is the sliding mode, and
    # over this matrix it closes between 12.61 and 14.51 deg, i.e. within half
    # a degree of the nominal 14.09. Marking it marks the normal terminal state
    # of a well flown recovery.
    #
    # What the mark is for looks nothing like that: a trajectory caught mid
    # second stall reaches 35 deg (the full-pull arms of maneuvers.json). One
    # degree above whichever is higher -- the stall boundary or the reference's
    # own close -- separates the two by a wide margin, and makes it structurally
    # impossible to flag the reference.
    ALPHA_S_DEG = 14.0
    afin_nom = data["cells"][NOMINAL_KEY][ck].get("alpha_final_deg")
    alpha_close_thr = max(ALPHA_S_DEG, afin_nom or ALPHA_S_DEG) + 1.0

    for i, mf in enumerate(M):
        for j, dx in enumerate(DX):
            cellinfo = data["cells"][cell_key(mf, dx)][ck]
            # two different marks, for two different failures of the metric:
            #   *  does not return to level within the horizon
            #   †  returns to level STILL STALLED: the trajectory oscillates
            #      and the gamma = 0 crossing catches it mid second attempt,
            #      so its loss is not comparable with a complete recovery
            afin = cellinfo.get("alpha_final_deg")
            if cellinfo["status"] != "recovered":
                mark = "*"
            elif afin is not None and afin > alpha_close_thr:
                mark = "†"
            else:
                mark = ""
            dark = abs(excess[i, j]) > 0.6 * lim
            ax.text(j, i, f"{excess[i, j]:+.1f}{mark}", ha="center",
                    va="center", fontsize=7.5,
                    color="white" if dark else "black")
    # %+.0f turned the 2.5 % steps into "+2%" and "+8%": one decimal, with
    # the trailing .0 trimmed so whole percentages stay clean.
    def pct(x):
        # `x + 0.0` collapses negative zero: flipping the CG axis to
        # forward-positive turns the 0.0 cell into -0.0, which formats as
        # "-0%" and reads like a real negative offset.
        return f"{x + 0.0:+.1f}%".replace(".0%", "%").replace("-0%", "0%")

    # FORWARD-POSITIVE on the axis, hence the minus: Riley's figure 1 puts the
    # body X axis out the nose ("Arrows indicate positive directions"), and
    # this paper adopts that frame, so a displacement measured in it is
    # positive forward. The plant keeps its own aft-positive CG_AFT and DX
    # untouched -- the sign is flipped HERE, at the presentation layer only, so
    # no stored result changes meaning and no sign can leak into the kernel.
    #
    # The cells are NOT mirrored: the nose stays left and x_cg/c above still
    # runs 0.15 -> 0.50 rightward, as a weight-and-balance chart does. The
    # consequence is an axis that increases leftward, which is exactly what
    # forward-positive means when the aeroplane is drawn nose-left.
    ax.set_xticks(range(len(DX)), [pct(-d * 100) for d in DX],
                  rotation=45, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(M)), [pct((m - 1) * 100) for m in M], fontsize=8.5)
    # Both directions go INSIDE the label rather than at the axis ends: an
    # annotation pinned to the right-hand end collides with the plant stamp in
    # the figure's bottom-right corner.
    # The datum lives in the label because the top x_cg/c axis that used to
    # carry it is gone, and "CG shift" alone invites the question "from what?".
    # Naming it here costs no extra element and answers that at the moment the
    # reader asks it.
    ax.set_xlabel(r"$\leftarrow$ nose      CG shift from $0.25\,\bar{c}$ "
                  r"(% chord, forward positive)      tail $\rightarrow$")
    ax.set_ylabel("Mass change vs. nominal")

    # The secondary x_cg/c axis on top is gone. Besides the clutter, it ran the
    # opposite way to the bottom axis once that became forward-positive -- two
    # x axes on one figure pointing in opposite directions, which is the very
    # confusion the sign change was meant to remove. The reference remains
    # x_cg = 0.25 c (Riley's moment reference), stated in the caption, so an
    # absolute position is still one subtraction away.
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, ax=ax, fraction=0.026, pad=0.02)
    cb.set_label("altitude loss (varied aircraft)\n"
                 "$-$ altitude loss (nominal aircraft)   (m)",
                 fontsize=10)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"fig_robustness_matrix.{ext}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)

    # Report what the marks cover, so a clean map is visibly clean rather than
    # merely unannotated.
    afins = np.array([data["cells"][cell_key(mf, dx)][ck]
                      ["alpha_final_deg"] for mf in M for dx in DX])
    n_star = sum(data["cells"][cell_key(mf, dx)][ck]["status"]
                 != "recovered" for mf in M for dx in DX)
    logger.info(f"[=] alpha at close: {afins.min():.2f}-{afins.max():.2f} deg "
                f"(nominal {afin_nom:.2f}); still-stalled threshold "
                f"{alpha_close_thr:.2f} -> {(afins > alpha_close_thr).sum()} "
                f"marked; {n_star} did not return to level")
    logger.info("[+] fig_robustness_matrix.{png,pdf} written")


def write_cg_gap_table(data=None):
    """Degradation of the nominal policy as the CG shifts, canonical IC and
    nominal mass. The excess is referenced to the nominal CG, so the dxcg=0 row
    gives zero by construction -- a free check that the column and the
    reference come from the same run."""
    if data is None:
        data = json.loads((OUT_DIR / "robustness.json").read_text())
    ck = f"a{CANONICAL[0]:.0f}_v{CANONICAL[1]:.2f}"
    lines = [
        r"\begin{table}[H]", r"    \centering",
        r"    \caption{Degradation of the \emph{nominal} policy with "
        r"longitudinal CG position (canonical IC, nominal mass). The policy "
        r"is trained once at the Riley reference CG ($0.25\,\bar{c}$) and "
        r"flown unchanged on each shifted-CG aircraft, which is what an "
        r"operator does: the CG of the day is not known to the controller. "
        r"The excess column is the altitude cost of that mismatch, "
        r"referenced to the nominal CG, and is zero there by construction.}",
        r"    \label{tab:robustness_cg}",
        r"    \begin{tabular}{c c c}",
        r"        \toprule",
        r"        $\Delta x_{cg}/\bar{c}$ & $\Delta h$ (m) & "
        r"Excess over nominal CG (m) \\",
        r"        \midrule",
    ]
    h_ref = data["cells"][NOMINAL_KEY][ck]["h"]
    for dx in DXCG_LIST:
        k = cell_key(1.00, dx)
        if k not in data["cells"]:
            continue
        h_pol = data["cells"][k][ck]["h"]
        lines.append(f"        ${dx:+.3f}$ & ${h_pol:.2f}$ & "
                     f"${h_ref - h_pol:+.2f}$ \\\\")
    lines += [
        r"        \bottomrule", r"    \end{tabular}", r"\end{table}", "",
    ]
    (OUT_DIR / "table_robustness_cg_gap.tex").write_text("\n".join(lines))
    logger.info("[+] table_robustness_cg_gap.tex written")


def characterize_steady_state(horizon_s: float = 120.0):
    """Extended-horizon characterization of the above-nominal-mass
    failure mode (canonical IC, nominal CG): the closed loop converges
    to a STEADY powered descent. Verifies that the simulated gamma_ss
    equals the force-balance equilibrium sin(g) = -qS*CD/(m g) at the
    policy's own operating point — i.e., the residual descent is
    physical, not a grid artifact. Source of the numbers quoted in the
    robustness subsection of the paper.

    Integrates WITHOUT the stopping rule, on purpose. It used to use
    `rollout`, which carries it, and that broke the +5% case: its steady
    descent is -0.44 deg, inside the +-0.5 deg band of the convergence clause
    in utils/recovery.py, so the rollout cut at 12.4 s and the "steady" value
    came from the average of the last 2 s BEFORE the cut (-0.17 deg), not from
    the real equilibrium. The three heavy rows share the same failure mode; the
    only thing that sets +5% apart is that its descent is shallow enough for
    the rule to read it as level.
    """
    try:
        pi = paths.load_policy(env=SymmetricStall())
        out = {}
        for mf in (1.05, 1.10, 1.15):
            env = perturbed_env(mf, 0.0)
            ap = env.airplane
            dt = ap.TIME_STEP
            obs, _ = env.specific_reset(
                0.0, CANONICAL[1] * float(np.sqrt(mf)),
                np.deg2rad(CANONICAL[0]), 0.0)
            t_ = h_ = 0.0
            hist = {"t": [], "h": [], "gamma": [], "v_norm": [],
                    "alpha": [], "q": [], "de": []}
            crossings = 0
            while t_ < horizon_s:
                act = get_optimal_action(obs, pi)[0]
                hist["t"].append(t_); hist["h"].append(h_)
                hist["gamma"].append(float(obs[0]))
                hist["v_norm"].append(float(obs[1]))
                hist["alpha"].append(float(obs[2]))
                hist["q"].append(float(obs[3]))
                hist["de"].append(float(act[0]))
                prev_g = float(obs[0])
                obs, _, _, _, _ = env.step(np.asarray(act, dtype=np.float32))
                if prev_g < 0.0 <= float(obs[0]) and t_ > 1.0:
                    crossings += 1
                h_ += obs[1] * ap.STALL_AIRSPEED * np.sin(obs[0]) * dt
                t_ += dt
            r = {"hist": hist}
            g_ss = float(np.mean(r["hist"]["gamma"][-200:]))
            v_ss = float(np.mean(r["hist"]["v_norm"][-200:]))
            a_ss = float(np.mean(r["hist"]["alpha"][-200:]))
            h = np.array(r["hist"]["h"]); t = np.array(r["hist"]["t"])
            sink = float(-(h[-1] - h[-500]) / (t[-1] - t[-500]))
            vt = v_ss * ap.STALL_AIRSPEED
            ct = ap._compute_ct(1.0, vt)
            de_ss = float(np.mean(r["hist"]["de"][-200:]))
            # FULL C_D. Until 2026-07-29 this used only _CD_O_TABLE: it was
            # written before the model carried Riley's two elevator drag
            # terms, so it compared an algebraic balance without them against a
            # simulation that does integrate them, and the paper reported an
            # agreement "to two decimals" that was really about 0.5 degrees.
            # The terms matter: in the steady descent the elevator sits at -4
            # to -7 deg and its drag is NOT negligible.
            cd = (float(ap._bilinear_interp(a_ss, ct, ap._CD_O_TABLE,
                                            ap._CD_O_TABLE_CT05))
                  + float(ap._bilinear_interp(a_ss, ct, ap._CD_DE_TABLE_CT0,
                                              ap._CD_DE_TABLE_CT05)) * de_ss
                  + float(ap._bilinear_interp(a_ss, ct, ap._CD_DE2_TABLE_CT0,
                                              ap._CD_DE2_TABLE_CT05))
                  * de_ss * de_ss)
            qS = 0.5 * ap.AIR_DENSITY * ap.WING_SURFACE_AREA * vt * vt
            g_pred = float(np.arcsin(np.clip(
                -qS * cd / (ap.MASS * ap.GRAVITY), -1, 1)))
            out[mf] = {"gamma_ss_deg": np.rad2deg(g_ss),
                       "gamma_balance_deg": np.rad2deg(g_pred),
                       "cd_total": cd, "de_ss_deg": np.rad2deg(de_ss),
                       "sink_mps": sink, "v_ss": v_ss,
                       "alpha_ss_deg": np.rad2deg(a_ss),
                       "gamma_crossings": crossings,
                       "horizon_s": horizon_s}
            logger.info(f"m×{mf}: gamma_ss={np.rad2deg(g_ss):+.2f} "
                        f"(balance {np.rad2deg(g_pred):+.2f}) "
                        f"sink={sink:.2f} m/s  "
                        f"alpha_ss={np.rad2deg(a_ss):.1f}  "
                        f"cruces de gamma>=0 en {horizon_s:.0f} s: {crossings}")
        (OUT_DIR / "robustness_steady_state.json").write_text(
            json.dumps(out, indent=1))
        logger.info("[+] robustness_steady_state.json written")
        return out
    finally:
        pass


def level_flight_feasibility():
    """Infeasibility proof for the overweight steady descent: at the
    policy's steady operating speed there is NO admissible trimmed
    state (alpha <= alpha_s, de_trim in bounds, full throttle) that
    satisfies BOTH level-flight constraints
        CL_trim(alpha, CT) >= 2 m g / (rho V^2 S)   [lift]
        CD_net(alpha, CT) <= 0                       [speed]
    and the minimum speed V* at which both become satisfiable. Source
    of the V* numbers quoted in the robustness subsection."""
    env = SymmetricStall(); ap = env.airplane
    rho, S, g = ap.AIR_DENSITY, ap.WING_SURFACE_AREA, ap.GRAVITY
    Vs, m0 = ap.STALL_AIRSPEED, ap.MASS

    def feasible(mf, vnorm, throttle=1.0):
        m = mf * m0; vt = vnorm * Vs
        qS = 0.5 * rho * S * vt * vt
        ct = ap._compute_ct(throttle, vt)
        cl_lvl = m * g / qS
        for a_deg in np.arange(0.0, 14.01, 0.25):
            a = np.deg2rad(a_deg)
            cm_o = float(ap._bilinear_interp(a, ct, ap._CM_O_TABLE,
                                             ap._CM_O_TABLE_CT05))
            cm_de = float(ap._bilinear_interp(a, ct, ap._CM_DE_TABLE_CT0,
                                              ap._CM_DE_TABLE_CT05))
            de = -cm_o / cm_de
            if not (np.deg2rad(-25) <= de <= np.deg2rad(15)):
                continue
            cl = (float(ap._bilinear_interp(a, ct, ap._CL_O_TABLE,
                                            ap._CL_O_TABLE_CT05))
                  + float(ap._bilinear_interp(a, ct, ap._CL_DE_TABLE_CT0,
                                              ap._CL_DE_TABLE_CT05)) * de)
            # FULL C_D, consistent with the C_L above: this routine already
            # computes the trim elevator and uses it for lift, so ignoring it
            # in the drag was inconsistent with itself. Same origin as the
            # gamma_balance bug: written before the model carried Riley's
            # elevator drag terms.
            cd = (float(ap._bilinear_interp(a, ct, ap._CD_O_TABLE,
                                            ap._CD_O_TABLE_CT05))
                  + float(ap._bilinear_interp(a, ct, ap._CD_DE_TABLE_CT0,
                                              ap._CD_DE_TABLE_CT05)) * de
                  + float(ap._bilinear_interp(a, ct, ap._CD_DE2_TABLE_CT0,
                                              ap._CD_DE2_TABLE_CT05)) * de * de)
            if cl >= cl_lvl and cd <= 0.0:
                return True
        return False

    # V_ss is READ from the characterisation, not hardcoded. It used to be
    # fixed at (1.013, 1.028, 1.043) -- values from an old run that also
    # included the +5% case truncated by the stopping rule. A hardcode here
    # goes silently out of sync every time the model or the policy changes,
    # which is exactly what happened.
    ss = json.loads((OUT_DIR / "robustness_steady_state.json").read_text())
    out = {}
    for mf in (1.05, 1.10, 1.15):
        v_ss = round(float(ss[str(mf)]["v_ss"]), 4)
        vstar = next(vn for vn in np.arange(0.90, 2.01, 0.005)
                     if feasible(mf, vn))
        # Counterfactual: with 30% more available thrust (throttle map
        # extended to 1.3) V* drops below the policy's own operating
        # speed -- the deficit is modest, but the software fix is free.
        vstar13 = next(vn for vn in np.arange(0.90, 2.01, 0.005)
                       if feasible(mf, vn, throttle=1.3))
        out[mf] = {"v_ss_nom": v_ss, "feasible_at_v_ss": feasible(mf, v_ss),
                   "v_star_nom": round(float(vstar), 3),
                   "v_star_over_vs_real": round(float(vstar / np.sqrt(mf)), 3),
                   "v_star_thr13_nom": round(float(vstar13), 3)}
        logger.info(f"m×{mf}: level flight at V_ss "
                    f"{'FEASIBLE' if out[mf]['feasible_at_v_ss'] else 'infeasible'}; "
                    f"V* = {vstar:.3f} Vs_nom = "
                    f"{vstar / np.sqrt(mf):.3f} Vs_real")
    (OUT_DIR / "robustness_feasibility.json").write_text(
        json.dumps(out, indent=1))
    logger.info("[+] robustness_feasibility.json written")
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pi = paths.load_policy(env=SymmetricStall())
    data = run_matrix(pi)
    make_matrix_figure(data)
    write_cg_gap_table(data)


ETA_NOMINAL = 0.903           # propeller efficiency IMPLICIT in the model
ETA_LIST = [0.75, 0.80, 0.85, 0.90, 0.95]


def thrust_sensitivity():
    """What the thrust-model assumption costs.

    Under paper 1's model this measured the cost of a calibration:
    `THROTTLE_LINEAR_MAPPING` did not come from Riley but from assuming the
    aircraft trims at 2*Vs at full throttle, which against the 108 continuous
    hp of Table I implies a propeller efficiency of 0.903 -- high for a
    fixed-pitch propeller, where 0.75-0.85 is typical.

    That calibration no longer exists: Riley's Appendix A supplies T(delta_t, V)
    directly, and under THRUST_MODEL=riley the constant is not read at all.
    Scaling it, as this experiment used to, now perturbs nothing and would
    report a flat table as though the model were insensitive to thrust. The
    perturbation therefore moves onto `THRUST_SCALE`, which multiplies the
    delivered thrust under either model, and the question becomes the one that
    survives the correction: what an engine that underdelivers costs, whatever
    the reason -- a tired engine, a coarse propeller, a hot day.

    Same pattern as the mass and CG matrix: the plant is perturbed, not the
    controller. The policy keeps flying as though the engine were nominal.
    """
    pi = paths.load_policy(env=SymmetricStall())
    data = {"eta_nominal": ETA_NOMINAL, "eta": ETA_LIST,
            "alpha0_deg": list(ALPHA_GRID_DEG), "v0_frac": list(VNORM_GRID),
            "cells": {}}
    base = None
    for eta in ETA_LIST:
        env = SymmetricStall()
        env.airplane.THRUST_SCALE = eta / ETA_NOMINAL
        cell = {}
        for a0 in ALPHA_GRID_DEG:
            for v0 in VNORM_GRID:
                r = rollout(env, pi, ctrl_optimal, a0, v0)
                cell[f"a{a0:.0f}_v{v0:.2f}"] = {
                    "h": float(r["h"]), "t": float(r["t"]),
                    "status": r["status"]}
        data["cells"][f"eta{eta:.2f}"] = cell
        can = cell[f"a{ALPHA_GRID_DEG[-1]:.0f}_v{VNORM_GRID[-1]:.2f}"]["h"]
        if abs(eta - 0.90) < 1e-9:
            base = can
        logger.info(f"eta={eta:.2f}: canonico h={can:.3f} m")
    if base is not None:
        logger.info("--- excess against eta=0.90 ---")
        for eta in ETA_LIST:
            can = data["cells"][f"eta{eta:.2f}"][
                f"a{ALPHA_GRID_DEG[-1]:.0f}_v{VNORM_GRID[-1]:.2f}"]["h"]
            logger.info(f"  eta={eta:.2f}: {can - base:+.3f} m")
    dump_json(OUT_DIR / "thrust_sensitivity.json", data, indent=1)
    logger.info("[+] thrust_sensitivity.json written")


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    if "--figs-only" in sys.argv:
        make_matrix_figure()
        write_cg_gap_table()
    elif "--steady-state" in sys.argv:
        characterize_steady_state()
    elif "--feasibility" in sys.argv:
        level_flight_feasibility()
    elif "--thrust" in sys.argv:
        thrust_sensitivity()
    elif "--normalization" in sys.argv:
        normalization_gap()
    else:
        main()
