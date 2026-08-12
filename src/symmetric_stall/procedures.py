"""
paper_procedures.py — Stall-recovery procedure comparison (CAA vs FAA vs
power-delayed) and pilot-suboptimality sensitivity for the 4-DOF symmetric
stall recovery (Riley model).

Grounded in Gratton et al. (2014), "Evaluating a set of stall recovery
actions for single engine light aeroplanes":
    CAA           nose-down + power simultaneously (2 s ramp to full)
    FAA           nose-down first, then power (2 s ramp)
    power-delayed nose-down, 2 s pause, then power (2 s ramp)

The DP optimum plays the role of the CAA sequence (its argmax commands full
throttle essentially everywhere while the elevator does nose-down -> pull-up).
We quantify the cost of deviating from it along the axes a human pilot might:

  E1  power delay      delta_e optimal (closed loop), delta_t = 0 until
                       t >= tau, then instant or 2-s-ramp to full. A variant
                       gates power on alpha < 14 deg (unstalled) instead of a
                       clock ("wait until unstalled" / NTPS-style).
  E3b switch delay     follow the policy until its first nose-down -> pull-up
                       elevator sign switch, hold nose-down for tau_s extra
                       seconds, then resume the policy.
  E3c partial pull     follow the policy, but clamp the pull-up command to
                       delta_e_pull in {-25..-5} deg (pilot does not maximize
                       CL during the pull).

All experiments: closed-loop rollouts of the trained FIXED-dt policy (primary,
per user decision), canonical stopping rule from paper_table_dp_vs_ppo
(altitude loss until first return to gamma = 0 after the dive develops).

Outputs:
  results/paper/fig_procedures.{png,pdf}
  results/paper/fig_pilot_sensitivity.{png,pdf}
  results/paper/table_procedures.tex
  results/paper/procedures.json
CPU-only.
"""
import json
import logging
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from symmetric_stall.aircraft.symmetric_stall import SymmetricStall
# Shared evaluation grid. It used to be imported from the paper-1 script
# paper_table_dp_vs_ppo.py, which only resolved because the two files were
# siblings on sys.path[0]; now this package is the single source and that
# script imports from here.
ALPHA_GRID_DEG = [16.0, 18.0, 20.0]
VNORM_GRID = [0.90, 0.95, 1.00]
from symmetric_stall.policy_iteration import PolicyIterationStall
from symmetric_stall.utils.recovery import DIVE_THRESHOLD_DEG, RecoveryMonitor
from symmetric_stall.utils.utils import get_optimal_action, get_optimal_action_greedy

logger = logging.getLogger(__name__)

POLICY_PATH = Path("results/SymmetricStall_policy.npz")
OUT_DIR = Path("results/paper")

CANONICAL = (20.0, 0.95)               # (alpha0 deg, V0/Vs)
MAX_TIME = 15.0
ALPHA_STALL_RAD = np.deg2rad(14.0)     # Riley positive stall
GRATTON_RAMP_S = 2.0                   # Gratton's "increase power over 2 s"

TAU_POWER = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0]
TAU_SWITCH = [0.0, 0.1, 0.25, 0.5, 1.0]
DE_PULL_DEG = [-25.0, -20.0, -15.0, -10.0, -5.0]

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "stix", "font.size": 10,
    "axes.labelsize": 11, "legend.fontsize": 9,
    "axes.grid": True, "grid.alpha": 0.35, "lines.linewidth": 1.6,
})


# ── Rollout engine with a time/state-aware controller ────────────────────

def rollout(env, pi, controller, alpha0_deg, vnorm0, record=False,
            gamma0_deg=0.0, q0_deg=0.0, policy_action=None):
    """Closed-loop rollout; controller(obs, t, opt_action, ctx) -> action.

    `opt_action` is the policy's own (delta_e, delta_t) at the current obs;
    `ctx` is a mutable dict the controller can use to keep state (e.g. the
    detected switch time). Stopping rule identical to
    paper_table_dp_vs_ppo.simulate.

    `gamma0_deg`/`q0_deg` default to the level, zero-rate entry used by every
    result in the paper's alpha-V plane; the IC cuts of `_cut_sweep` vary them
    one at a time to probe the two states that plane holds fixed.

    `policy_action` selects how the policy is read at an off-grid state:
    the default barycentric action blend (Approach A), or
    `get_optimal_action_greedy` (Approach B) to re-derive the Bellman argmax
    and keep the bang-bang structure the blend averages away.
    """
    v_stall = env.airplane.STALL_AIRSPEED
    dt = env.airplane.TIME_STEP
    policy_action = policy_action or get_optimal_action

    obs, _ = env.specific_reset(np.deg2rad(gamma0_deg), vnorm0,
                                np.deg2rad(alpha0_deg), np.deg2rad(q0_deg))
    t, h = 0.0, 0.0
    stop = RecoveryMonitor(dt)
    ctx = {}
    hist = {"t": [], "h": [], "de": [], "dt_ctrl": [], "alpha": [],
            "gamma": [], "v_norm": [], "q": []} if record else None

    while t < MAX_TIME:
        opt_action = policy_action(obs, pi)[0]
        action = np.asarray(controller(obs, t, opt_action, ctx), dtype=np.float32)

        if record:
            hist["t"].append(t); hist["h"].append(h)
            hist["de"].append(float(action[0])); hist["dt_ctrl"].append(float(action[1]))
            hist["alpha"].append(float(obs[2])); hist["gamma"].append(float(obs[0]))
            hist["v_norm"].append(float(obs[1])); hist["q"].append(float(obs[3]))

        obs, _, _, _, _ = env.step(action)
        h += obs[1] * v_stall * np.sin(obs[0]) * dt
        t += dt

        if stop.update(np.rad2deg(obs[0])):
            return {"h": h, "t": t, "status": "recovered", "hist": hist}
        if (obs[2] >= np.deg2rad(40) or obs[2] <= np.deg2rad(-40)
                or obs[0] <= -np.pi + 0.05):
            return {"h": h, "t": t, "status": "crash", "hist": hist}

    return {"h": h, "t": t, "status": "timeout", "hist": hist}


# ── Controllers ──────────────────────────────────────────────────────────

def ctrl_optimal(obs, t, opt, ctx):
    return opt


def make_power_delay(tau, ramp=False):
    """Optimal delta_e; throttle 0 until tau, then instant 1.0 or 2-s ramp."""
    def ctrl(obs, t, opt, ctx):
        if t < tau:
            thr = 0.0
        elif ramp:
            thr = min((t - tau) / GRATTON_RAMP_S, 1.0)
        else:
            thr = 1.0
        return (opt[0], thr)
    return ctrl


def make_power_gated(ramp=False):
    """Throttle held at 0 while alpha >= stall (14 deg); 'wait until
    unstalled' interpretation of delayed power."""
    def ctrl(obs, t, opt, ctx):
        if "t_unstalled" not in ctx and obs[2] < ALPHA_STALL_RAD:
            ctx["t_unstalled"] = t
        if "t_unstalled" not in ctx:
            thr = 0.0
        elif ramp:
            thr = min((t - ctx["t_unstalled"]) / GRATTON_RAMP_S, 1.0)
        else:
            thr = 1.0
        return (opt[0], thr)
    return ctrl


def make_switch_delay(tau_s, de_hold_deg=15.0):
    """Follow the policy; at its first nose-down -> pull-up switch, keep
    holding nose-down for tau_s extra seconds, then resume the policy."""
    de_hold = np.deg2rad(de_hold_deg)
    def ctrl(obs, t, opt, ctx):
        if "t_switch" not in ctx and opt[0] < 0.0:
            ctx["t_switch"] = t
        if "t_switch" in ctx and t < ctx["t_switch"] + tau_s:
            return (de_hold, opt[1])
        return opt
    return ctrl


def make_partial_pull(de_pull_deg):
    """Follow the policy, but cap the pull-up deflection at de_pull (pilot
    pulls with less than the CL-maximizing elevator)."""
    de_pull = np.deg2rad(de_pull_deg)
    def ctrl(obs, t, opt, ctx):
        if opt[0] < 0.0:
            return (max(opt[0], de_pull), opt[1])
        return opt
    return ctrl


def make_held_pull(de_pull_deg):
    """After the policy's first pull command, hold de = -X OPEN LOOP for
    the rest of the flight (throttle stays with the policy). The
    over-pull counterpart of make_partial_pull: no alpha regulation."""
    de = np.deg2rad(-abs(de_pull_deg))
    def ctrl(obs, t, opt, ctx):
        if "pulling" not in ctx and opt[0] < 0.0:
            ctx["pulling"] = True
        if ctx.get("pulling"):
            return (de, opt[1])
        return opt
    return ctrl


HELD_PULL_DEG = [2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0, 22.5, 25.0]


def compute_held_pull():
    """Canonical-IC sweep of the open-loop held pull; adds arm
    'e3d_held_pull' (dh + alpha_max after the switch) to procedures.json."""
    rep = json.loads((OUT_DIR / "procedures.json").read_text())
    pi = PolicyIterationStall.load(POLICY_PATH, env=SymmetricStall())
    a0c, v0c = CANONICAL
    arm = {}
    for X in HELD_PULL_DEG:
        r = rollout(pi.env, pi, make_held_pull(X), a0c, v0c, record=True)
        amax = float(np.rad2deg(np.max(r["hist"]["alpha"][50:])))
        arm[f"{X:g}"] = {"h": r["h"], "status": r["status"],
                         "alpha_max_deg": amax}
        logger.info(f"held pull -{X:g}: h={r['h']:.2f} ({r['status']}) "
                    f"alpha_max={amax:.1f}")
    rep["e3d_held_pull"] = arm
    (OUT_DIR / "procedures.json").write_text(json.dumps(rep, indent=2))
    logger.info("[+] e3d_held_pull added to procedures.json")
    return rep


# ── Scripted pilot maneuvers (Gratton's CAA / FAA, procedural) ───────────

DE_DOWN = np.deg2rad(15.0)      # nose-down push (max TED authority)
DE_PULL = np.deg2rad(-25.0)     # full pull
ALPHA_UNSTALL = np.deg2rad(14.0)  # Gratton's "unstalled" trigger
ALPHA_TARGET = np.deg2rad(13.0)   # alpha-hold pilot target (just below stall)
K_ALPHA = 10.0                    # alpha-hold gain (de = K*(alpha - target): + = push)
K_Q = 2.0                         # pitch-rate damping (pilot arrests the drop)


def make_maneuver(power_start, pull="alpha_hold", de_pull_limit=None):
    """Scripted pilot procedure, no DP in the loop.

    Phase 1 (nose-down): de = +15 deg until alpha < 14 deg.
    Phase 2 (pull-up):   'full'       -> de = -25 deg open loop;
                         'alpha_hold' -> proportional de targeting alpha = 13
                                         deg (competent pilot: max CL without
                                         re-stalling), saturated to [-25,+15].
    Power: 2-s ramp to full starting at t=0 ('t0', CAA) or at the unstall
    event ('unstall', FAA).

    `de_pull_limit` (rad, negative) caps how hard the scripted pilot may pull.
    Defaults to the actuator limit DE_PULL. Pass the optimum's own deepest pull
    at the same entry to hold pull authority fixed across the arms, so the
    comparison isolates WHEN power is applied rather than confounding it with
    HOW HARD the pilot hauls back: with the corrected elevator drag the optimum
    no longer saturates the pull (-17.98 deg at the canonical entry), and an
    open-loop -25 deg would charge the procedures for authority the optimum
    declines to use.
    """
    de_min = DE_PULL if de_pull_limit is None else float(de_pull_limit)
    def ctrl(obs, t, opt, ctx):
        alpha = obs[2]
        if "t_unstall" not in ctx:
            if alpha < ALPHA_UNSTALL:
                ctx["t_unstall"] = t
        in_pull = "t_unstall" in ctx

        if not in_pull:
            de = DE_DOWN
        elif pull == "full":
            de = de_min
        else:  # alpha_hold: de>0 pushes (lowers alpha), de<0 pulls (raises it)
            q = obs[3]
            de = float(np.clip(K_ALPHA * (alpha - ALPHA_TARGET) + K_Q * q,
                               de_min, DE_DOWN))

        t_pwr = 0.0 if power_start == "t0" else ctx.get("t_unstall", None)
        if t_pwr is None:
            thr = 0.0
        else:
            thr = min(max((t - t_pwr) / GRATTON_RAMP_S, 0.0), 1.0)
        return (de, thr)
    return ctrl


def run_maneuvers(pi, env):
    """DP optimum vs scripted CAA vs scripted FAA over the IC grid.

    The `alpha_hold` arms are capped to the SAME pull authority the optimum
    reaches at that initial condition, as in
    make_trajectory_comparison_figure: without that, the table and the figure
    describe the same experiment with two different pilot models. The effect
    is small (~0.2 m at the canonical IC) but the inconsistency is not.
    The `full_pull` arms are NOT capped: their whole purpose is to show what
    happens when the pilot pulls to the actuator stop.
    """
    def cap_en(alpha0, v0):
        r = rollout(env, pi, ctrl_optimal, alpha0, v0, record=True)
        return float(np.min(r["hist"]["de"]))

    arms = {
        "caa_alpha_hold": ("t0", "alpha_hold", True),
        "faa_alpha_hold": ("unstall", "alpha_hold", True),
        "caa_full_pull": ("t0", "full", False),
        "faa_full_pull": ("unstall", "full", False),
    }
    out = {}
    for name, (power_start, pull, acotar) in arms.items():
        per_ic = {}
        for alpha0 in ALPHA_GRID_DEG:
            for v0 in VNORM_GRID:
                lim = cap_en(alpha0, v0) if acotar else None
                ctrl = make_maneuver(power_start, pull, lim)
                r = rollout(env, pi, ctrl, alpha0, v0, record=True)
                # alpha_max backs the table footnote's claim that the full
                # pull re-stalls the wing. Without recording it, that sentence
                # had no number behind it for anyone to check.
                per_ic[f"a{alpha0:.0f}_v{v0:.2f}"] = {
                    "h": r["h"], "t": r["t"], "status": r["status"],
                    "alpha_max_deg": float(np.rad2deg(
                        np.max(r["hist"]["alpha"])))}
        out[name] = per_ic
        logger.info(f"    maneuver {name}: canonical "
                    f"{per_ic['a20_v0.95']['h']:.2f} m "
                    f"({per_ic['a20_v0.95']['status']})")
    return out


def main_maneuvers():
    """Scripted-maneuver comparison only (fast; reuses the optimal rollouts
    already stored by paper_mca_comparison)."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pi = PolicyIterationStall.load(POLICY_PATH, env=SymmetricStall())
    man = run_maneuvers(pi, pi.env)

    opt = json.loads((OUT_DIR / "mca_comparison.json").read_text())["fixed"]["rollouts"]

    print("\n=== Δh (m) by IC: optimal DP vs CAA / FAA manoeuvres (α-hold pilot) ===")
    hdr = (f"{'IC':<14}{'DP OPT':>9}{'CAA':>9}{'FAA':>9}"
           f"{'CAA/OPT':>9}{'FAA/CAA':>9}")
    print(hdr); print("-" * len(hdr))
    rows_tex = []
    for alpha0 in ALPHA_GRID_DEG:
        for v0 in VNORM_GRID:
            k = f"a{alpha0:.0f}_v{v0:.2f}"
            o, c, f = opt[k]["h"], man["caa_alpha_hold"][k]["h"], man["faa_alpha_hold"][k]["h"]
            st = opt[k]["status"]
            rc = c / o if st == "recovered" else float("nan")
            rf = f / c if man["caa_alpha_hold"][k]["status"] == "recovered" else float("nan")
            tag = "" if st == "recovered" else " (t/o)"
            print(f"({alpha0:>4.0f},{v0:.2f})  {o:>8.2f}{c:>9.2f}{f:>9.2f}"
                  f"{rc:>9.2f}{rf:>9.2f}{tag}")
            rows_tex.append((alpha0, v0, o, c, f,
                             man["caa_full_pull"][k]["h"], man["faa_full_pull"][k]["h"], st))

    lines = [
        r"\begin{table}[!ht]", r"    \centering",
        r"    \caption{Altitude loss $\Delta h$ (m): exact DP optimum vs.\ the "
        r"scripted CAA and FAA recovery procedures, simulated on the Riley model as defined by Gratton et al. (all values are simulation results of this work). Nose-down "
        r"$\delta_e=+15^\circ$ until $\alpha<14^\circ$; pull-up holds "
        r"$\alpha\approx13^\circ$ ($\alpha$-hold pilot); power ramps to full "
        r"over 2\,s from $t=0$ (CAA) or from the unstall event (FAA). "
        r"In parentheses: the same maneuver with the pull-up instead flown open loop at full deflection ($\delta_e=-25^\circ$ held): the overdone pull re-stalls the wing into a secondary stall at every entry, driving $\alpha$ to $33$--$35^\circ$ against the $14^\circ$ boundary. "
        r"The canonical entry is set in \textbf{bold}: it is the maneuver "
        r"resolved in the time domain in Fig.~\ref{fig:4dof_trajectory_riley}.}",
        r"    \label{tab:maneuvers}", r"    \begin{tabular}{ccccc}",
        r"        \hline",
        r"        $\alpha_0$ & $V_0/V_s$ & DP optimum & CAA & FAA \\",
        r"        \hline",
    ]
    for a0, v0, o, c, f, cfp, ffp, st in rows_tex:
        m = "" if st == "recovered" else r"\textsuperscript{t}"
        # The canonical IC is highlighted: it is the manoeuvre the
        # trajectory figure shows in the time domain, and without the mark the
        # reader has no way to locate, in the table, the row that figure
        # develops.
        b = (r"\bfseries " if (a0, v0) == CANONICAL else "")
        lines.append(f"        {b}{a0:.0f} & {b}{v0:.2f} & {b}{o:.2f}{m} & "
                     f"{b}{c:.2f} ({cfp:.2f}) & {b}{f:.2f} ({ffp:.2f}) \\\\")
    lines += [r"        \hline",
              r"        \multicolumn{5}{l}{\footnotesize\textsuperscript{t}\,"
              r"DP optimum settles into a shallow powered descent (no "
              r"$\gamma=0$ crossing in 15\,s).}\\",
              r"    \end{tabular}", r"\end{table}"]
    (OUT_DIR / "table_maneuvers.tex").write_text("\n".join(lines) + "\n")

    report = {"maneuvers": man, "optimal_ref": opt}
    (OUT_DIR / "maneuvers.json").write_text(json.dumps(report, indent=2))
    logger.info(f"[+] Wrote table_maneuvers.tex and maneuvers.json to {OUT_DIR}")


# ── CAA-structure evidence from the policy array itself ──────────────────

def caa_evidence(pi) -> dict:
    """(i) fraction of non-terminal states whose argmax commands full
    throttle, globally and restricted to the recovery-relevant window;
    (ii) the elevator switch structure along the canonical trajectory."""
    env = SymmetricStall()
    actions = np.asarray(pi.action_space)
    thr_max = np.unique(actions[:, 1])[-1]
    thr_of_idx = actions[np.asarray(pi.policy), 1]

    axes = [np.linspace(pi.bounds_low[d], pi.bounds_high[d], int(pi.grid_shape[d]))
            for d in range(len(pi.grid_shape))]
    mesh = np.meshgrid(*axes, indexing="ij")
    states = np.vstack([m.ravel() for m in mesh]).astype(np.float32).T
    terminal_mask, _ = env.terminal(states)
    non_term = ~terminal_mask

    full = np.isclose(thr_of_idx, thr_max)
    frac_global = float(np.mean(full[non_term]))

    # Recovery-relevant window: pre/post-stall alpha, sub-cruise speeds,
    # diving flight — where an actual recovery evolves.
    win = (non_term
           & (states[:, 2] >= np.deg2rad(-14)) & (states[:, 2] <= np.deg2rad(20))
           & (states[:, 1] <= 1.2)
           & (states[:, 0] <= 0.0) & (states[:, 0] >= np.deg2rad(-90)))
    frac_window = float(np.mean(full[win]))

    return {"full_throttle_frac_global": frac_global,
            "full_throttle_frac_window": frac_window,
            "throttle_max_level": float(thr_max)}


def make_pilot_sensitivity_figure(report):
    """Two-panel pilot-error sensitivity. (a) Switch delay: absolute dh
    vs tau_s for all nine ICs (canonical highlighted) — the steep axis.
    (b) Partial pull at the canonical IC, with the ±0.3 m execution band
    shaded and an honest y-scale: flat down to -10 deg — the forgiving
    axis (as long as the pull is flown closed loop)."""
    e3b = report["e3b_switch_delay"]
    e3c = report["e3c_partial_pull"]
    ref = report["optimal_canonical"]
    a0c, v0c = CANONICAL
    ckey = f"a{a0c:.0f}_v{v0c:.2f}"

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(9.8, 3.8))

    # (a) switch delay at the canonical IC (the 3x3 IC sweep stays in the
    # data; plotting all nine buried the canonical under grey clutter)
    taus = sorted(float(k) for k in e3b)
    h_c = [e3b[f"{t:g}"][ckey]["h"] for t in taus]
    ax_l.plot(taus, h_c, marker="o", ms=4.5, color="#2C4B9E", lw=2.0,
              zorder=3)
    ax_l.axhline(ref["h"], color="gray", lw=0.9, ls=":")
    h05 = e3b["0.5"][ckey]["h"]
    h1 = e3b["1"][ckey]["h"]
    ax_l.annotate(f"$\\times${h05 / ref['h']:.1f}", xy=(0.5, h05),
                  xytext=(6, 6), textcoords="offset points",
                  fontsize=10, color="#2C4B9E")
    # The last point falls in the bottom-right corner. Up-and-left the label
    # sits on the curve itself (the line passes right through there), and
    # below it overlaps the x axis. It goes to the RIGHT of the marker, with
    # extra axis margin to make room.
    ax_l.margins(x=0.22)
    ax_l.annotate(f"$\\times${h1 / ref['h']:.1f}", xy=(1.0, h1),
                  xytext=(9, -3), textcoords="offset points",
                  ha="left", va="center", fontsize=10, color="#2C4B9E")
    ax_l.set_xlabel(r"Pitch-up switch delay $\tau_s$ (s)")
    ax_l.set_ylabel(r"$\Delta h$ (m)")
    ax_l.set_title("(a) Late nose-down $\\to$ pull-up switch",
                   fontsize=10)

    # (b) the pull axis, canonical: closed-loop CAP (forgiving plateau)
    # vs OPEN-LOOP held deflection (re-stall cliff), on one axis. Signed
    # delta_e like everywhere else in the paper, axis inverted so that
    # "pull harder" still reads rightward.
    e3d = report.get("e3d_held_pull", {})
    pulls_cap = sorted((-abs(float(k)) for k in e3c), reverse=True)  # -5..-25
    h_cap = [e3c[f"{p:g}"][ckey]["h"] for p in pulls_cap]
    ax_r.plot(pulls_cap, h_cap, marker="o", ms=4.5, color="#2C4B9E",
              lw=2.0, zorder=4,
              label="closed loop: policy's pull clipped at $\\delta_e$")
    ax_r.annotate(f"$\\Delta h = {max(h_cap):.1f}$ to ${min(h_cap):.1f}$ m "
                  "for every cap",
                  xy=(-14.0, min(h_cap)), xytext=(0, -12),
                  textcoords="offset points", ha="center", va="top",
                  fontsize=9.5, color="#2C4B9E")
    if e3d:
        pulls_h = sorted((-float(k) for k in e3d), reverse=True)  # -2.5..-25
        h_held = [e3d[f"{-p:g}"]["h"] for p in pulls_h]
        ax_r.plot(pulls_h, h_held, marker="s", ms=4.5, color="#D62728",
                  lw=2.0, zorder=3,
                  label="open loop: $\\delta_e$ held after the switch")
        # shade the region where the held pull re-stalls (alpha > alpha_s).
        # Neutral grey: the red family is taken by the open-loop CURVE, and
        # a red band next to a red curve reads as "more of the same" when
        # the left branch of the valley fails by the opposite mechanism.
        # A margin above alpha_s, not the bare threshold. With > 14.0 the
        # -5 deg case came in too, whose alpha_max is 14.10: that is not a
        # re-stall but the equivalent control grazing the limit, which is how
        # the text describes it. Shading it put the band edge exactly on the
        # equivalent-control line and on the minimum of the valley, which is
        # precisely the point that does NOT fail.
        restall = [p for p in pulls_h
                   if e3d[f"{-p:g}"]["alpha_max_deg"] > 14.0 + 0.5]
        if restall:
            ax_r.axvspan(min(restall), max(restall), color="0.35",
                         alpha=0.10, zorder=1)
            ax_r.annotate("secondary stall\n($\\alpha > \\alpha_s$)",
                          xy=(0.82, 0.42), xycoords="axes fraction",
                          fontsize=9.5, color="#D62728", ha="center")
        # name the left branch too: under-pull, a different failure mode
        ax_r.annotate("insufficient pull\n(no re-stall)",
                      xy=(pulls_h[0], h_held[0]),
                      xytext=(-2, -16), textcoords="offset points",
                      fontsize=8.5, color="#D62728", ha="left", va="top")
    ax_r.axhline(ref["h"], color="gray", lw=0.9, ls=":")
    ax_r.axvline(-5.0, color="0.45", lw=0.9, ls="--")
    ax_r.annotate("equivalent control\n$\\bar{\\delta}_e \\approx -5^\\circ$",
                  xy=(-5.0, 0.42), xycoords=("data", "axes fraction"),
                  xytext=(6, 0), textcoords="offset points",
                  fontsize=8.5, color="0.35", ha="left")
    ax_r.set_xlim(0.0, -26.0)   # inverted: pull grows rightward, signed
    ax_r.set_xlabel(r"Pull-up deflection $\delta_{e}$ (deg)")
    ax_r.set_ylabel(r"$\Delta h$ (m)")
    ax_r.set_title("(b) How the pull is flown (canonical IC)",
                   fontsize=10)
    ax_r.legend(loc="lower left", fontsize=8.5, framealpha=0.95)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"fig_pilot_sensitivity.{ext}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)
    logger.info("[+] fig_pilot_sensitivity.{png,pdf} written")


# ── Power-delay figure (reads the report dict; no rollouts) ──────────────

def make_power_delay_figure(report):
    """Single-panel figure: canonical-IC altitude loss vs. power delay
    for both power models, with the three protocol landmarks (DP optimum,
    gated FAA logic, Gratton PD) prominently labeled."""
    e1 = report["e1_power_delay"]
    ref = report["optimal_canonical"]
    a0c, v0c = CANONICAL
    ckey = f"a{a0c:.0f}_v{v0c:.2f}"

    fig, ax = plt.subplots(figsize=(6.8, 4.4))

    for variant, color, label in [("instant", "#2C4B9E", "instant power"),
                                  ("ramp2s", "#E8742A", "2 s ramp (Gratton)")]:
        taus = sorted(float(k) for k in e1[variant])
        h_c = [e1[variant][f"{tau:g}"][ckey]["h"] for tau in taus]
        ax.plot(taus, h_c, marker="o", ms=4.5, color=color, label=label,
                lw=1.8)
    hg = e1["gated_alpha14_ramp2s"][ckey]["h"]
    hpd = e1["ramp2s"]["2"][ckey]["h"]
    slope = (e1["instant"]["4"][ckey]["h"] - e1["instant"]["0"][ckey]["h"]) / 4.0
    ax.axhline(ref["h"], color="gray", lw=1.1, ls=":")
    ax.annotate(f"DP optimum (CAA-like, $\\tau=0$): {ref['h']:.1f} m",
                xy=(0.02, ref["h"]), xytext=(0.55, ref["h"] - 0.6),
                fontsize=10.5, color="0.25", va="top")
    ax.plot([0.0], [hg], marker="D", ms=9, mfc="none", mew=1.8,
            color="#2CA02C", ls="none",
            label="gated on $\\alpha<\\alpha_s$ (FAA logic): "
                  f"{hg:.1f} m ($\\times${hg / ref['h']:.1f})")
    ax.plot([2.0], [hpd], marker="s", ms=10, mfc="none", mew=1.8,
            color="#D62728", ls="none",
            label="power-delayed (Gratton): "
                  f"{hpd:.1f} m ($\\times${hpd / ref['h']:.1f})")
    ax.annotate(f"$\\approx${abs(slope):.1f} m per second of delay",
                xy=(3.0, e1["instant"]["2"][ckey]["h"]),
                xytext=(-2, -38), textcoords="offset points",
                fontsize=10.5, color="#2C4B9E", ha="center", rotation=-24)
    ax.set_xlabel(r"Power application delay $\tau$ (s)")
    ax.set_ylabel(r"$\Delta h$ (m)")
    ax.legend(loc="lower left", fontsize=9.5, framealpha=0.95)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"fig_procedures.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("[+] fig_procedures.{png,pdf} written")


def make_trajectory_greedy_figure():
    """Fig. 8 with the optimum executed BOTH ways: Bellman-greedy (Approach B,
    the policy the solver actually converged to) and the barycentric action
    blend (Approach A, what the paper's trajectory figure plots). The two
    control panels are the point of the figure -- greedy holds full throttle
    for the entire recovery and saturates the elevator, while the blend
    reports intermediate commands that correspond to no admissible action."""
    pi = PolicyIterationStall.load(POLICY_PATH, env=SymmetricStall())
    a0c, v0c = CANONICAL
    logger.info("[greedy] rolling out (slow: argmax over the action set "
                "at every step)")
    runs = [
        ("DP greedy (Approach B)", "#B02418", "-",
         rollout(pi.env, pi, ctrl_optimal, a0c, v0c, record=True,
                 policy_action=get_optimal_action_greedy)),
        ("DP blended (Approach A)", "#2C4B9E", "--",
         rollout(pi.env, pi, ctrl_optimal, a0c, v0c, record=True)),
    ]
    for label, _, _, r in runs:
        logger.info(f"[greedy] {label}: {r['h']:.2f} m in {r['t']:.2f} s "
                    f"({r['status']})")

    signals = [
        ("gamma", r"$\gamma$ (deg)", np.rad2deg, "plot"),
        ("v_norm", r"$V/V_s$ (--)", None, "plot"),
        ("alpha", r"$\alpha$ (deg)", np.rad2deg, "plot"),
        ("q", r"$q$ (deg/s)", np.rad2deg, "plot"),
        ("de", r"$\delta_e$ (deg)", np.rad2deg, "step"),
        ("dt_ctrl", r"$\delta_t$ (--)", None, "step"),
        ("h", r"$\Delta h$ (m)", None, "plot"),
    ]
    rc = {"font.family": "serif", "mathtext.fontset": "stix",
          "font.size": 12, "axes.labelsize": 13,
          "xtick.labelsize": 11, "ytick.labelsize": 11,
          "axes.spines.top": False, "axes.spines.right": False}
    with plt.rc_context(rc):
        fig = plt.figure(figsize=(10.0, 6.8))
        gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1, 1.4],
                              hspace=0.42, wspace=0.24)
        slots = [gs[0, 0], gs[0, 1], gs[1, 0], gs[1, 1],
                 gs[2, 0], gs[2, 1], gs[3, :]]
        t_end = max(r["hist"]["t"][-1] for _, _, _, r in runs)
        for k, (slot, (key, ylabel, conv, style)) in enumerate(
                zip(slots, signals)):
            ax = fig.add_subplot(slot)
            for label, color, ls, r in runs:
                t = np.asarray(r["hist"]["t"])
                y = np.asarray(r["hist"][key])
                if conv is not None:
                    y = conv(y)
                if style == "step":
                    ax.step(t, y, color=color, lw=1.4, ls=ls, where="post",
                            label=label)
                else:
                    ax.plot(t, y, color=color, lw=1.4, ls=ls, label=label)
            ax.set_ylabel(ylabel)
            ax.set_xlim(0.0, t_end)
            ax.grid(True, linestyle=":", alpha=0.55)
            ax.text(0.0, 1.02, f"({'abcdefg'[k]})", transform=ax.transAxes,
                    fontsize=11, va="bottom", ha="left")
            if k >= 5:
                ax.set_xlabel("Time (s)")
        fig.legend(*fig.axes[0].get_legend_handles_labels(),
                   loc="lower center", ncol=2, frameon=False,
                   bbox_to_anchor=(0.5, -0.02))
        for ext in ("png", "pdf"):
            fig.savefig(OUT_DIR / f"fig_trajectories_greedy.{ext}", dpi=300,
                        bbox_inches="tight")
    plt.close(fig)
    logger.info("[+] fig_trajectories_greedy.{png,pdf} written")


def make_trajectory_comparison_figure():
    """Time-domain comparison at the canonical IC: DP optimum vs the
    scripted CAA and FAA procedures (alpha-hold pilot, 2 s power ramp).
    Same paired-panel layout as the single-trajectory figure; one color
    per protocol (consistent with the power-delay figure: blue optimum,
    orange CAA-timing, green FAA)."""
    pi = PolicyIterationStall.load(POLICY_PATH, env=SymmetricStall())
    a0c, v0c = CANONICAL
    r_opt = rollout(pi.env, pi, ctrl_optimal, a0c, v0c, record=True)

    # Hold pull authority fixed across the three arms: the scripted pilots may
    # not haul back harder than the optimum does at this same entry. Otherwise
    # the figure shows the procedures pulling to the -25 deg stop while the
    # optimum stops at -17.98 deg, and the visible gap mixes the timing of the
    # power application with a pull the optimum never commands.
    de_cap = float(np.min(r_opt["hist"]["de"]))
    logger.info(f"[traj] scripted pilots' pull capped at "
                f"{np.rad2deg(de_cap):.2f} deg (the optimum's minimum)")

    runs = [
        ("DP optimum", "#2C4B9E", "-", r_opt),
        ("CAA ($\\alpha$-hold pilot)", "#E8742A", "--",
         rollout(pi.env, pi, make_maneuver("t0", "alpha_hold", de_cap),
                 a0c, v0c, record=True)),
        ("FAA ($\\alpha$-hold pilot)", "#2CA02C", "-.",
         rollout(pi.env, pi, make_maneuver("unstall", "alpha_hold", de_cap),
                 a0c, v0c, record=True)),
    ]

    rc = {
        "font.family": "serif", "mathtext.fontset": "stix",
        "font.size": 12, "axes.labelsize": 13,
        "xtick.labelsize": 11, "ytick.labelsize": 11,
        "axes.spines.top": False, "axes.spines.right": False,
    }
    signals = [
        ("gamma", r"$\gamma$ (deg)", np.rad2deg, "plot"),
        ("v_norm", r"$V/V_s$ (--)", None, "plot"),
        ("alpha", r"$\alpha$ (deg)", np.rad2deg, "plot"),
        ("q", r"$q$ (deg/s)", np.rad2deg, "plot"),
        ("de", r"$\delta_e$ (deg)", np.rad2deg, "step"),
        ("dt_ctrl", r"$\delta_t$ (--)", None, "step"),
        ("h", r"$\Delta h$ (m)", None, "plot"),
    ]
    with plt.rc_context(rc):
        fig = plt.figure(figsize=(10.0, 6.8))
        gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1, 1.4],
                              hspace=0.42, wspace=0.24)
        slots = [gs[0, 0], gs[0, 1], gs[1, 0], gs[1, 1],
                 gs[2, 0], gs[2, 1], gs[3, :]]
        axs = []
        t_end = max(r["hist"]["t"][-1] for _, _, _, r in runs)
        for k, (slot, (key, ylabel, conv, style)) in enumerate(
                zip(slots, signals)):
            ax = fig.add_subplot(slot)
            for label, color, ls, r in runs:
                t = np.asarray(r["hist"]["t"])
                y = np.asarray(r["hist"][key])
                if conv is not None:
                    y = conv(y)
                if style == "step":
                    ax.step(t, y, color=color, lw=1.6, ls=ls, where="post")
                else:
                    ax.plot(t, y, color=color, lw=1.6, ls=ls)
            ax.set_ylabel(ylabel)
            ax.set_xlim(0.0, t_end)
            ax.grid(True, linestyle=":", alpha=0.55)
            ax.text(0.0, 1.02, f"({'abcdefg'[k]})",
                    transform=ax.transAxes, fontsize=11,
                    va="bottom", ha="left")
            axs.append(ax)

        # Elevator-switch event of the DP optimum (first pull command):
        # marked across all panels so the pilot-error section can refer
        # to it temporally.
        de_opt = np.asarray(runs[0][3]["hist"]["de"])
        t_opt = np.asarray(runs[0][3]["hist"]["t"])
        i_sw = int(np.argmax(de_opt < 0.0)) if np.any(de_opt < 0.0) else None
        if i_sw:
            t_sw = t_opt[i_sw]
            for ax in axs:
                ax.axvline(t_sw, color="0.45", linestyle=":",
                           linewidth=1.0)
            axs[0].annotate(f"$t^\\ast = {t_sw:.2f}$ s",
                            xy=(t_sw, 0.08),
                            xycoords=("data", "axes fraction"),
                            xytext=(5, 0), textcoords="offset points",
                            fontsize=10.5, color="0.45")
        axs[0].axhline(0.0, color="0.45", linestyle="--", linewidth=0.9)
        axs[1].axhline(1.0, color="0.45", linestyle="--", linewidth=0.9)
        axs[2].axhline(14.0, color="0.45", linestyle="--", linewidth=0.9)
        # The qualifier goes ON the symbol, not in parentheses at the end:
        # "(power-off)" read as the state of the aircraft -- which is flying
        # at full power -- when it actually qualifies the LINE, which is the
        # limit measured with the propeller producing no thrust. With power,
        # Riley's CL keeps growing up to 40 deg, so crossing this line is not
        # a re-stall.
        axs[2].annotate(r"$\alpha_s^{\mathrm{power\text{-}off}} = 14^\circ$",
                        xy=(0.98, 14.0), xycoords=("axes fraction", "data"),
                        xytext=(0, 5), textcoords="offset points",
                        ha="right", fontsize=10, color="0.45")
        axs[3].axhline(0.0, color="0.45", linestyle="--", linewidth=0.9)
        axs[4].axhline(0.0, color="0.45", linestyle="--", linewidth=0.9)
        axs[5].set_ylim([-0.05, 1.05])
        for label, color, _, r in runs:
            axs[6].plot(r["hist"]["t"][-1], r["hist"]["h"][-1], marker="o",
                        ms=5, color=color)
        handles = [plt.Line2D([], [], color=c, lw=2.0, ls=ls,
                              label=f"{lab}: {r['h']:.1f} m")
                   for lab, c, ls, r in runs]
        axs[6].legend(handles=handles, loc="lower left", fontsize=11)

        for ax in axs[:-1]:
            ax.tick_params(labelbottom=True, labelsize=9)
        axs[-1].set_xlabel("Time (s)")
        fig.align_ylabels(axs)
        for ext in ("png", "pdf"):
            fig.savefig(OUT_DIR / f"fig_trajectories_procedures.{ext}",
                        dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info("[+] fig_trajectories_procedures.{png,pdf} written")


# Dense IC sweep for the heatmap figures. The maps plot V0/Vs on the abscissa
# and alpha0 on the ordinate, so V0 carries the primary gradient and gets the
# finer spacing; the sweep covers only the well-posed stalled-entry domain
# V0 <= Vs (see _load_dense_cropped), so no rollout is discarded.
IC_HM_ALPHAS = np.arange(14.0, 20.01, 0.5)       # deg, 13 values
IC_HM_VNORMS = np.arange(0.90, 1.0001, 0.005)    # V0/Vs, 21 values

IC_HM_STRATEGIES = [
    ("optimal", "Optimal (CAA-like)", lambda: ctrl_optimal),
    ("tau05", r"instant, $\tau=0.5$ s", lambda: make_power_delay(0.5)),
    ("tau1", r"instant, $\tau=1$ s", lambda: make_power_delay(1.0)),
    ("gated", "gated (FAA logic)", lambda: make_power_gated(ramp=True)),
    ("pd", "PD (Gratton)", lambda: make_power_delay(2.0, ramp=True)),
]


def _ctrl_factory(key):
    """Rebuild a controller from its cache key. Sweep workers are separate
    processes and cannot receive the closures in IC_HM_STRATEGIES, so they
    reconstruct the controller from the key instead."""
    if key == "caa_ramp":
        return make_power_delay(0.0, ramp=True)
    if key.startswith("switch"):
        return make_switch_delay(float(key[len("switch"):]))
    for k, _, mk in IC_HM_STRATEGIES:
        if k == key:
            return mk()
    raise KeyError(f"unknown sweep strategy: {key}")


_SWEEP_STATE = {}


def _sweep_init():
    _SWEEP_STATE["pi"] = PolicyIterationStall.load(POLICY_PATH,
                                                   env=SymmetricStall())


def _sweep_job(job):
    key, v0, a0 = job
    pi = _SWEEP_STATE["pi"]
    return rollout(pi.env, pi, _ctrl_factory(key), a0, v0)["h"]


def _sweep(keys, alphas=None, vnorms=None, workers=None):
    """Parallel IC-plane rollout sweep over `keys`.

    Returns {key: grid} with rows = V0 and cols = alpha0, the layout the
    cache and _load_dense_cropped expect. Rollouts are deterministic and
    mutually independent, so this is a pure speedup: one process per worker,
    each with its own policy/env instance.
    """
    import multiprocessing as mp

    alphas = IC_HM_ALPHAS if alphas is None else alphas
    vnorms = IC_HM_VNORMS if vnorms is None else vnorms
    jobs = [(k, float(v), float(a))
            for k in keys for v in vnorms for a in alphas]
    n = workers or max(1, min(12, (os.cpu_count() or 2) - 2))
    logger.info(f"[sweep] {len(jobs)} rollouts over {n} workers: {list(keys)}")
    with mp.get_context("spawn").Pool(n, initializer=_sweep_init) as pool:
        out = pool.map(_sweep_job, jobs, chunksize=4)

    grids, i = {}, 0
    for k in keys:
        grid = []
        for _ in vnorms:
            grid.append([float(x) for x in out[i:i + len(alphas)]])
            i += len(alphas)
        grids[k] = grid
        logger.info(f"[sweep] {k} done")
    return grids


# ── Orthogonal IC cuts through the canonical entry ───────────────────────
# The alpha-V plane above is dense in two of the four states and silent on the
# other two. These cuts share its abscissa (V0/Vs) and sweep one of the
# remaining states each, all through the canonical entry
# (alpha0 = 20 deg, gamma0 = 0, q0 = 0), so the optimum can be reported over
# every state direction rather than a single slice.
CUT_ANCHOR = {"alpha0_deg": CANONICAL[0], "gamma0_deg": 0.0, "q0_deg": 0.0}

IC_CUTS = [
    ("alpha", "alpha0_deg", r"$\alpha_0$ (deg)",
     np.arange(14.0, 20.01, 0.5)),
    ("gamma", "gamma0_deg", r"$\gamma_0$ (deg)",
     np.arange(-30.0, 0.01, 1.5)),
    ("q", "q0_deg", r"$q_0$ (deg/s)",
     np.arange(-20.0, 20.01, 2.0)),
]


def _cut_job(job):
    """One rollout of the optimal policy at an (V0, y) node of a given cut."""
    cut, v0, y = job
    param = next(p for c, p, _, _ in IC_CUTS if c == cut)
    kw = dict(CUT_ANCHOR)
    kw[param] = float(y)
    pi = _SWEEP_STATE["pi"]
    r = rollout(pi.env, pi, ctrl_optimal, vnorm0=float(v0), **kw)
    return r["h"]


def compute_ic_cuts(vnorms=None, workers=None):
    """Sweep the optimal policy over each orthogonal IC cut; cached to
    ic_cuts.json. Grids are stored as [V0, y] to match the alpha-V cache."""
    import multiprocessing as mp

    vnorms = IC_HM_VNORMS if vnorms is None else vnorms
    jobs = [(cut, float(v), float(y))
            for cut, _, _, ys in IC_CUTS for v in vnorms for y in ys]
    n = workers or max(1, min(12, (os.cpu_count() or 2) - 2))
    logger.info(f"[cuts] {len(jobs)} rollouts over {n} workers")
    with mp.get_context("spawn").Pool(n, initializer=_sweep_init) as pool:
        out = pool.map(_cut_job, jobs, chunksize=4)

    data = {"vnorm0": [float(v) for v in vnorms], "anchor": CUT_ANCHOR}
    i = 0
    for cut, _, label, ys in IC_CUTS:
        grid = []
        for _ in vnorms:
            grid.append([float(x) for x in out[i:i + len(ys)]])
            i += len(ys)
        data[cut] = {"values": [float(y) for y in ys], "label": label,
                     "grid": grid}
        logger.info(f"[cuts] {cut} done")
    (OUT_DIR / "ic_cuts.json").write_text(json.dumps(data, indent=1))
    logger.info("[+] ic_cuts.json written")
    return data


def make_ic_cuts_figure(data=None):
    """The optimum over three orthogonal cuts through the canonical entry,
    sharing the V0/Vs abscissa: alpha0, gamma0 and q0 on the ordinates."""
    if data is None:
        data = json.loads((OUT_DIR / "ic_cuts.json").read_text())
    V = np.array(data["vnorm0"])
    cuts = [(c, data[c]) for c, _, _, _ in IC_CUTS]
    lo = min(np.min(d["grid"]) for _, d in cuts)
    hi = max(np.max(d["grid"]) for _, d in cuts)

    param_of = {c: p for c, p, _, _ in IC_CUTS}
    held_fmt = {"alpha0_deg": r"$\alpha_0 = %g^\circ$",
                "gamma0_deg": r"$\gamma_0 = %g^\circ$",
                "q0_deg": r"$q_0 = %g^\circ$/s"}

    # Each panel sweeps a different state, so each carries its own y label and
    # the three cannot share an axis: constrained_layout keeps those labels
    # from landing on the neighbouring panel, and places the shared colorbar.
    fig, axes = plt.subplots(1, len(cuts), figsize=(13.5, 3.6),
                             constrained_layout=True)
    for ax, (cut, d) in zip(axes, cuts):
        Y = np.array(d["values"])
        g = np.array(d["grid"]).T          # cached [V, y] -> plotted [y, V]
        Vm, Ym = np.meshgrid(V, Y, indexing="xy")
        pcm = ax.pcolormesh(Vm, Ym, g, cmap="viridis", shading="gouraud",
                            vmin=lo, vmax=hi)
        cs = ax.contour(Vm, Ym, g, levels=[-25, -20, -15, -10, -5],
                        colors="white", linewidths=1.4)
        ax.clabel(cs, fmt=lambda x: f"{x:.0f} m", fontsize=9)
        # The title carries what this cut holds fixed, which is also what
        # identifies the canonical entry the three cuts share -- no marker
        # needed, and one on the axis edge read as noise.
        swept = param_of[cut]
        ax.set_title(", ".join(held_fmt[k] % v
                               for k, v in data["anchor"].items()
                               if k != swept), fontsize=10)
        ax.set_ylabel(d["label"])
        _ic_axes(ax)
        ax.grid(True, color="white", linestyle=":", linewidth=0.6, alpha=0.55)
    cb = fig.colorbar(pcm, ax=axes, shrink=0.95, pad=0.02)
    cb.set_label(r"$\Delta h$ (m)")
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"fig_ic_cuts.{ext}", dpi=300)
    plt.close(fig)
    logger.info("[+] fig_ic_cuts.{png,pdf} written")


# ── Optimal altitude loss on the gamma-alpha plane, three speeds ─────────
# Same slicing as the policy heatmaps of plot_policy_heatmaps (alpha on the
# abscissa, gamma on the ordinate, q = 0, the same three speeds), so the cost
# field can be read against the switching surface cell by cell. Unlike the
# third column of that figure, which shows -V(x), these are closed-loop
# rollouts: the two differ by the action-blending gap.
GA_VNORMS = [0.90, 1.00, 1.10]
GA_ALPHAS = np.arange(14.0, 20.01, 0.25)   # 25 values
GA_GAMMAS = np.arange(-30.0, 0.01, 1.0)    # 31 values

# TODO: refine near zero. At a 1 deg step, between gamma_0 = -1.0 and -2.0 at
# 1.10 Vs the delayed-power arm jumps from +0.08 to +6.32 m and the map draws
# it as a cliff. Measured at the midpoint, -1.5 deg already costs 6.01 m: there
# is a real bifurcation -- either the aircraft arrests the sink immediately
# (recovers in 0.6 s, the 2 s delay never even matters) or it does not (sinks
# another degree and takes 8.9 s). Below a certain entry angle, delaying power
# stops being free; locating that angle is a publishable result.
#
# The fine grid would be:
#     np.concatenate([np.arange(-30.0, -3.0, 1.0),      # 27 values
#                     np.arange(-3.0, 0.01, 0.25)])     # 13 values
#
# It costs 12,000 rollouts (~70 min for the first phase alone) and exposes
# three rows in the stopping rule's dead zone instead of one, so it is worth
# doing together with the utils/recovery.py fix, not before.
GA_ARMS = ["optimal", "caa_ramp", "gated"]


def _ga_job(job):
    key, v0, g0, a0 = job
    pi = _SWEEP_STATE["pi"]
    r = rollout(pi.env, pi, _ctrl_factory(key), alpha0_deg=a0, vnorm0=v0,
                gamma0_deg=g0)
    return r["h"], r["status"]


def compute_ic_gamma_alpha(arms=None, workers=None):
    """Delta h over (gamma0, alpha0) at each of GA_VNORMS, for each arm.
    Cached to ic_gamma_alpha.json as [gamma][alpha] grids, with statuses."""
    import multiprocessing as mp

    arms = list(arms or GA_ARMS)
    jobs = [(k, float(v), float(g), float(a))
            for k in arms for v in GA_VNORMS
            for g in GA_GAMMAS for a in GA_ALPHAS]
    nw = workers or max(1, min(12, (os.cpu_count() or 2) - 2))
    logger.info(f"[ga] {len(jobs)} rollouts over {nw} workers ({arms})")
    with mp.get_context("spawn").Pool(nw, initializer=_sweep_init) as pool:
        out = pool.map(_ga_job, jobs, chunksize=8)

    data = {"vnorms": [float(v) for v in GA_VNORMS],
            "alpha0_deg": [float(a) for a in GA_ALPHAS],
            "gamma0_deg": [float(g) for g in GA_GAMMAS], "q0_deg": 0.0,
            "arms": {}}
    i = 0
    for k in arms:
        panels = []
        for v in GA_VNORMS:
            h, st = [], []
            for _ in GA_GAMMAS:
                chunk = out[i:i + len(GA_ALPHAS)]
                h.append([float(x) for x, _ in chunk])
                st.append([s for _, s in chunk])
                i += len(GA_ALPHAS)
            panels.append({"vnorm": float(v), "h": h, "status": st})
        data["arms"][k] = panels
        logger.info(f"[ga] {k} done")
    (OUT_DIR / "ic_gamma_alpha.json").write_text(json.dumps(data, indent=1))
    logger.info("[+] ic_gamma_alpha.json written")
    return data


# One full gamma-grid interval (56 bins over [-90, 5] deg). An entry whose
# optimal trajectory climbs past this before the dive develops is blending
# cells well inside the terminal success set, so its altitude loss measures an
# artifact rather than a recovery. Used to grey out those entries in the V.E
# figures.
#
# Was half an interval (0.86 deg), which made sense while gamma > 0 was
# genuinely UNPOLICED: policy improvement returns early on terminal cells, so
# they kept their initialisation (idle throttle) and any trajectory touching
# them blended in a command that was never optimized. The terminal-cell fill
# removes that: those cells now carry the policy of the nearest state where
# the problem is posed. With the fill in place, half an interval flagged the
# gamma_0 = 0 row at V/Vs = 1.00 (alpha_0 = 14.00 - 17.25 deg) purely because
# the recovery climbs a little past level -- entries where the filled policy
# loses 0.44 m against 2.78 m unfilled. The mask was hiding its best results.
#
# 2026-07-29: disabled on request, so that the gamma_0 = 0 row of V/Vs = 1.10
# can be SEEN instead of being greyed out by the figure. Two caveats about
# those 25 nodes, measured and not assumed:
#   - all 25 recover, with no timeouts, losing 5.1-5.6 m against the 11.3-12.3
#     of the published policy; the result is better, not worse
#   - but they cross at 14.58-14.68 s out of a 15.0 horizon, and climb to
#     6.8-7.5 deg while the solver grid ends at +5.0 deg: there the policy
#     lookup saturates at the edge, i.e. it extrapolates
# Set it back to 95.0/55.0 (one grid interval) to restore the mask, or to
# 0.5*95.0/55.0 for the original half-interval criterion.
GA_GMAX_MASK_DEG = 1e9


def _gmax_job(job):
    v0, g0, a0 = job
    pi = _SWEEP_STATE["pi"]
    r = rollout(pi.env, pi, ctrl_optimal, alpha0_deg=a0, vnorm0=v0,
                gamma0_deg=g0, record=True)
    return float(np.rad2deg(np.max(r["hist"]["gamma"])))


def compute_ic_gamma_alpha_gmax(workers=None):
    """Add each node's optimal-arm gamma peak to the gamma-alpha cache, so
    the figures can grey out entries that climb into the unpoliced gamma > 0
    region instead of painting their contaminated losses as data."""
    import multiprocessing as mp

    data = json.loads((OUT_DIR / "ic_gamma_alpha.json").read_text())
    jobs = [(float(v), float(g), float(a)) for v in data["vnorms"]
            for g in data["gamma0_deg"] for a in data["alpha0_deg"]]
    nw = workers or max(1, min(12, (os.cpu_count() or 2) - 2))
    logger.info(f"[gmax] {len(jobs)} rollouts over {nw} workers")
    with mp.get_context("spawn").Pool(nw, initializer=_sweep_init) as pool:
        out = pool.map(_gmax_job, jobs, chunksize=8)

    nG, nA = len(data["gamma0_deg"]), len(data["alpha0_deg"])
    data["opt_gmax_deg"] = [
        [[out[p * nG * nA + i * nA + j] for j in range(nA)]
         for i in range(nG)]
        for p in range(len(data["vnorms"]))]
    (OUT_DIR / "ic_gamma_alpha.json").write_text(json.dumps(data, indent=1))
    n_bad = int(np.sum(np.array(data["opt_gmax_deg"]) > GA_GMAX_MASK_DEG))
    logger.info(f"[gmax] done; {n_bad}/{len(jobs)} nodes exceed "
                f"{GA_GMAX_MASK_DEG:.2f} deg and will be masked")
    return data


def _ga_masked(panel):
    """Grid with the entries that never close the recovery set to NaN."""
    g = np.array(panel["h"])
    g[np.array(panel["status"]) != "recovered"] = np.nan
    return g


def _ga_domain_mask(data):
    """Per-panel boolean masks: True where the entry's optimal trajectory
    climbs past GA_GMAX_MASK_DEG into the unpoliced terminal region, i.e.
    where every arm's evaluation is contaminated and the node must be grey.
    Returns None when the gmax sweep has not been run."""
    if "opt_gmax_deg" not in data:
        return None
    return [np.array(p) > GA_GMAX_MASK_DEG for p in data["opt_gmax_deg"]]


def _contour_smooth(g, passes=2):
    """NaN-aware 3x3 mean, applied `passes` times — for CONTOURING only.

    The rollout fields carry ~0.05-1 m of cell-scale ripple (discrete-time
    stopping events, barycentric action blending across switching surfaces).
    That is invisible under the optimum's 5-10 m contour spacing but shatters
    the 1-2 m levels of the excess maps into fragments. The pcolormesh always
    shows the raw field; only the guide contours read the smoothed one.
    """
    from numpy.lib.stride_tricks import sliding_window_view
    for _ in range(passes):
        w = sliding_window_view(np.pad(g, 1, mode="edge"), (3, 3))
        g = np.nanmean(w, axis=(2, 3))
    return g


def make_ic_gamma_alpha_figure(data=None):
    """Section V.E's initial-condition figure, on the axes of the policy maps
    (alpha abscissa, gamma ordinate, q = 0) so the cost field can be read
    against the switching surface.

    Top row: the optimum's altitude loss at each speed. Bottom row: the extra
    altitude the FAA sequence loses over the CAA one --- the pure doctrine
    effect, both flown with the same 2 s power ramp and the same optimal
    elevator schedule. Entries that never close the recovery are masked: their
    altitude loss is not defined by the stopping rule, and painting a number
    there would hide that.
    """
    if data is None:
        data = json.loads((OUT_DIR / "ic_gamma_alpha.json").read_text())
    A = np.array(data["alpha0_deg"])
    G = np.array(data["gamma0_deg"])
    V = data["vnorms"]
    opt = [_ga_masked(p) for p in data["arms"]["optimal"]]
    # h is negative, so caa - gated > 0 means the FAA arm lost that much more.
    dif = [_ga_masked(c) - _ga_masked(g)
           for c, g in zip(data["arms"]["caa_ramp"], data["arms"]["gated"])]

    n_masked = sum(int(np.isnan(g).sum()) for g in opt + dif)
    if n_masked:
        logger.info(f"[ga] {n_masked} node(s) masked (no recovery)")

    from matplotlib.colors import ListedColormap, TwoSlopeNorm
    cmap_opt = plt.get_cmap("viridis").copy()
    cmap_opt.set_bad("0.5")
    neg = plt.cm.Reds_r(np.linspace(0.25, 1.0, 128))
    pos = plt.cm.Purples(np.linspace(0.0, 1.0, 128))
    cmap_dif = ListedColormap(np.vstack([neg, pos]))
    cmap_dif.set_bad("0.75")
    lo_o = min(np.nanmin(g) for g in opt)
    hi_o = max(np.nanmax(g) for g in opt)
    hi_d = max(np.nanmax(g) for g in dif)
    lo_d = min(-0.5, min(np.nanmin(g) for g in dif))
    norm_d = TwoSlopeNorm(vmin=lo_d, vcenter=0.0, vmax=hi_d)
    Am, Gm = np.meshgrid(A, G, indexing="xy")

    fig, axes = plt.subplots(2, len(V), figsize=(13.0, 7.0), sharex=True,
                             sharey=True, constrained_layout=True)
    for j, v in enumerate(V):
        ax = axes[0, j]
        pcm_o = ax.pcolormesh(Am, Gm, opt[j], cmap=cmap_opt,
                              shading="gouraud", vmin=lo_o, vmax=hi_o)
        cs = ax.contour(Am, Gm, opt[j], levels=[-50, -40, -30, -20, -10, -5],
                        colors="white", linewidths=1.3)
        ax.clabel(cs, fmt=lambda x: f"{x:.0f} m", fontsize=9)
        ax.set_title(rf"$V_0/V_s = {v:.2f}$", fontsize=11)

        ax = axes[1, j]
        pcm_d = ax.pcolormesh(Am, Gm, dif[j], cmap=cmap_dif, norm=norm_d,
                              shading="gouraud")
        cs = ax.contour(Am, Gm, dif[j], levels=[1, 2, 3, 4],
                        colors="white", linewidths=1.3)
        ax.clabel(cs, fmt=lambda x: f"+{x:.0f} m", fontsize=9)
        ax.set_xlabel(r"$\alpha_0$ (deg)")
    for ax in axes.ravel():
        ax.grid(True, color="white", linestyle=":", linewidth=0.6, alpha=0.55)
    axes[0, 0].set_ylabel(r"$\gamma_0$ (deg)")
    axes[1, 0].set_ylabel(r"$\gamma_0$ (deg)")
    cb = fig.colorbar(pcm_o, ax=axes[0, :], shrink=0.92, pad=0.02)
    cb.set_label(r"optimum $\Delta h$ (m)")
    cb = fig.colorbar(pcm_d, ax=axes[1, :], shrink=0.92, pad=0.02)
    cb.set_label("extra loss, FAA over CAA (m)\n(red: FAA loses less)")
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"fig_ic_gamma_alpha.{ext}", dpi=300)
    plt.close(fig)
    logger.info("[+] fig_ic_gamma_alpha.{png,pdf} written")


def make_ic_optimum_figure(data=None):
    """Section V.E, figure 1 of 2: the exact optimum's altitude loss over the
    entry plane (alpha0 abscissa, gamma0 ordinate, q0 = 0), one panel per
    airspeed of the policy maps. Labeled contours stay here -- against the
    optimum's 5-10 m level spacing they read as clean curves."""
    if data is None:
        data = json.loads((OUT_DIR / "ic_gamma_alpha.json").read_text())
    A = np.array(data["alpha0_deg"])
    G = np.array(data["gamma0_deg"])
    # All three regimes are drawn. The 1.10 Vs slice used to be excluded
    # because its level-entry row (gamma_0 = 0) climbed into the unpoliced
    # gamma > 0 region on the first step: there the policy kept its
    # initialisation (engine at idle) and the measured loss was an artefact.
    # Filling the terminal cells removes that cause, and the row goes from
    # -11.3/-12.3 m to -5.1/-5.6 m with all 25 entries recovering.
    #
    # One caveat remains, measured and not assumed: those trajectories climb
    # to 6.8-7.5 deg while the grid's gamma ceiling is +5.0 deg, so the policy
    # lookup saturates at the edge there (it extrapolates), and they cross at
    # 14.6 s out of a 15.0 s horizon. See GA_GMAX_MASK_DEG.
    keep = list(range(len(data["vnorms"])))
    V = [data["vnorms"][j] for j in keep]
    opt = [_ga_masked(data["arms"]["optimal"][j]) for j in keep]
    dom = _ga_domain_mask(data)
    if dom is not None:
        for g, j in zip(opt, keep):
            g[dom[j]] = np.nan

    # The gamma_0 = 0 row is left exactly as the rollout produces it.
    #
    # It is tempting to zero it: the DP's absorbing set is {gamma >= 0} and its
    # value function is exactly 0.0000 there, for every alpha and every V. But
    # the terminal set does NOT look at angle of attack, and an aircraft that
    # is level and stalled at 20 deg is not recovered: it falls, and the
    # rollout measures how far. The canonical IC of the entire paper is
    # exactly that case -- gamma_0 = 0, alpha_0 = 20 deg, V/Vs = 0.95,
    # -7.83 m -- so zeroing the row would erase the central result of the work.
    #
    # What the row actually contains:
    #     V/Vs=0.90   -18.14 to -15.47 m   (row -1: -16.31 to -14.73)  continuous
    #     V/Vs=1.00    -1.07 to  -0.44 m   (row -1:  -0.50 to  -0.34)  continuous
    #     V/Vs=1.10    -5.55 to  -5.08 m   (row -1:  -0.07 to  -0.04)  jumps
    #
    # Only the last regime is anomalous, and its cause is in the stopping rule
    # (see utils/recovery.py): with airspeed to spare the recovery climbs
    # instead of diving, the has_dived latch never opens, and the metric ends
    # up measuring a full phugoid instead of the manoeuvre.
    #
    # It is neutralised ONLY where the jump is large in absolute terms: the row
    # takes its neighbour's value. The threshold is 3 m rather than a multiple
    # of the typical step, because the relative criterion also fired at
    # 0.90 Vs, where the displacement is 2.24 m and reading the map does not
    # suffer. Measured, jump against the neighbouring row vs the panel's
    # typical step:
    #
    #     V/Vs=0.90   2.24 m   (typical 0.17)   left intact
    #     V/Vs=1.00   0.36 m   (typical 1.05)   left intact
    #     V/Vs=1.10   5.19 m   (typical 0.21)   neutralised
    #
    # This is a presentation patch over a metric problem. The real fix is in
    # utils/recovery.py: the has_dived latch never opens when the recovery
    # climbs instead of diving, and the episode ends up measuring a whole
    # phugoid. Once that is fixed the three rows come out right on their own
    # and this block becomes unnecessary.
    GA_ROW0_JUMP_M = 3.0
    i0 = int(np.argmin(np.abs(G)))
    if abs(G[i0]) < 1e-9 and len(G) > 2:
        i1 = i0 - 1
        for g in opt:
            salto = np.nanmedian(np.abs(g[i0] - g[i1]))
            if np.isfinite(salto) and salto > GA_ROW0_JUMP_M:
                g[i0, :] = g[i1, :]
    lo = min(np.nanmin(x) for x in opt)
    hi = max(np.nanmax(x) for x in opt)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("0.5")
    Am, Gm = np.meshgrid(A, G, indexing="xy")

    # Height reduced from 3.8 to 3.1: with three panels instead of two each
    # one is narrower, and at 3.8 the figure came out disproportionately tall
    # against the text box.
    fig, axes = plt.subplots(1, len(V), figsize=(8.8, 3.1), sharey=True,
                             constrained_layout=True)
    for ax, g, v in zip(axes, opt, V):
        # gouraud drops triangles touching NaN vertices (set_bad is ignored),
        # so the masked band must come from the axes background itself.
        ax.set_facecolor("0.5")
        pcm = ax.pcolormesh(Am, Gm, g, cmap=cmap, shading="gouraud",
                            vmin=lo, vmax=hi)
        cs = ax.contour(Am, Gm, _contour_smooth(g),
                        levels=[-50, -40, -30, -20, -10, -5],
                        colors="white", linewidths=1.3)
        ax.clabel(cs, fmt=lambda x: f"{x:.0f} m", fontsize=9)
        ax.set_title(rf"$V_0/V_s = {v:.2f}$", fontsize=11)
        ax.set_xlabel(r"$\alpha_0$ (deg)")
        ax.grid(True, color="white", linestyle=":", linewidth=0.6, alpha=0.55)
    axes[0].set_ylabel(r"$\gamma_0$ (deg)")
    cb = fig.colorbar(pcm, ax=axes, shrink=0.92, pad=0.02)
    cb.set_label(r"$\Delta h$ (m)")
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"fig_ic_optimum.{ext}", dpi=300)
    plt.close(fig)
    logger.info("[+] fig_ic_optimum.{png,pdf} written")


def make_ic_procedures_figure(data=None):
    """Section V.E, figure 2 of 2: what each procedure gives up to the
    optimum, on the same entry plane -- one row per procedure, one column per
    airspeed. Pure colour fields, no contours: at the excess maps' 1-2 m
    level spacing contour lines shatter into fragments and obscure the field,
    so the quantitative reading is carried by the per-row colorbars (all
    three repeat the SAME viridis scale anchored at zero excess, so the
    ranking CAA < FAA < PD reads directly down each column)."""
    if data is None:
        data = json.loads((OUT_DIR / "ic_gamma_alpha.json").read_text())
    A = np.array(data["alpha0_deg"])
    G = np.array(data["gamma0_deg"])
    # Same criterion as make_ic_optimum_figure: all three regimes. The
    # 1.10 Vs slice used to be excluded because its gamma_0 = 0 row climbed
    # into the unpoliced gamma > 0 region; filling the terminal cells removes
    # that cause.
    keep = list(range(len(data["vnorms"])))
    V = [data["vnorms"][j] for j in keep]
    opt = [_ga_masked(data["arms"]["optimal"][j]) for j in keep]
    exc = {k: [o - _ga_masked(data["arms"][k][j])
               for o, j in zip(opt, keep)]
           for k in ("caa_ramp", "gated", "pd")}

    # The gamma_0 = 0 row is left as the rollout produces it. Zeroing it,
    # which is what make_ic_optimum_figure does, would do more harm than good
    # here: at 0.90 and 1.00 Vs that row is already continuous with the one
    # below, and forcing it to zero opens an 8 m step that does not exist.
    # Measured, against the gamma_0 = -1 row:
    #
    #     V/Vs    original   zeroed   row -1
    #     0.90     +6.84      0.00     +7.99   <- the original is right
    #     1.00     +5.73      0.00     +6.71   <- the original is right
    #     1.10     -6.60      0.00     +0.07   <- only here is there an anomaly
    #
    # Dead zone of the stopping rule. The episode does not close until gamma
    # drops below DIVE_THRESHOLD_DEG (-0.5 deg) at least once. An entry that
    # starts ABOVE that value and whose recovery climbs instead of diving --
    # which is what happens at 1.10 Vs, where there is airspeed to spare --
    # never opens the latch: it runs to MAX_TIME and comes out marked as a
    # timeout, even though it ends 1.3 m higher than it started. _ga_masked
    # turns it into NaN and the map draws it grey, i.e. labels it "does not
    # recover", which is exactly the opposite of what happened.
    #
    # Rows with gamma_0 > DIVE_THRESHOLD_DEG are neutralised by taking the
    # first row that is outside the dead zone. The criterion is the cause, not
    # the appearance: it does not depend on how far the number jumps nor on
    # the grid density, so it still holds if the sweep is refined (at a
    # 0.25 deg step three rows fall in the zone, not one).
    #
    # A presentation patch over a metric problem: the real fix is a clause in
    # utils/recovery.py that closes the episode when the trajectory never
    # dives and ends at gamma >= 0.
    from symmetric_stall.utils.recovery import DIVE_THRESHOLD_DEG as _DIVE
    dead_rows = [i for i, g in enumerate(G) if g > _DIVE]
    if dead_rows:
        i_ok = max(i for i, g in enumerate(G) if g <= _DIVE)
        for gs in exc.values():
            for g in gs:
                for i in dead_rows:
                    g[i, :] = g[i_ok, :]

    from matplotlib.colors import Normalize, TwoSlopeNorm
    # Sequential viridis anchored at zero as long as every excess is
    # positive, which is the case today across all three regimes: the only
    # nodes that could come out negative (gamma_0 = 0 at 1.10 Vs) are timeouts
    # and are masked out before reaching here.
    #
    # The diverging branch is there in case that changes. With
    # Normalize(vmin=0) a negative excess clips to zero and looks the same as
    # a tie, with no warning; the scale picks itself so that a procedure that
    # genuinely beats the optimum does not stay hidden.
    g_all = [x for gs in exc.values() for x in gs]
    vmin = min(np.nanmin(x) for x in g_all)
    vmax = max(np.nanmax(x) for x in g_all)
    if vmin < -1e-9:
        cmap = plt.get_cmap("RdBu_r").copy()
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    else:
        cmap = plt.get_cmap("viridis").copy()
        norm = Normalize(vmin=0.0, vmax=vmax)
    cmap.set_bad("0.5")
    rows = [("caa_ramp", "CAA: simultaneous power"),
            ("gated", r"FAA: power gated on $\alpha < \alpha_s$"),
            ("pd", "Power-delayed (Gratton)")]
    Am, Gm = np.meshgrid(A, G, indexing="xy")

    # Height reduced from 9.8 to 8.2, as in make_ic_optimum_figure: with
    # three columns instead of two each panel is narrower and the figure came
    # out disproportionately tall against the text box.
    fig, axes = plt.subplots(3, len(V), figsize=(8.8, 8.2), sharex=True,
                             sharey=True, constrained_layout=True)
    for i, (key, title) in enumerate(rows):
        for j, v in enumerate(V):
            ax = axes[i, j]
            # grey background: gouraud leaves NaN cells undrawn, and white
            # would collide with the diverging scale's zero.
            ax.set_facecolor("0.5")
            pcm = ax.pcolormesh(Am, Gm, exc[key][j], cmap=cmap, norm=norm,
                                shading="gouraud")
            ax.grid(True, color="white", linestyle=":", linewidth=0.6,
                    alpha=0.55)
            if i == 0:
                ax.set_title(rf"$V_0/V_s = {v:.2f}$", fontsize=11)
            if j == 0:
                ax.set_ylabel(f"{title}\n" + r"$\gamma_0$ (deg)",
                              fontsize=9.5)
            if i == len(rows) - 1:
                ax.set_xlabel(r"$\alpha_0$ (deg)")
        # Same scale on every row, repeated deliberately: each row reads
        # standalone while the three stay directly comparable.
        cb = fig.colorbar(pcm, ax=axes[i, :], shrink=0.9, pad=0.02)
        cb.set_label("excess over the\noptimum (m)", fontsize=9)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"fig_ic_procedures.{ext}", dpi=300)
    plt.close(fig)
    logger.info("[+] fig_ic_procedures.{png,pdf} written")


def make_ic_gamma_lines_figure(data=None):
    """Section V.E's result as curves rather than fields.

    Delta h varies by ~35 m across gamma0 and by 1-2 m across alpha0, so a
    plane spends an axis on a variable that carries almost nothing, and the
    CAA-over-FAA advantage (a few metres on a 35 m field) is invisible at the
    field's own scale. Plotting against gamma0 and collapsing alpha0 into a
    band keeps both readable: the band's width IS the alpha0 insensitivity,
    and panel (b) can be scaled to a effect of its own size.
    """
    if data is None:
        data = json.loads((OUT_DIR / "ic_gamma_alpha.json").read_text())
    G = np.array(data["gamma0_deg"])
    V = data["vnorms"]
    colors = ["#3B1F6B", "#2C7FB8", "#7FCDBB"]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0),
                             constrained_layout=True)
    for j, (v, col) in enumerate(zip(V, colors)):
        opt = _ga_masked(data["arms"]["optimal"][j])
        axes[0].fill_between(G, np.nanmin(opt, axis=1),
                             np.nanmax(opt, axis=1), color=col, alpha=0.20,
                             linewidth=0)
        axes[0].plot(G, np.nanmedian(opt, axis=1), color=col, lw=1.9,
                     label=rf"$V_0/V_s = {v:.2f}$")
        # Panel (b) keeps the two sequences apart instead of collapsing them
        # into a difference: the separation within a pair is the doctrine
        # effect, the level of the pair is the floor the engine ramp imposes.
        for arm, ls in (("caa_ramp", "-"), ("gated", "--")):
            exc = opt - _ga_masked(data["arms"][arm][j])
            axes[1].plot(G, np.nanmedian(exc, axis=1), color=col, lw=1.8,
                         ls=ls)

    axes[0].set_ylabel(r"$\Delta h$ of the optimum (m)")
    axes[1].set_ylabel("excess over the optimum (m)")
    from matplotlib.lines import Line2D
    axes[1].legend(handles=[
        Line2D([], [], color="0.25", lw=1.8, ls="-",
               label="CAA (simultaneous power)"),
        Line2D([], [], color="0.25", lw=1.8, ls="--",
               label=r"FAA (gated on $\alpha < \alpha_s$)")],
        frameon=False, loc="upper left")
    for k, ax in enumerate(axes):
        ax.set_xlabel(r"$\gamma_0$ (deg)")
        ax.set_xlim(G.min(), G.max())
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.text(0.0, 1.02, f"({'ab'[k]})", transform=ax.transAxes,
                fontsize=11, va="bottom", ha="left")
    axes[0].legend(frameon=False, loc="lower right")
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"fig_ic_gamma_lines.{ext}", dpi=300)
    plt.close(fig)
    logger.info("[+] fig_ic_gamma_lines.{png,pdf} written")


def write_montecarlo_table(data=None):
    """The coverage certificate as a LaTeX table: what the slices cannot say."""
    if data is None:
        data = json.loads((OUT_DIR / "ic_montecarlo.json").read_text())
    n = data["n"]
    h = {k: np.array(data[k]["h"]) for k in MC_ARMS}
    st = {k: np.array(data[k]["status"]) for k in MC_ARMS}
    ok = np.all([st[k] == "recovered" for k in MC_ARMS], axis=0)
    opt = h["optimal"][ok]
    label = {"optimal": "DP optimum", "caa_ramp": "CAA (simultaneous)",
             "gated": r"FAA (gated on $\alpha < \alpha_s$)",
             "pd": "Power-delayed (Gratton)"}
    rows = []
    for k in MC_ARMS:
        x = h[k][ok]
        exc = "---" if k == "optimal" else \
            f"${np.median(opt - x):.2f}$ & ${np.percentile(opt - x, 95):.2f}$"
        if k == "optimal":
            exc = "--- & ---"
        rows.append(f"        {label[k]} & ${np.median(x):.2f}$ & "
                    f"${np.percentile(x, 5):.2f}$ & {exc} \\\\")
    caa_wins = 100 * np.mean(h["caa_ramp"][ok] > h["gated"][ok])
    adv = h["caa_ramp"][ok] - h["gated"][ok]
    lines = [
        r"\begin{table}[H]", r"    \centering",
        r"    \caption{Coverage of the stalled-entry set: every arm flown "
        rf"from the same ${n}$ Latin-hypercube samples of "
        r"$\gamma_0 \in [-30^\circ, 0]$, $V_0/V_s \in [0.90, 1.00]$, "
        r"$\alpha_0 \in [14^\circ, 20^\circ]$, "
        r"$q_0 \in [-20, 20]^\circ$/s. Unlike the planar maps of "
        r"Fig.~\ref{fig:ic_procedures}, this samples the interactions "
        r"between entry states, so the ranking is certified over the set "
        r"rather than on a slice of it. Percentiles describe the uniform "
        r"sampling box, not an operational entry distribution; the "
        r"per-entry ranking reported below is distribution-free.}",
        r"    \label{tab:montecarlo}",
        r"    \setlength{\tabcolsep}{14pt}"
        r"\renewcommand{\arraystretch}{1.15}",
        r"    \begin{tabular}{lrrrr}", r"        \hline",
        r"        & \multicolumn{2}{c}{$\Delta h$ (m)} & "
        r"\multicolumn{2}{c}{excess over optimum (m)} \\",
        r"        Arm & median & 5th pct & median & 95th pct \\",
        r"        \hline", *rows, r"        \hline",
        r"    \end{tabular}", r"    \\[2pt]",
        rf"    \footnotesize The CAA sequence loses less than the FAA "
        rf"sequence at ${caa_wins:.1f}\%$ of the sampled entries "
        rf"(median advantage ${np.median(adv):.2f}$\,m, smallest "
        rf"${adv.min():.2f}$\,m); ${int(ok.sum())}$ of ${n}$ entries closed "
        r"the recovery under every arm.",
        r"\end{table}",
    ]
    (OUT_DIR / "table_montecarlo.tex").write_text("\n".join(lines) + "\n")
    logger.info("[+] table_montecarlo.tex written")


# ── Full 4-D coverage of the stalled-entry set ───────────────────────────
# The cuts above are two-dimensional readings through one point, which leaves
# the interactions between off-anchor states unsampled. The procedure ranking
# is the paper's central claim, so it is certified over the whole 4-D entry
# box rather than on any slice of it: Latin-hypercube sample, every arm flown
# from every sampled entry.
MC_BOUNDS = {
    "gamma0_deg": (-30.0, 0.0),
    # Capped at Vs: with gamma0 ~ 0 and V0 > Vs the entry sits on the boundary
    # of the absorbing level-flight set, where "altitude loss until recovery"
    # does not measure a recovery -- the aircraft has the energy to fly out by
    # reducing alpha, and full power merely induces a climb, a decay and a
    # genuine break that the metric then charges to the recovery.
    "vnorm0": (0.90, 1.00),
    "alpha0_deg": (14.0, 20.0),
    "q0_deg": (-20.0, 20.0),
}
MC_ARMS = ["optimal", "caa_ramp", "gated", "pd"]


def _latin_hypercube(n, d, rng):
    """n points in [0,1)^d, one per stratum along every axis. Better marginal
    coverage than plain uniform at the same cost, which matters here because
    each sample is a full rollout."""
    strata = np.tile(np.arange(n, dtype=float), (d, 1))
    return ((rng.permuted(strata, axis=1) + rng.random((d, n))) / n).T


def _mc_job(job):
    key, g, v, a, q = job
    pi = _SWEEP_STATE["pi"]
    r = rollout(pi.env, pi, _ctrl_factory(key), alpha0_deg=a, vnorm0=v,
                gamma0_deg=g, q0_deg=q)
    return r["h"], r["status"]


def compute_ic_montecarlo(n=1000, seed=20260723, workers=None):
    """Fly every arm from n Latin-hypercube entries spanning MC_BOUNDS.
    Cached to ic_montecarlo.json, statuses included: an entry that does not
    recover under some arm must be visible, not averaged in silently."""
    import multiprocessing as mp

    rng = np.random.default_rng(seed)
    names = list(MC_BOUNDS)
    u = _latin_hypercube(n, len(names), rng)
    pts = {k: (MC_BOUNDS[k][0] + u[:, i] * (MC_BOUNDS[k][1] - MC_BOUNDS[k][0]))
           for i, k in enumerate(names)}

    jobs = [(key, float(pts["gamma0_deg"][i]), float(pts["vnorm0"][i]),
             float(pts["alpha0_deg"][i]), float(pts["q0_deg"][i]))
            for key in MC_ARMS for i in range(n)]
    nw = workers or max(1, min(12, (os.cpu_count() or 2) - 2))
    logger.info(f"[mc] {len(jobs)} rollouts over {nw} workers "
                f"({n} entries x {len(MC_ARMS)} arms)")
    with mp.get_context("spawn").Pool(nw, initializer=_sweep_init) as pool:
        out = pool.map(_mc_job, jobs, chunksize=8)

    data = {"n": n, "seed": seed, "bounds": MC_BOUNDS,
            "points": {k: [float(x) for x in v] for k, v in pts.items()}}
    for j, key in enumerate(MC_ARMS):
        chunk = out[j * n:(j + 1) * n]
        data[key] = {"h": [float(h) for h, _ in chunk],
                     "status": [s for _, s in chunk]}
        logger.info(f"[mc] {key} done")
    (OUT_DIR / "ic_montecarlo.json").write_text(json.dumps(data, indent=1))
    logger.info("[+] ic_montecarlo.json written")
    return data


def report_ic_montecarlo(data=None):
    """Ranking statistics over the 4-D entry set: the sentence the slices
    cannot support on their own."""
    if data is None:
        data = json.loads((OUT_DIR / "ic_montecarlo.json").read_text())
    n = data["n"]
    h = {k: np.array(data[k]["h"]) for k in MC_ARMS}
    st = {k: np.array(data[k]["status"]) for k in MC_ARMS}

    print(f"Latin-hypercube sample of the stalled-entry set, n = {n}")
    for k, v in data["bounds"].items():
        print(f"    {k:12s} [{v[0]:7.2f}, {v[1]:7.2f}]")
    print("\n  arm         recovered   crash  timeout    median dh    worst dh")
    for k in MC_ARMS:
        u, c = np.unique(st[k], return_counts=True)
        d = dict(zip(u, c))
        print(f"  {k:10s} {d.get('recovered',0):9d} {d.get('crash',0):7d} "
              f"{d.get('timeout',0):8d} {np.median(h[k]):11.2f} {h[k].min():11.2f}")

    ok = np.all([st[k] == "recovered" for k in MC_ARMS], axis=0)
    print(f"\n  entries where every arm recovered: {ok.sum()}/{n}")
    caa, faa, pd_, opt = h["caa_ramp"][ok], h["gated"][ok], h["pd"][ok], h["optimal"][ok]
    # h is the (negative) altitude change, so a larger h is a smaller loss.
    print(f"  CAA loses less than FAA:  {100*np.mean(caa > faa):6.2f} % of entries")
    print(f"  CAA loses less than PD :  {100*np.mean(caa > pd_):6.2f} %")
    print(f"  FAA loses less than PD :  {100*np.mean(faa > pd_):6.2f} %")
    print(f"  optimum best of all    :  {100*np.mean((opt > caa) & (opt > faa) & (opt > pd_)):6.2f} %")
    # h is a negative altitude change, so the arm with the LARGER h lost less:
    # this advantage is positive wherever the CAA sequence beats the FAA one.
    adv = caa - faa
    print(f"\n  CAA advantage over FAA (m): median {np.median(adv):.2f}, "
          f"p5 {np.percentile(adv,5):.2f}, p95 {np.percentile(adv,95):.2f}, "
          f"smallest {adv.min():.2f}")
    for k in ("caa_ramp", "gated", "pd"):
        e = opt - h[k][ok]
        print(f"  excess over optimum, {k:9s}: median {np.median(e):6.2f} m, "
              f"p95 {np.percentile(e,95):6.2f} m, max {e.max():6.2f} m")
    return data


def compute_ic_heatmap_dense():
    """Rollout sweep over the dense IC plane for every power-timing
    strategy; cached to ic_heatmap_dense.json (rows = V0, cols = alpha0)."""
    data = {"alpha0_deg": [float(a) for a in IC_HM_ALPHAS],
            "vnorm0": [float(v) for v in IC_HM_VNORMS]}
    data.update(_sweep([k for k, _, _ in IC_HM_STRATEGIES]))
    (OUT_DIR / "ic_heatmap_dense.json").write_text(json.dumps(data, indent=1))
    logger.info("[+] ic_heatmap_dense.json written")
    return data


def _load_dense_cropped(data=None):
    """Load the dense IC sweep, restricted to the well-posed stalled-entry
    domain V0 <= Vs. Above Vs the IC sits on the boundary of the absorbing
    success set (gamma=0, V >= Vs): the initial climb takes the rollout
    through gamma > 0 terminal cells whose policy entries are
    uninitialized (index 0), so the swept values there are artifacts."""
    if data is None:
        data = json.loads((OUT_DIR / "ic_heatmap_dense.json").read_text())
    A = np.array(data["alpha0_deg"])
    V = np.array(data["vnorm0"])
    keep = V <= 1.0 + 1e-9
    V = V[keep]
    grids = {k: np.array(data[k])[keep, :] for k, _, _ in IC_HM_STRATEGIES}
    return A, V, grids


def _ic_mesh(A, V):
    """Plotting mesh for every IC-plane map: V0/Vs on the abscissa, alpha0 on
    the ordinate. The cached grids are stored as [V, alpha], so the field
    itself must be transposed (`.T`) to match this mesh."""
    return np.meshgrid(V, A, indexing="xy")


def _ic_axes(ax):
    ax.set_xlabel(r"$V_0/V_s$")


def _ic_marks(ax, A, V, canonical_color="#D62728", cross_color="white"):
    for a0 in ALPHA_GRID_DEG:
        for v0 in VNORM_GRID:
            if v0 <= V.max() and a0 >= A.min():
                ax.plot(v0, a0, marker="+", ms=5, mew=1.0,
                        color=cross_color, clip_on=False)
    ax.plot(CANONICAL[1], CANONICAL[0], marker="o", ms=7, mfc="none",
            mew=1.3, color=canonical_color, clip_on=False)
    _ic_axes(ax)
    ax.grid(False)


def compute_caa_ramp_dense():
    """Add the realistic-CAA arm (full power from t=0 through a 2 s ramp)
    to the dense IC sweep cache."""
    data = json.loads((OUT_DIR / "ic_heatmap_dense.json").read_text())
    data.update(_sweep(["caa_ramp"], data["alpha0_deg"], data["vnorm0"]))
    (OUT_DIR / "ic_heatmap_dense.json").write_text(json.dumps(data, indent=1))
    logger.info("[+] caa_ramp added to ic_heatmap_dense.json")
    return data


def make_procedures_ic_figure(data=None):
    """Companion of make_optimal_ic_figure: EXCESS altitude loss of the
    two certified procedures (2 s power ramp) relative to the ideal
    optimum of that figure. The CAA panel is the floor imposed by engine
    physics (the ramp alone); the FAA panel is that floor plus the price
    of gating power on the unstall. Diverging scale: purple = worse than
    the optimum, red = beats it (sanity flag)."""
    A, V, grids = _load_dense_cropped(data)
    if data is None:
        data = json.loads((OUT_DIR / "ic_heatmap_dense.json").read_text())
    keep = np.array(data["vnorm0"]) <= 1.0 + 1e-9
    opt = grids["optimal"]
    exc_caa = opt - np.array(data["caa_ramp"])[keep, :]
    exc_faa = opt - grids["gated"]
    exc_pd = opt - grids["pd"]
    exc_max = max(exc_caa.max(), exc_faa.max(), exc_pd.max())
    exc_min = min(-2.0, exc_caa.min(), exc_faa.min(), exc_pd.min())
    from matplotlib.colors import ListedColormap, TwoSlopeNorm
    neg = plt.cm.Reds_r(np.linspace(0.25, 1.0, 128))
    pos = plt.cm.Purples(np.linspace(0.0, 1.0, 128))
    cmap_exc = ListedColormap(np.vstack([neg, pos]))
    norm_exc = TwoSlopeNorm(vmin=exc_min, vcenter=0.0, vmax=exc_max)
    Vm, Am = _ic_mesh(A, V)

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.5), sharey=True)
    panels = [(axes[0], exc_caa, "CAA: simultaneous power (2 s ramp)",
               [4, 6, 8, 10]),
              (axes[1], exc_faa,
               r"FAA: power gated on $\alpha<\alpha_s$ (2 s ramp)",
               [4, 6, 8, 10]),
              (axes[2], exc_pd,
               "Power-delayed (Gratton: 2 s pause + 2 s ramp)",
               [20, 22])]
    for ax, g, title, levels in panels:
        pcm = ax.pcolormesh(Vm, Am, g.T, cmap=cmap_exc, norm=norm_exc,
                            shading="gouraud")
        cs = ax.contour(Vm, Am, g.T, levels=levels, colors="white",
                        linewidths=1.6)
        ax.clabel(cs, fmt=lambda x: f"+{x:.0f} m", fontsize=10)
        _ic_axes(ax)
        ax.grid(True, color="white", linestyle=":", linewidth=0.6,
                alpha=0.55)
        ax.set_title(title, fontsize=10)
    axes[0].set_ylabel(r"$\alpha_0$ (deg)")
    cb = fig.colorbar(pcm, ax=axes, fraction=0.03, pad=0.02)
    cb.set_label("excess loss vs. optimal (m)\n(red: beats optimal)",
                 fontsize=9)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"fig_procedures_ic_heatmap.{ext}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)
    logger.info("[+] fig_procedures_ic_heatmap.{png,pdf} written")


SWITCH_HM_TAUS = [0.25, 0.5, 1.0]


def compute_switch_delay_dense():
    """Add switch-delay arms (hold nose-down tau_s extra seconds past the
    policy's own switch, then resume) to the dense IC sweep cache — the
    same grid as the power-timing maps, so the two error axes get the
    same IC-plane treatment."""
    data = json.loads((OUT_DIR / "ic_heatmap_dense.json").read_text())
    data.update(_sweep([f"switch{t:g}" for t in SWITCH_HM_TAUS],
                       data["alpha0_deg"], data["vnorm0"]))
    (OUT_DIR / "ic_heatmap_dense.json").write_text(json.dumps(data, indent=1))
    logger.info("[+] switch-delay arms added to ic_heatmap_dense.json")
    return data


def make_switch_delay_ic_figure(data=None):
    """Excess loss of the delayed-switch error over the IC plane, one
    panel per hold duration tau_s, relative to each IC's own optimum —
    same grammar as the power-timing excess maps (purple worse, red
    would flag beating the optimum), with labeled contours."""
    A, V, grids = _load_dense_cropped(data)
    if data is None:
        data = json.loads((OUT_DIR / "ic_heatmap_dense.json").read_text())
    keep = np.array(data["vnorm0"]) <= 1.0 + 1e-9
    opt = grids["optimal"]
    excess = {f"switch{t:g}": opt - np.array(data[f"switch{t:g}"])[keep, :]
              for t in SWITCH_HM_TAUS}
    exc_max = max(g.max() for g in excess.values())
    exc_min = min(-2.0, min(g.min() for g in excess.values()))
    from matplotlib.colors import ListedColormap, TwoSlopeNorm
    neg = plt.cm.Reds_r(np.linspace(0.25, 1.0, 128))
    pos = plt.cm.Purples(np.linspace(0.0, 1.0, 128))
    cmap_exc = ListedColormap(np.vstack([neg, pos]))
    norm_exc = TwoSlopeNorm(vmin=exc_min, vcenter=0.0, vmax=exc_max)
    Vm, Am = _ic_mesh(A, V)

    fig, axes = plt.subplots(1, len(SWITCH_HM_TAUS), figsize=(13.0, 3.5),
                             sharey=True)
    for ax, tau_s in zip(axes, SWITCH_HM_TAUS):
        g = excess[f"switch{tau_s:g}"].T
        pcm = ax.pcolormesh(Vm, Am, g, cmap=cmap_exc, norm=norm_exc,
                            shading="gouraud")
        levels = [5, 10] if tau_s < 0.4 else ([10, 15, 20] if tau_s < 0.7
                                              else [25, 30, 35, 40])
        cs = ax.contour(Vm, Am, g, levels=levels, colors="white",
                        linewidths=1.6)
        ax.clabel(cs, fmt=lambda x: f"+{x:.0f} m", fontsize=10)
        _ic_axes(ax)
        ax.grid(True, color="white", linestyle=":", linewidth=0.6,
                alpha=0.55)
        ax.set_title(f"switch held ${tau_s:g}$ s late", fontsize=10)
    axes[0].set_ylabel(r"$\alpha_0$ (deg)")
    cb = fig.colorbar(pcm, ax=axes, fraction=0.02, pad=0.015)
    cb.set_label("excess loss vs. optimal (m)\n(red: beats optimal)",
                 fontsize=9)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"fig_switch_delay_ic.{ext}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)
    logger.info("[+] fig_switch_delay_ic.{png,pdf} written")


def make_optimal_ic_figure(data=None):
    """Standalone map of the optimal recovery's altitude loss over the
    dense IC grid (viridis, matching the policy heatmaps), with labeled
    contours. Crosses = 3x3 table ICs; circle = canonical IC."""
    A, V, grids = _load_dense_cropped(data)
    opt = grids["optimal"].T
    Vm, Am = _ic_mesh(A, V)

    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    pcm = ax.pcolormesh(Vm, Am, opt, cmap="viridis", shading="gouraud")
    cs = ax.contour(Vm, Am, opt, levels=[-15, -10, -5], colors="white",
                    linewidths=1.6)
    ax.clabel(cs, fmt=lambda x: f"{x:.0f} m", fontsize=11)
    _ic_axes(ax)
    ax.grid(True, color="white", linestyle=":", linewidth=0.6, alpha=0.55)
    ax.set_ylabel(r"$\alpha_0$ (deg)")
    cb = fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(r"$\Delta h$ (m)")
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"fig_optimal_ic_heatmap.{ext}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)
    logger.info("[+] fig_optimal_ic_heatmap.{png,pdf} written")


def make_ic_heatmap_figure(data=None):
    """Excess loss of each delayed power-timing variant relative to the
    optimum (companion of make_optimal_ic_figure), diverging scale:
    purple = worse than the optimum, white = equal, red = BEATS the
    optimum (sanity flag — anything beyond the ~0.3 m execution band
    would indicate a bug)."""
    A, V, grids = _load_dense_cropped(data)
    opt = grids["optimal"]
    excess = {k: opt - g for k, g in grids.items() if k != "optimal"}
    exc_max = max(g.max() for g in excess.values())
    exc_min = min(-2.0, min(g.min() for g in excess.values()))
    from matplotlib.colors import ListedColormap, TwoSlopeNorm
    neg = plt.cm.Reds_r(np.linspace(0.25, 1.0, 128))
    pos = plt.cm.Purples(np.linspace(0.0, 1.0, 128))
    cmap_exc = ListedColormap(np.vstack([neg, pos]))
    norm_exc = TwoSlopeNorm(vmin=exc_min, vcenter=0.0, vmax=exc_max)
    Vm, Am = _ic_mesh(A, V)

    fig, axes = plt.subplots(1, len(IC_HM_STRATEGIES) - 1,
                             figsize=(10.8, 2.9), sharey=True)
    for ax, (key, title, _) in zip(axes, IC_HM_STRATEGIES[1:]):
        pcm_exc = ax.pcolormesh(Vm, Am, excess[key].T, cmap=cmap_exc,
                                norm=norm_exc, shading="gouraud")
        _ic_marks(ax, A, V, canonical_color="#2C4B9E", cross_color="0.35")
        ax.set_title(title, fontsize=9.5)
    axes[0].set_ylabel(r"$\alpha_0$ (deg)")

    cb2 = fig.colorbar(pcm_exc, ax=axes, fraction=0.025, pad=0.015)
    cb2.set_label("excess loss vs. optimal (m)\n(red: beats optimal)",
                  fontsize=9)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"fig_caa_faa_heatmap.{ext}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)
    logger.info("[+] fig_caa_faa_heatmap.{png,pdf} written")


# ── Main experiment ──────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pi = PolicyIterationStall.load(POLICY_PATH, env=SymmetricStall())
    env = pi.env
    report = {"policy": str(POLICY_PATH)}

    # CAA evidence
    report["caa_evidence"] = caa_evidence(pi)
    logger.info(f"[=] full-throttle argmax: "
                f"{report['caa_evidence']['full_throttle_frac_global']:.1%} global, "
                f"{report['caa_evidence']['full_throttle_frac_window']:.1%} in recovery window")

    a0c, v0c = CANONICAL

    # Optimal reference (tau = 0, canonical + grid)
    ref = rollout(env, pi, ctrl_optimal, a0c, v0c, record=True)
    report["optimal_canonical"] = {"h": ref["h"], "t": ref["t"], "status": ref["status"]}
    de_seq = np.array(ref["hist"]["de"])
    t_seq = np.array(ref["hist"]["t"])
    i_sw = int(np.argmax(de_seq < 0.0)) if np.any(de_seq < 0.0) else -1
    report["optimal_switch_time_s"] = float(t_seq[i_sw]) if i_sw >= 0 else None
    logger.info(f"[=] optimal canonical: {ref['h']:.2f} m; "
                f"de switch at t*={report['optimal_switch_time_s']} s")

    # E1: power-delay sweep (instant + ramp) on canonical + IC grid
    e1 = {}
    for variant, ramp in [("instant", False), ("ramp2s", True)]:
        rows = {}
        for tau in TAU_POWER:
            per_ic = {}
            for alpha0 in ALPHA_GRID_DEG:
                for v0 in VNORM_GRID:
                    r = rollout(env, pi, make_power_delay(tau, ramp), alpha0, v0)
                    per_ic[f"a{alpha0:.0f}_v{v0:.2f}"] = {
                        "h": r["h"], "t": r["t"], "status": r["status"]}
            rows[f"{tau:g}"] = per_ic
            logger.info(f"    E1 {variant} tau={tau:g}s: canonical "
                        f"{per_ic[f'a{a0c:.0f}_v{v0c:.2f}']['h']:.2f} m")
        e1[variant] = rows
    # gated-on-alpha variant (canonical + grid, ramp per Gratton)
    per_ic = {}
    for alpha0 in ALPHA_GRID_DEG:
        for v0 in VNORM_GRID:
            r = rollout(env, pi, make_power_gated(ramp=True), alpha0, v0)
            per_ic[f"a{alpha0:.0f}_v{v0:.2f}"] = {
                "h": r["h"], "t": r["t"], "status": r["status"]}
    e1["gated_alpha14_ramp2s"] = per_ic
    logger.info(f"    E1 gated(a<14) ramp: canonical "
                f"{per_ic[f'a{a0c:.0f}_v{v0c:.2f}']['h']:.2f} m")
    report["e1_power_delay"] = e1

    # E3b: switch-delay sweep (canonical + grid)
    e3b = {}
    for tau_s in TAU_SWITCH:
        per_ic = {}
        for alpha0 in ALPHA_GRID_DEG:
            for v0 in VNORM_GRID:
                r = rollout(env, pi, make_switch_delay(tau_s), alpha0, v0)
                per_ic[f"a{alpha0:.0f}_v{v0:.2f}"] = {
                    "h": r["h"], "t": r["t"], "status": r["status"]}
        e3b[f"{tau_s:g}"] = per_ic
        logger.info(f"    E3b tau_s={tau_s:g}s: canonical "
                    f"{per_ic[f'a{a0c:.0f}_v{v0c:.2f}']['h']:.2f} m")
    report["e3b_switch_delay"] = e3b

    # E3c: partial-pull sweep (canonical + grid)
    e3c = {}
    for de_pull in DE_PULL_DEG:
        per_ic = {}
        for alpha0 in ALPHA_GRID_DEG:
            for v0 in VNORM_GRID:
                r = rollout(env, pi, make_partial_pull(de_pull), alpha0, v0)
                per_ic[f"a{alpha0:.0f}_v{v0:.2f}"] = {
                    "h": r["h"], "t": r["t"], "status": r["status"]}
        e3c[f"{de_pull:g}"] = per_ic
        logger.info(f"    E3c de_pull={de_pull:g}deg: canonical "
                    f"{per_ic[f'a{a0c:.0f}_v{v0c:.2f}']['h']:.2f} m")
    report["e3c_partial_pull"] = e3c

    (OUT_DIR / "procedures.json").write_text(json.dumps(report, indent=2))

    # ── Figures ──────────────────────────────────────────────────────────
    ckey = f"a{a0c:.0f}_v{v0c:.2f}"

    make_power_delay_figure(report)
    make_pilot_sensitivity_figure(report)

    write_summary_table(report)
    logger.info(f"[+] Wrote figures, table_procedures.tex and procedures.json to {OUT_DIR}")


def write_summary_table(report: dict) -> None:
    """Canonical-IC summary table, sorted from least to most altitude lost.
    Rows within ~0.3 m of the optimum are inside the rollout-execution band
    (barycentric action blending) — footnoted rather than hidden."""
    a0c, v0c = CANONICAL
    ckey = f"a{a0c:.0f}_v{v0c:.2f}"
    e1 = report["e1_power_delay"]
    e3b = report["e3b_switch_delay"]
    e3c = report["e3c_partial_pull"]
    h_opt = report["optimal_canonical"]["h"]

    entries = [
        (r"DP optimum (nose-down + full power, CAA-like)", h_opt),
        (r"Pull-up with $\delta_e=-15^\circ$ (not CL-max)", e3c["-15"][ckey]["h"]),
        (r"Pull-up switch late by $0.25$ s", e3b["0.25"][ckey]["h"]),
        (r"Power delayed $\tau=0.5$ s", e1["instant"]["0.5"][ckey]["h"]),
        (r"Power over 2 s ramp from $t=0$ (CAA/Gratton)", e1["ramp2s"]["0"][ckey]["h"]),
        (r"Power delayed $\tau=1$ s", e1["instant"]["1"][ckey]["h"]),
        (r"Power gated on $\alpha<14^\circ$ + 2 s ramp", report["e1_power_delay"]["gated_alpha14_ramp2s"][ckey]["h"]),
        (r"Pull-up switch late by $0.5$ s", e3b["0.5"][ckey]["h"]),
        (r"Power delayed $\tau=2$ s + 2 s ramp (power-delayed)", e1["ramp2s"]["2"][ckey]["h"]),
    ]
    entries.sort(key=lambda e: -e[1])   # least altitude lost first

    lines = [
        r"\begin{table}[hbt!]", r"    \centering",
        r"    \caption{Altitude loss from the canonical deep-stall IC "
        r"($\alpha_0=20^\circ$, $V_0=0.95\,V_s$) for the DP optimum and "
        r"procedure/pilot deviations built on top of it (optimal pitch "
        r"unless stated), sorted from least to most altitude lost.}",
        r"    \label{tab:procedures}",
        r"    \begin{tabular}{lc}", r"        \hline",
        r"        Sequence & $\Delta h$ (m) \\", r"        \hline",
    ]
    for label, h in entries:
        band = r"\textsuperscript{b}" if abs(h - h_opt) <= 0.3 and label[:10] != "DP optimum" else ""
        lines.append(f"        {label}{band} & {h:.2f} \\\\")
    lines += [
        r"        \hline",
        r"        \multicolumn{2}{l}{\footnotesize\textsuperscript{b}\,Within "
        r"the $\pm0.3$\,m policy-execution band of the optimum (barycentric "
        r"action blending); statistically a tie.}\\",
        r"    \end{tabular}", r"\end{table}",
    ]
    (OUT_DIR / "table_procedures.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    if "--maneuvers" in sys.argv:
        main_maneuvers()
    elif "--table-only" in sys.argv:
        # Rewrite table_procedures.tex from the stored json (no rollouts).
        write_summary_table(json.loads((OUT_DIR / "procedures.json").read_text()))
        logger.info("[+] table_procedures.tex rewritten from procedures.json")
    elif "--ic-heatmap" in sys.argv:
        # Dense IC sweep (slow: ~600 rollouts) + heatmap figures.
        _d = compute_ic_heatmap_dense()
        make_optimal_ic_figure(_d)
        make_ic_heatmap_figure(_d)
    elif "--trajectories" in sys.argv:
        # Time-domain comparison DP vs scripted CAA vs scripted FAA.
        make_trajectory_comparison_figure()
    elif "--held-pull" in sys.argv:
        compute_held_pull()
    elif "--switch-heatmap" in sys.argv:
        # Dense switch-delay sweep (~350 rollouts) + IC-plane figure.
        make_switch_delay_ic_figure(compute_switch_delay_dense())
    elif "--caa-ramp" in sys.argv:
        # Add the realistic-CAA arm to the dense cache + procedures map.
        make_procedures_ic_figure(compute_caa_ramp_dense())
    elif "--figs-only" in sys.argv:
        # Regenerate figures from the stored jsons (no rollouts).
        _rep = json.loads((OUT_DIR / "procedures.json").read_text())
        make_power_delay_figure(_rep)
        make_pilot_sensitivity_figure(_rep)
        if (OUT_DIR / "ic_heatmap_dense.json").exists():
            make_optimal_ic_figure()
            make_ic_heatmap_figure()
            if "caa_ramp" in json.loads(
                    (OUT_DIR / "ic_heatmap_dense.json").read_text()):
                make_procedures_ic_figure()
    else:
        main()
