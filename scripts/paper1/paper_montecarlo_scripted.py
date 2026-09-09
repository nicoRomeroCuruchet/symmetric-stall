"""paper_montecarlo_scripted.py — the Monte Carlo certificate of the
Time-Domain section, flown with the SAME three pilots as the canonical
trajectory figure.

The sibling paper_montecarlo.py isolates power timing by giving every
arm the DP-optimal elevator. This script instead certifies the
maneuvers as flown: the DP optimum against the scripted CAA and FAA
procedures of make_maneuver (nose-down +15 deg until alpha < 14 deg,
alpha-hold pull at 13 deg, 2-s power ramp from t=0 or from the unstall
event), so the table and Fig. 4dof_trajectory describe one and the
same experiment. As there, the scripted pulls are capped to the pull
authority the optimum itself commands at each entry (see run_maneuvers
for why).

Same n = 1000 Latin-hypercube entries, same seed and bounds as the
sibling, engine lag from the run configuration.

Output: ic_montecarlo_scripted.json and table_montecarlo_scripted.tex
in paths.out_dir() (+ manuscript table copy if stall-paper/tables is
present), ranking statistics on stdout.
"""
import json
import logging
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

N = 1000
SEED = 20260723
BOUNDS = {
    "gamma0_deg": (-30.0, 0.0),
    "vnorm0": (0.40, 0.90),
    "alpha0_deg": (14.0, 20.0),
    "q0_deg": (-20.0, 20.0),
}
ARMS = ("optimal", "caa_scripted", "faa_scripted")

_env = _pi = None


def _init():
    global _env, _pi
    os.environ.setdefault("THRUST_MODEL", "riley")
    os.environ.setdefault(
        "STALL_POLICY",
        "data/policies/SymmetricStall_riley_56x81x80x41_thrust-riley.npz")
    logging.disable(logging.INFO)
    from symmetric_stall import paths
    from symmetric_stall.aircraft.symmetric_stall import SymmetricStall
    _env = SymmetricStall()
    _pi = paths.load_policy(env=_env)


def _job(args):
    g0, v0, a0, q0 = args
    from symmetric_stall import procedures as P
    # One entry, three arms. The optimal arm goes first and recorded:
    # its deepest COMMANDED pull caps the scripted pilots, exactly as
    # in run_maneuvers / the trajectory figure.
    r_opt = P.rollout(_env, _pi, P.ctrl_optimal, a0, v0,
                      gamma0_deg=g0, q0_deg=q0, record=True)
    cap = float(np.min(r_opt["hist"]["de_cmd"]))
    out = [(float(r_opt["h"]), r_opt["status"])]
    for start in ("t0", "unstall"):
        r = P.rollout(_env, _pi,
                      P.make_maneuver(start, de_pull_limit=cap),
                      a0, v0, gamma0_deg=g0, q0_deg=q0)
        out.append((float(r["h"]), r["status"]))
    return out


def _lhs(n, d, rng):
    u = (rng.random((n, d)) + np.arange(n)[:, None]) / n
    for j in range(d):
        rng.shuffle(u[:, j])
    return u


def main():
    rng = np.random.default_rng(SEED)
    names = list(BOUNDS)
    u = _lhs(N, len(names), rng)
    pts = {k: BOUNDS[k][0] + u[:, i] * (BOUNDS[k][1] - BOUNDS[k][0])
           for i, k in enumerate(names)}

    jobs = [(float(pts["gamma0_deg"][i]), float(pts["vnorm0"][i]),
             float(pts["alpha0_deg"][i]), float(pts["q0_deg"][i]))
            for i in range(N)]
    nw = max(1, min(12, (os.cpu_count() or 2) - 2))
    logger.info("[mc] %d entries x %d arms over %d workers",
                len(jobs), len(ARMS), nw)
    with mp.get_context("spawn").Pool(nw, initializer=_init) as pool:
        out = pool.map(_job, jobs, chunksize=4)

    from symmetric_stall import paths, runconfig
    data = {"run_config": runconfig.describe(), "n": N, "seed": SEED,
            "bounds": BOUNDS,
            "points": {k: [float(x) for x in v] for k, v in pts.items()}}
    for j, arm in enumerate(ARMS):
        data[arm] = {"h": [row[j][0] for row in out],
                     "status": [row[j][1] for row in out]}

    out_dir = paths.out_dir()
    (out_dir / "ic_montecarlo_scripted.json").write_text(
        json.dumps(data, indent=1))

    h = {k: np.array(data[k]["h"]) for k in ARMS}
    st = {k: np.array(data[k]["status"]) for k in ARMS}
    ok = np.all([st[k] == "recovered" for k in ARMS], axis=0)
    opt, caa, faa = (h[k][ok] for k in ARMS)
    adv = caa - faa
    frac = 100 * np.mean(caa > faa)
    exc = {k: np.median(opt - h[k][ok])
           for k in ("caa_scripted", "faa_scripted")}
    e95 = {k: np.percentile(opt - h[k][ok], 95)
           for k in ("caa_scripted", "faa_scripted")}
    med = {k: np.median(h[k][ok]) for k in ARMS}
    p5 = {k: np.percentile(h[k][ok], 5) for k in ARMS}
    for arm in ARMS:
        bad = np.where(st[arm] != "recovered")[0]
        if len(bad):
            logger.info("%s: %d not recovered (entries %s)",
                        arm, len(bad), bad[:10].tolist())
    logger.info("recovered under every arm: %d/%d", ok.sum(), N)
    logger.info("CAA beats FAA at %.1f%% of entries; advantage median "
                "%.2f m, smallest %.2f m", frac, np.median(adv), adv.min())

    table = r"""\begin{table}[H]
    \centering
    \caption{The three maneuvers of Fig.~\ref{fig:4dof_trajectory_riley}
    flown from $%(n)d$ Latin-hypercube samples of the stalled-entry set
    $\gamma_0 \in [-30^\circ, 0]$, $V_0/V_s \in [0.40, 0.90]$,
    $\alpha_0 \in [14^\circ, 20^\circ]$, $q_0 \in [-20, 20]^\circ$/s,
    through the engine lag. The scripted pulls are capped to the pull
    authority the optimum commands at each entry, so the comparison
    isolates the procedure rather than the pilot's aggressiveness.
    Percentiles describe the uniform sampling box, not an operational
    entry distribution; the per-entry ranking reported below is
    distribution-free.}
    \label{tab:montecarlo_scripted}
    \setlength{\tabcolsep}{14pt}\renewcommand{\arraystretch}{1.15}
    \begin{tabular}{lrrrr}
        \hline
        & \multicolumn{2}{c}{$\Delta h$ (m)} & \multicolumn{2}{c}{excess over optimum (m)} \\
        Maneuver & median & 5th pct & median & 95th pct \\
        \hline
        DP optimum & $%(o_m).2f$ & $%(o_5).2f$ & --- & --- \\
        Scripted CAA (power with the push) & $%(c_m).2f$ & $%(c_5).2f$ & $%(c_e).2f$ & $%(c_e95).2f$ \\
        Scripted FAA (power after unstall) & $%(f_m).2f$ & $%(f_5).2f$ & $%(f_e).2f$ & $%(f_e95).2f$ \\
        \hline
    \end{tabular}
    \\[2pt]
    \footnotesize The CAA sequence loses less than the FAA sequence at $%(frac).1f\%%$ of the sampled entries (median advantage $%(adv_m).2f$\,m, smallest $%(adv_min).2f$\,m); $%(nok)d$ of $%(n)d$ entries closed the recovery under every maneuver.
\end{table}
""" % dict(n=N, nok=int(ok.sum()), frac=frac,
           adv_m=np.median(adv), adv_min=adv.min(),
           o_m=med["optimal"], o_5=p5["optimal"],
           c_m=med["caa_scripted"], c_5=p5["caa_scripted"],
           c_e=exc["caa_scripted"], c_e95=e95["caa_scripted"],
           f_m=med["faa_scripted"], f_5=p5["faa_scripted"],
           f_e=exc["faa_scripted"], f_e95=e95["faa_scripted"])

    (out_dir / "table_montecarlo_scripted.tex").write_text(table)
    out_paper = Path("stall-paper/tables")
    if out_paper.is_dir():
        (out_paper / "table_montecarlo_scripted.tex").write_text(table)
    logger.info("[+] ic_montecarlo_scripted.json and "
                "table_montecarlo_scripted.tex written")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
