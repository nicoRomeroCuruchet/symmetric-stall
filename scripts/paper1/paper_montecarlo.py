"""
paper_montecarlo.py — the 4-D Monte Carlo certificate of Sec. V.E,
re-flown on Riley's engine.

Every arm (DP optimum, CAA simultaneous-with-ramp, FAA gated on
alpha < alpha_s, Gratton power-delayed) is flown from the same n
Latin-hypercube entries spanning the stalled-entry set

    gamma0 in [-30, 0] deg,   V0/Vs in [0.40, 0.90],
    alpha0 in [14, 20] deg,   q0 in [-20, 20] deg/s,

with the engine lag of the run configuration. The planar maps of the
dense IC sweep hold gamma0 = q0 = 0; this samples the interactions
they cannot show. Statuses are recorded per entry and arm: a sample
that does not recover under some arm must be visible, not averaged
in silently.

Output: ic_montecarlo.json and table_montecarlo.tex in
paths.out_dir() (+ manuscript table copy if stall-paper/tables is
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
ARMS = ("optimal", "caa_ramp", "gated", "pd")

_env = _pi = _ctrls = None


def _init():
    global _env, _pi, _ctrls
    os.environ.setdefault("THRUST_MODEL", "riley")
    os.environ.setdefault(
        "STALL_POLICY",
        "data/policies/SymmetricStall_riley_56x81x80x41_thrust-riley.npz")
    logging.disable(logging.INFO)
    from symmetric_stall import paths
    from symmetric_stall.aircraft.symmetric_stall import SymmetricStall
    from symmetric_stall import procedures as P
    _env = SymmetricStall()
    _pi = paths.load_policy(env=_env)
    _ctrls = {
        "optimal": P.ctrl_optimal,
        "caa_ramp": P.make_power_delay(0.0, ramp=True),
        "gated": P.make_power_gated(ramp=True),
        "pd": P.make_power_delay(2.0, ramp=True),
    }


def _job(args):
    arm, g0, v0, a0, q0 = args
    from symmetric_stall.procedures import rollout
    r = rollout(_env, _pi, _ctrls[arm], a0, v0,
                gamma0_deg=g0, q0_deg=q0)
    return float(r["h"]), r["status"]


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

    jobs = [(arm, float(pts["gamma0_deg"][i]), float(pts["vnorm0"][i]),
             float(pts["alpha0_deg"][i]), float(pts["q0_deg"][i]))
            for arm in ARMS for i in range(N)]
    nw = max(1, min(12, (os.cpu_count() or 2) - 2))
    logger.info("[mc] %d rollouts over %d workers", len(jobs), nw)
    with mp.get_context("spawn").Pool(nw, initializer=_init) as pool:
        out = pool.map(_job, jobs, chunksize=8)

    from symmetric_stall import paths, runconfig
    data = {"run_config": runconfig.describe(), "n": N, "seed": SEED,
            "bounds": BOUNDS,
            "points": {k: [float(x) for x in v] for k, v in pts.items()}}
    for j, arm in enumerate(ARMS):
        chunk = out[j * N:(j + 1) * N]
        data[arm] = {"h": [h for h, _ in chunk],
                     "status": [s for _, s in chunk]}

    out_dir = paths.out_dir()
    (out_dir / "ic_montecarlo.json").write_text(json.dumps(data, indent=1))

    h = {k: np.array(data[k]["h"]) for k in ARMS}
    st = {k: np.array(data[k]["status"]) for k in ARMS}
    ok = np.all([st[k] == "recovered" for k in ARMS], axis=0)
    opt, caa, faa, pd_ = (h[k][ok] for k in ARMS)
    adv = caa - faa
    frac = 100 * np.mean(caa > faa)
    exc = {k: np.median(opt - h[k][ok]) for k in ("caa_ramp", "gated", "pd")}
    e95 = {k: np.percentile(opt - h[k][ok], 95)
           for k in ("caa_ramp", "gated", "pd")}
    med = {k: np.median(h[k][ok]) for k in ARMS}
    p5 = {k: np.percentile(h[k][ok], 5) for k in ARMS}
    logger.info("recovered under every arm: %d/%d", ok.sum(), N)
    logger.info("CAA beats FAA at %.1f%% of entries; advantage median "
                "%.2f m, smallest %.2f m", frac, np.median(adv), adv.min())

    table = r"""\begin{table}[H]
    \centering
    \caption{Coverage of the stalled-entry set: every arm flown from
    the same $%(n)d$ Latin-hypercube samples of
    $\gamma_0 \in [-30^\circ, 0]$, $V_0/V_s \in [0.40, 0.90]$,
    $\alpha_0 \in [14^\circ, 20^\circ]$, $q_0 \in [-20, 20]^\circ$/s,
    through the engine lag. Unlike the planar maps of
    Fig.~\ref{fig:ic_procedures}, this samples the interactions
    between entry states, so the ranking is certified over the set
    rather than on a slice of it. Percentiles describe the uniform
    sampling box, not an operational entry distribution; the
    per-entry ranking reported below is distribution-free.}
    \label{tab:montecarlo}
    \setlength{\tabcolsep}{14pt}\renewcommand{\arraystretch}{1.15}
    \begin{tabular}{lrrrr}
        \hline
        & \multicolumn{2}{c}{$\Delta h$ (m)} & \multicolumn{2}{c}{excess over optimum (m)} \\
        Arm & median & 5th pct & median & 95th pct \\
        \hline
        DP optimum & $%(o_m).2f$ & $%(o_5).2f$ & --- & --- \\
        CAA (simultaneous) & $%(c_m).2f$ & $%(c_5).2f$ & $%(c_e).2f$ & $%(c_e95).2f$ \\
        FAA (gated on $\alpha < \alpha_s$) & $%(f_m).2f$ & $%(f_5).2f$ & $%(f_e).2f$ & $%(f_e95).2f$ \\
        Power-delayed (Gratton) & $%(p_m).2f$ & $%(p_5).2f$ & $%(p_e).2f$ & $%(p_e95).2f$ \\
        \hline
    \end{tabular}
    \\[2pt]
    \footnotesize The CAA sequence loses less than the FAA sequence at $%(frac).1f\%%$ of the sampled entries (median advantage $%(adv_m).2f$\,m, smallest $%(adv_min).2f$\,m); $%(nok)d$ of $%(n)d$ entries closed the recovery under every arm.
\end{table}
""" % dict(n=N, nok=int(ok.sum()), frac=frac,
           adv_m=np.median(adv), adv_min=adv.min(),
           o_m=med["optimal"], o_5=p5["optimal"],
           c_m=med["caa_ramp"], c_5=p5["caa_ramp"],
           c_e=exc["caa_ramp"], c_e95=e95["caa_ramp"],
           f_m=med["gated"], f_5=p5["gated"],
           f_e=exc["gated"], f_e95=e95["gated"],
           p_m=med["pd"], p_5=p5["pd"],
           p_e=exc["pd"], p_e95=e95["pd"])

    (out_dir / "table_montecarlo.tex").write_text(table)
    out_paper = Path("stall-paper/tables")
    if out_paper.is_dir():
        (out_paper / "table_montecarlo.tex").write_text(table)
    logger.info("[+] ic_montecarlo.json and table_montecarlo.tex written")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
