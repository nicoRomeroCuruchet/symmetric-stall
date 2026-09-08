"""
paper_closed_loop_alpha_stats.py — the closed-loop angle-of-attack
statistics quoted in Sec. V.D, and the LaTeX table that carries them.

For every stalled entry on a 5 x 7 grid (alpha0 in {16,18,20,22,25}
deg, V0 in {0.70..1.00} Vs, gamma0 = 0, q0 = 0), the reference policy
is flown closed loop through the engine lag of the run configuration,
and two angles are extracted per recovered trajectory:

  * reversal alpha — the angle of attack at the first push-to-pull
    sign change of the COMMANDED elevator;
  * arc alpha — the mean angle of attack while the descent is being
    arrested (gamma < -0.5 deg and rising).

The first lands well above the power-off stall angle (the optimum
stops pushing while pitch-rate momentum completes the crossing); the
second settles on the marginal-trade crossing of the alpha-trade
figure to within a tenth of a degree.

Output: table_closed_loop_alpha.tex in paths.out_dir() (+ manuscript
copy in stall-paper/tables if present), stats on stdout.
"""
import logging
import os
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

ALPHA0_DEG = (16.0, 18.0, 20.0, 22.0, 25.0)
V0_NORM = (0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00)


def main():
    os.environ.setdefault("THRUST_MODEL", "riley")
    os.environ.setdefault(
        "STALL_POLICY",
        "data/policies/SymmetricStall_riley_56x81x80x41_thrust-riley.npz")
    from symmetric_stall import paths
    from symmetric_stall.aircraft.symmetric_stall import SymmetricStall
    from symmetric_stall.procedures import rollout, ctrl_optimal

    env = SymmetricStall()
    pi = paths.load_policy(env=env)

    sw, arc, n_ok, n_tot = [], [], 0, 0
    for a0 in ALPHA0_DEG:
        for v0 in V0_NORM:
            n_tot += 1
            r = rollout(env, pi, ctrl_optimal, a0, v0, record=True)
            if r["status"] != "recovered":
                logger.info("[!] not recovered: alpha0=%g V0=%g", a0, v0)
                continue
            n_ok += 1
            h = r["hist"]
            al = np.rad2deg(np.asarray(h["alpha"]))
            de = np.rad2deg(np.asarray(h["de_cmd"] if "de_cmd" in h
                                       else h["de"]))
            s = np.sign(de)
            idx = np.where((s[:-1] > 0) & (s[1:] <= 0))[0]
            if len(idx):
                sw.append(al[idx[0]])
            g = np.rad2deg(np.asarray(h["gamma"]))
            mask = (g < -0.5) & (np.gradient(g) > 0)
            if mask.sum() > 5:
                arc.append(al[mask].mean())

    sw, arc = np.array(sw), np.array(arc)

    def row(x):
        return np.median(x), np.percentile(x, 5), np.percentile(x, 95)

    logger.info("recovered %d/%d", n_ok, n_tot)
    logger.info("reversal alpha: median %.1f  p5 %.1f  p95 %.1f", *row(sw))
    logger.info("arc alpha:      median %.1f  p5 %.1f  p95 %.1f", *row(arc))

    table = r"""\begin{table}[H]
    \centering
    \caption{Where the flown recoveries reverse and ride: closed-loop
    statistics over %(ntot)d stalled entries
    ($\alpha_0 = 16$--$25^\circ$, $V_0 = 0.70$--$1.00\,V_s$, engine
    lag on; %(nok)d recovered). Reversal: first push-to-pull sign
    change of the commanded elevator. Arrest: mean $\alpha$ while the
    descent is being arrested ($\gamma < -0.5^\circ$ and rising).}
    \label{tab:closed_loop_alpha}
    \begin{tabular}{l c c c}
        \toprule
         & Median & $p_5$ & $p_{95}$ \\
        \midrule
        Elevator reversal $\alpha$ (deg) & $%(sw_m).1f$ & $%(sw_5).1f$ & $%(sw_95).1f$ \\
        Arrest $\alpha$ (deg) & $%(arc_m).1f$ & $%(arc_5).1f$ & $%(arc_95).1f$ \\
        \bottomrule
    \end{tabular}
\end{table}
""" % dict(nok=n_ok, ntot=n_tot,
           sw_m=row(sw)[0], sw_5=row(sw)[1], sw_95=row(sw)[2],
           arc_m=row(arc)[0], arc_5=row(arc)[1], arc_95=row(arc)[2])

    out_results = paths.out_dir()
    (out_results / "table_closed_loop_alpha.tex").write_text(table)
    out_paper = Path("stall-paper/tables")
    if out_paper.is_dir():
        (out_paper / "table_closed_loop_alpha.tex").write_text(table)
    else:
        logger.info("[i] %s not present; manuscript copy skipped", out_paper)
    logger.info("[+] table_closed_loop_alpha.tex written")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
