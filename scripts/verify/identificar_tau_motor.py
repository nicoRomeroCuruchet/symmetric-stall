"""Identify Riley's engine time constant tau_e from his own published figures.

TM-86309 puts a first-order lag between the throttle handle and the throttle the
aerodynamic model sees, eq. (A4), and then never tabulates its time constant. It
is the only constant of appendix A left out: (A3) prints 0.65 and 0.35, (A9)
prints the T0/T1 table, (A12) prints the N0/N1/N2 table, (A4) prints nothing.

The value survives in the figures. Riley flies throttle chops and plots the
COMMAND and the ENGINE SPEED against the same time axis (figs. 16 and 18), and
appendix A closes the loop between them:

    (A4)   delta_t  = 1/(tau_e s + 1) delta_t,c          <- the unknown
    (A3)   delta_t' = 0.65 delta_t + 0.35
    (A12)  N        = N0(delta_t') + N1(delta_t') V + N2(delta_t') V^2

so N(t) is an observation of delta_t(t), V(t) is plotted two panels below, and
tau_e is the only free parameter left. This script digitises the three traces
off the rasterised page, replays the published command through the lag, and
fits tau_e to the published engine speed.

    ./.venv/bin/python scripts/verify/identificar_tau_motor.py [--plot]

Reads papers/Riley.pdf, which is gitignored; the numbers it produces are quoted
in symmetric_stall/engine.py, which is not.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PDF = ROOT / "papers" / "Riley.pdf"

#: page in the PDF -> what the figure is. Riley's four throttle chops.
PAGES = {69: "fig 16(a) baseline", 70: "fig 16(b) modified",
         73: "fig 18(a) baseline", 74: "fig 18(b) modified"}

#: Panels down the right-hand column of every one of these figures.
#: psi, theta, phi, delta_t,c, N, V, h, hdot
I_THR, I_N, I_V = 3, 4, 5

#: Full-scale of the three panels we read, and the time span of the page.
THR_FS, N_FS, V_FS, T_SPAN = 1.0, 5000.0, 250.0, 30.0

#: Seeds for the curve tracker, in data units: roughly where each trace starts.
THR_0, N_0, V_0 = 0.82, 2400.0, 120.0

#: Riley (A12): engine speed against the intermediate throttle and airspeed.
DTP = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
N0 = np.array([-2350.0, -500.0, 700.0, 1620.0, 2100.0, 2520.0])
N1 = np.array([17.25, 11.00, 3.70, -1.65, -2.35, -2.55])
N2 = np.array([0.0, 0.0040, 0.0200, 0.0315, 0.0300, 0.0255])

TAU_GRID = np.arange(0.05, 3.001, 0.01)
FIT_HORIZON_S = 14.0          # past this the phugoid drags V and N with it


def rpm(delta_t, V):
    """Riley (A3) + (A12): engine speed from throttle and airspeed."""
    dtp = 0.65 * delta_t + 0.35
    return (np.interp(dtp, DTP, N0) + np.interp(dtp, DTP, N1) * V
            + np.interp(dtp, DTP, N2) * V * V)


def lag(cmd, dt, tau):
    """Riley (A4), exact discretisation over a step of `dt`."""
    out = np.empty_like(cmd)
    d, a = cmd[0], 1.0 - np.exp(-dt / tau)
    for k in range(cmd.size):
        out[k] = d
        d += (cmd[k] - d) * a
    return out


# ── reading the page ─────────────────────────────────────────────────────

def _longest_run(row):
    best = run = 0
    for v in row:
        run = run + 1 if v else 0
        best = max(best, run)
    return best


def page_panels(page, workdir):
    """Locate the panel boxes of the right-hand column on a rasterised page.

    A panel border is a long horizontal run of ink. Two traps: a panel that
    carries a zero reference line inside it is detected as two half panels and
    has to be merged back, and the page has a SECOND column of panels whose
    left border sits further left -- taking the min over the whole width picks
    that one and stretches every trace across both columns.
    """
    from PIL import Image

    stem = workdir / f"p{page}"
    subprocess.run(["pdftoppm", "-f", str(page), "-l", str(page), "-r", "400",
                    "-png", str(PDF), str(stem)], check=True)
    hits = sorted(workdir.glob(f"p{page}-*.png"))
    if not hits:
        raise FileNotFoundError(f"pdftoppm produced nothing for page {page}")
    dark = np.array(Image.open(hits[0]).convert("L")) < 128
    H, W = dark.shape
    x0c = W // 2

    col = dark[:, x0c:]
    runs = np.array([_longest_run(col[r]) for r in range(H)])
    # The panel box is ~620 px wide at 400 dpi. Bounding the run length from
    # BOTH sides rejects the caption rules, and works on the degraded scans
    # where a fraction-of-maximum threshold finds only the caption.
    cand = np.where((runs > 480) & (runs < 700))[0]
    if cand.size == 0:
        raise ValueError("no panel borders found (scan too degraded?)")

    edges, run = [], [cand[0]]
    for r in cand[1:]:
        if r - run[-1] <= 3:
            run.append(r)
        else:
            edges.append(int(np.mean(run)))
            run = [r]
    edges.append(int(np.mean(run)))

    raw = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
    raw = [(a, b) for a, b in raw if 100 < b - a < 400]
    if not raw:
        raise ValueError("panel borders found but none of a plausible height")
    full = max(b - a for a, b in raw)
    panels, k = [], 0
    while k < len(raw):
        a, b = raw[k]
        if (k + 1 < len(raw) and raw[k + 1][0] == b
                and abs((raw[k + 1][1] - a) - full) < 0.15 * full):
            panels.append((a, raw[k + 1][1]))       # zero line inside
            k += 2
        else:
            panels.append((a, b))
            k += 1
    if len(panels) <= max(I_THR, I_N, I_V):
        raise ValueError(f"only {len(panels)} panels resolved")

    a, b = panels[0]
    colsum = dark[a:b, x0c:].sum(axis=0)
    vcand = np.where(colsum > 0.8 * (b - a))[0] + x0c
    return dark, panels, int(vcand.min()), int(vcand.max())


def trace(dark, a, b, xl, xr, y_top, y_bot, start):
    """Digitise one curve inside a panel, following it by continuity.

    A column can hold ink from more than one curve, so at each column take the
    cluster of dark pixels nearest the last accepted point. Two guards: a real
    curve cannot jump a tenth of the panel between adjacent columns (anything
    that does is a tick, a label or scanner dust), and a curve riding the axis
    is invisible under the border margin, so an empty column holds the previous
    value rather than breaking the trace.
    """
    M = 3
    prev = (start - y_top) / (y_bot - y_top) * (b - a)
    t, y = [], []
    for x in range(xl + 4, xr - 3):
        colpix = np.where(dark[a + M:b - M, x])[0]
        pick = None
        if colpix.size:
            groups = [g for g in np.split(colpix, np.where(np.diff(colpix) > 3)[0] + 1)
                      if g.size >= 2]
            if groups:
                cents = np.array([g.mean() + M for g in groups])
                c = cents[np.argmin(np.abs(cents - prev))]
                if abs(c - prev) < 0.10 * (b - a):
                    pick = c
        prev = prev if pick is None else pick
        t.append(T_SPAN * (x - xl) / (xr - xl))
        y.append(y_top + (prev / (b - a)) * (y_bot - y_top))
    return np.asarray(t), np.asarray(y)


def identify(page, workdir):
    """Digitise one figure and fit tau_e to it."""
    dark, panels, xl, xr = page_panels(page, workdir)
    t_c, cmd_r = trace(dark, *panels[I_THR], xl, xr, THR_FS, 0.0, THR_0)
    t_N, N_r = trace(dark, *panels[I_N], xl, xr, N_FS, 0.0, N_0)
    t_V, V_r = trace(dark, *panels[I_V], xl, xr, V_FS, 0.0, V_0)

    hi, lo = cmd_r[0], cmd_r.min()
    def cross(f):
        return t_c[np.argmax(cmd_r <= hi - f * (hi - lo))]

    tf = np.linspace(0.0, FIT_HORIZON_S, 1400)
    cmd = np.interp(tf, t_c, cmd_r)
    V = np.interp(tf, t_V, V_r)
    n_obs = np.interp(tf, t_N, N_r)
    dt = tf[1] - tf[0]
    errs = np.array([np.sqrt(np.mean((rpm(lag(cmd, dt, tau), V) - n_obs) ** 2))
                     for tau in TAU_GRID])
    i = int(np.argmin(errs))
    return {"page": page, "tau": float(TAU_GRID[i]), "rms": float(errs[i]),
            "chop": (float(cross(0.10)), float(cross(0.90))),
            "n_hi": float(N_r.max()), "n_lo": float(N_r.min()),
            "thr0": float(cmd_r[0]), "v0": float(V_r[0]),
            "taus": TAU_GRID, "errs": errs,
            "t": tf, "cmd": cmd, "V": V, "n_obs": n_obs}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plot", action="store_true",
                    help="write the fit figure next to the results")
    args = ap.parse_args(argv)

    if not PDF.exists():
        print(f"error: {PDF} not found (papers/ is gitignored)", file=sys.stderr)
        return 1

    out, results = [], []
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        print(f"{'figure':>22} {'chop':>17} {'N, rpm':>16} {'tau_e':>8} {'rms':>7}")
        print("-" * 76)
        for page, name in PAGES.items():
            try:
                r = identify(page, work)
            except Exception as exc:
                print(f"{name:>22}   unreadable: {type(exc).__name__}: {exc}")
                continue
            results.append(r)
            print(f"{name:>22} {r['chop'][0]:6.2f}->{r['chop'][1]:5.2f} s "
                  f"{r['n_hi']:7.0f}->{r['n_lo']:5.0f} {r['tau']:8.2f} {r['rms']:7.0f}")
            out.append((name, r))

        if not results:
            print("\nno figure could be read", file=sys.stderr)
            return 1

        taus = np.array([r["tau"] for r in results])
        print(f"\ntau_e = {taus.mean():.2f} s   "
              f"(spread {taus.min():.2f}-{taus.max():.2f}, n = {taus.size})")

        # Independent checks that the digitiser is reading the page.
        r = results[0]
        print("\nchecks:")
        print(f"  chop duration        {r['chop'][1] - r['chop'][0]:.2f} s  "
              f"(Riley's text: 'occurred in 2 sec')")
        # Against the START of the trace, not its maximum: the max can sit on
        # a scanner speck and would flatter the check.
        print(f"  trim rpm, (A3)+(A12) {rpm(r['thr0'], r['v0']):.0f}  "
              f"(figure: {r['n_obs'][0]:.0f})")
        print(f"  idle rpm, (A3)+(A12) {rpm(0.0, r['V'].min()):.0f}  "
              f"(figure: {r['n_lo']:.0f})")

        if args.plot:
            _plot(out)
    return 0


def _plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from symmetric_stall import paths
    from symmetric_stall.engine import RILEY_TAU_E

    with plt.rc_context({"font.family": "serif", "mathtext.fontset": "stix",
                         "font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False}):
        fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.0))
        for name, r in out:
            axs[0].plot(r["taus"], r["errs"], lw=1.6, label=name)
        axs[0].axvline(RILEY_TAU_E, color="0.45", ls="--", lw=1.0)
        axs[0].annotate(rf"$\tau_e = {RILEY_TAU_E:g}$ s", xy=(RILEY_TAU_E, 0.95),
                        xycoords=("data", "axes fraction"), xytext=(5, 0),
                        textcoords="offset points", fontsize=10, color="0.45")
        axs[0].set_xlabel(r"$\tau_e$ (s)")
        axs[0].set_ylabel("rms residual in $N$ (rpm)")
        axs[0].set_xlim(0, 2.0)
        axs[0].legend(fontsize=9, frameon=False)
        axs[0].grid(True, ls=":", alpha=0.55)
        axs[0].text(0.0, 1.02, "(a)", transform=axs[0].transAxes, va="bottom")

        name, r = out[0]
        axs[1].plot(r["t"], r["n_obs"], color="0.25", lw=2.0,
                    label=f"{name}, digitised")
        dt = r["t"][1] - r["t"][0]
        for tau, ls in ((0.1, ":"), (RILEY_TAU_E, "-"), (2.0, "--")):
            axs[1].plot(r["t"], rpm(lag(r["cmd"], dt, tau), r["V"]), lw=1.4,
                        ls=ls, label=rf"(A4) with $\tau_e = {tau:g}$ s")
        axs[1].set_xlabel("Time (s)")
        axs[1].set_ylabel(r"$N$ (rpm)")
        axs[1].legend(fontsize=9, frameon=False)
        axs[1].grid(True, ls=":", alpha=0.55)
        axs[1].text(0.0, 1.02, "(b)", transform=axs[1].transAxes, va="bottom")

        fig.tight_layout()
        for ext in ("png", "pdf"):
            dest = paths.out_dir() / f"fig_engine_tau_id.{ext}"
            fig.savefig(dest, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"\n[+] fig_engine_tau_id.{{png,pdf}} -> {paths.out_dir()}")


if __name__ == "__main__":
    sys.exit(main())
