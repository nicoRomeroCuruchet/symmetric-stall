"""Cuatro brazos: quien activa el controlador, y cuando.

  (1) DP disparado por el DETECTOR automatico a 0.33 s
  (2) DP disparado por el PILOTO a 1.00 s (aprieta el boton cuando se da cuenta)
  (3) CAA volada por el piloto desde 1.00 s
  (4) FAA volada por el piloto desde 1.00 s

Los cuatro: el piloto TIRA mientras no reacciona, y el motor tiene el retardo
de primer orden de Riley. (2) aisla cuanto vale el detector automatico: mismo
controlador, misma planta, solo cambia quien lo engancha y cuando.

Estilo del paper 1. Uso: fig_activacion.py <npz> <V0> [t_det] [t_pil] [tau_m]
"""
import sys
import logging
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.disable(logging.INFO)

from symmetric_stall import train as main
import escenario as E

# rodar() de escenario.py usa T_DP para el modo dp; se lo cambiamos al vuelo
def correr(modo, t_det):
    orig_dp, orig_pil = E.T_DP, E.T_PIL
    if modo == "dp":
        E.T_DP = t_det
    else:
        E.T_PIL = t_det
    try:
        return E.rodar(modo)
    finally:
        E.T_DP, E.T_PIL = orig_dp, orig_pil


T_DET, T_PIL = E.T_DP, E.T_PIL
RUNS = [
    ("DP, automatic trigger (%.2f s)" % T_DET, "#2C4B9E", "-",  correr("dp",  T_DET)),
    ("DP, pilot trigger (%.2f s)"     % T_PIL, "#2C4B9E", ":",  correr("dp",  T_PIL)),
    ("CAA pilot (%.2f s)"             % T_PIL, "#E8742A", "--", correr("caa", T_PIL)),
    ("FAA pilot (%.2f s)"             % T_PIL, "#2CA02C", "-.", correr("faa", T_PIL)),
]

print("V=%.2f Vs   motor tau=%.2f s   sin reaccionar de=%.0f deg\n"
      % (E.V0, E.TAU_M, np.rad2deg(E.DE_NOREAC)))
print("%-34s %10s %8s %11s" % ("brazo", "h_min", "dur", "estado"))
print("-" * 66)
for lab, _, _, r in RUNS:
    print("%-34s %+10.3f %7.2fs %11s" % (lab, r["hmin"], r["t"], r["est"]))
auto, pil = RUNS[0][3]["hmin"], RUNS[1][3]["hmin"]
print("\n  el DETECTOR automatico vale:  %+.3f m  (mismo controlador, %.2f s antes)"
      % (auto - pil, T_PIL - T_DET))
print("  el CONTROLADOR vale, a igual instante de activacion (%.2f s):" % T_PIL)
print("      contra CAA  %+.3f m" % (pil - RUNS[2][3]["hmin"]))
print("      contra FAA  %+.3f m" % (pil - RUNS[3][3]["hmin"]))

rc = {"font.family": "serif", "mathtext.fontset": "stix",
      "font.size": 12, "axes.labelsize": 13,
      "xtick.labelsize": 11, "ytick.labelsize": 11,
      "axes.spines.top": False, "axes.spines.right": False}
SIG = [("gamma",  r"$\gamma$ (deg)",  np.rad2deg, "plot"),
       ("v_norm", r"$V/V_s$ (--)",    None,       "plot"),
       ("alpha",  r"$\alpha$ (deg)",  np.rad2deg, "plot"),
       ("q",      r"$q$ (deg/s)",     np.rad2deg, "plot"),
       ("de",     r"$\delta_e$ (deg)", np.rad2deg, "step"),
       ("dt_ef",  r"$\delta_t$ (--)",  None,       "step"),
       ("h",      r"$\Delta h$ (m)",   None,       "plot")]

with plt.rc_context(rc):
    fig = plt.figure(figsize=(10.0, 6.8))
    gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1, 1.4],
                          hspace=0.42, wspace=0.24)
    slots = [gs[0, 0], gs[0, 1], gs[1, 0], gs[1, 1], gs[2, 0], gs[2, 1], gs[3, :]]
    axs = []
    t_end = max(r["H"]["t"][-1] for _, _, _, r in RUNS)
    for k, (slot, (key, ylab, conv, estilo)) in enumerate(zip(slots, SIG)):
        ax = fig.add_subplot(slot)
        for lab, col, ls, r in RUNS:
            t = np.asarray(r["H"]["t"]); y = np.asarray(r["H"][key])
            if conv is not None:
                y = conv(y)
            ancho = 2.4 if ls == "-." else 1.6
            if estilo == "step":
                ax.step(t, y, color=col, lw=ancho, ls=ls, where="post", label=lab,
                        zorder=2 if ancho > 2 else 3)
            else:
                ax.plot(t, y, color=col, lw=ancho, ls=ls, label=lab,
                        zorder=2 if ancho > 2 else 3)
        ax.set_ylabel(ylab); ax.set_xlim(0.0, t_end)
        ax.grid(True, linestyle=":", alpha=0.55)
        ax.text(0.0, 1.02, "(%s)" % "abcdefg"[k], transform=ax.transAxes,
                fontsize=11, va="bottom", ha="left")
        axs.append(ax)
    for ax in axs:
        ax.axvline(T_DET, color="0.45", linestyle=":", linewidth=1.0)
        ax.axvline(T_PIL, color="0.45", linestyle=":", linewidth=1.0)
    axs[0].annotate(r"$t_{det}$", xy=(T_DET, 0.06), xycoords=("data", "axes fraction"),
                    fontsize=9, ha="right", va="bottom", color="0.35",
                    xytext=(-3, 0), textcoords="offset points")
    axs[0].annotate(r"$t_{pilot}$", xy=(T_PIL, 0.06), xycoords=("data", "axes fraction"),
                    fontsize=9, ha="left", va="bottom", color="0.35",
                    xytext=(3, 0), textcoords="offset points")
    axs[2].axhline(14.0, color="0.45", linestyle="--", linewidth=1.0)
    axs[6].set_xlabel("$t$ (s)")
    axs[6].legend(loc="lower left", fontsize=9.5, frameon=False, ncol=2)
    for ext, kw in (("png", dict(dpi=300)), ("pdf", {})):
        fig.savefig(main.RESULTS_DIR / ("3_maniobras/fig_activacion_v%03d.%s"
                                        % (round(E.V0 * 100), ext)),
                    bbox_inches="tight", **kw)
print("\n-> 3_maniobras/fig_activacion_v%03d.{png,pdf}" % round(E.V0 * 100))
