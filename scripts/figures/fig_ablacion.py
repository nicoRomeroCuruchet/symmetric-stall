s = open("ablacion.py").read()
s = s.replace("import numpy as np\n\nlogging.disable",
              "import numpy as np\nimport matplotlib; matplotlib.use('Agg')\n"
              "import matplotlib.pyplot as plt\nimport main\n\nlogging.disable")
s += '''

# ───────────────────────── figura ─────────────────────────
ESTILO = {
    "optimum (DP)":                  ("#0072B2", "-",  1.9),
    "trigger 14 + ramp   (= CAA)": ("#D55E00", "-",  1.4),
    "trigger 17 + ramp":           ("#D55E00", ":",  1.4),
    "trigger 14 + step":         ("#009E73", "-",  1.4),
    "trigger 17 + step":         ("#009E73", ":",  1.4),
}
PAN = [("gamma", r"$\\gamma$ (deg)", np.rad2deg), ("v_norm", r"$V/V_s$", lambda x: x),
       ("alpha", r"$\\alpha$ (deg)", np.rad2deg), ("q", r"$q$ (deg/s)", np.rad2deg),
       ("de", r"$\\delta_e$ (deg)", np.rad2deg), ("dt_ctrl", r"$\\delta_t$", lambda x: x),
       ("h", "altitude (m)", lambda x: x)]

fig, axes = plt.subplots(len(PAN), 1, figsize=(7.4, 12.2), sharex=True)
for ax, (k, et, cv) in zip(axes, PAN):
    for nom, (col, ls, lw) in ESTILO.items():
        h = res[nom]["hist"]
        ax.plot(h["t"], cv(np.asarray(h[k])), lw=lw, ls=ls, color=col,
                label=nom, zorder=3)
        if k == "h":
            y = cv(np.asarray(h[k])); i = int(np.argmin(y))
            ax.annotate("%.1f" % y[i], xy=(h["t"][i], y[i]), xytext=(6, 0),
                        textcoords="offset points", fontsize=7, color=col,
                        va="center")
    if k == "alpha":
        ax.axhline(14.0, color="0.5", lw=0.8, ls="--", zorder=1)
        ax.annotate(r"$\\alpha_s=14^\\circ$", xy=(0.99, 14.0),
                    xycoords=("axes fraction", "data"), ha="right", va="bottom",
                    fontsize=7, color="0.35")
    if k == "dt_ctrl":
        ax.set_ylim(-0.02, 1.05)
        ax.annotate("2 s ramp vs step", xy=(0.99, 0.45),
                    xycoords=("axes fraction", "data"), ha="right",
                    fontsize=7.5, color="0.35")
    if k in ("gamma", "h"):
        ax.axhline(0.0, color="0.85", lw=0.6, zorder=0)
    ax.set_ylabel(et, fontsize=9); ax.grid(True, color="0.92", lw=0.5)
    ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"): ax.spines[sp].set_color("0.6")
    ax.tick_params(labelsize=8, color="0.6")
axes[-1].set_xlabel("time (s)", fontsize=9)
axes[0].legend(loc="lower left", fontsize=7.5, frameon=False, ncol=2,
               bbox_to_anchor=(0.0, 1.02))
fig.suptitle(r"Trigger vs power ablation — $V_0=%.2f\\,V_s$, $\\alpha_0=%.0f^\\circ$"
             % (V0, A0), fontsize=10, y=0.988)
fig.tight_layout(rect=[0, 0, 1, 0.955])
out = main.RESULTS_DIR / ("ablacion_v%03d.png" % round(V0 * 100))
fig.savefig(out, dpi=200)
print("-> %s" % out)
'''
open("ablacion.py", "w").write(s)
print("ablacion.py ahora grafica")
