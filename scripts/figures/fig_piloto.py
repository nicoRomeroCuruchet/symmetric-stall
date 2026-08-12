s = open("piloto_realista.py").read()
s = s.replace("import numpy as np\n\nlogging.disable",
              "import numpy as np\nimport matplotlib; matplotlib.use('Agg')\n"
              "import matplotlib.pyplot as plt\nimport main\n\nlogging.disable")
s += '''

# ───────────────────────── figura ─────────────────────────
SEL = [
    ("optimo, motor ideal",            (0.0, "optimo"), "#0072B2", "-"),
    ("optimo, motor tau=1.0 s",        (1.0, "optimo"), "#0072B2", "--"),
    ("CAA: de del DP + 1 s, motor ideal", (0.0, "caa"),  "#D55E00", "-"),
    ("FAA: de del DP + 1 s, motor ideal", (0.0, "faa"),  "#009E73", "-"),
]
SEL = [(n, k, c, l) for (n, k, c, l) in SEL if k in guardar]

PAN = [("gamma", r"$\\gamma$ (deg)", np.rad2deg),
       ("v_norm", r"$V/V_s$", lambda x: x),
       ("alpha", r"$\\alpha$ (deg)", np.rad2deg),
       ("q", r"$q$ (deg/s)", np.rad2deg),
       ("de", r"$\\delta_e$ (deg)", np.rad2deg),
       ("dt", r"$\\delta_t$", lambda x: x),
       ("h", "altura (m)", lambda x: x)]

fig, axes = plt.subplots(len(PAN), 1, figsize=(7.4, 12.2), sharex=True)
for ax, (k, et, cv) in zip(axes, PAN):
    for nom, key, col, ls in SEL:
        h = guardar[key]["hist"]
        if k == "dt":
            ax.plot(h["t"], h["dt_cmd"], lw=1.0, ls=":", color=col, alpha=0.55,
                    zorder=2)
            ax.plot(h["t"], h["dt_ef"], lw=1.4, ls=ls, color=col, label=nom,
                    zorder=3)
        else:
            ax.plot(h["t"], cv(np.asarray(h[k])), lw=1.4, ls=ls, color=col,
                    label=nom, zorder=3)
    if k == "alpha":
        ax.axhline(14.0, color="0.5", lw=0.8, ls="--", zorder=1)
        ax.axhline(40.0, color="#b2182b", lw=0.8, ls="--", zorder=1)
        ax.annotate("crash: 40", xy=(0.99, 40.0),
                    xycoords=("axes fraction", "data"), ha="right", va="top",
                    fontsize=7, color="#b2182b")
    if k == "dt":
        ax.set_ylim(-0.02, 1.05)
        ax.annotate("punteado = comandado, lleno = efectivo (motor)",
                    xy=(0.99, 0.32), xycoords=("axes fraction", "data"),
                    ha="right", fontsize=7.5, color="0.35")
    if k in ("gamma", "h"):
        ax.axhline(0.0, color="0.85", lw=0.6, zorder=0)
    ax.set_ylabel(et, fontsize=9); ax.grid(True, color="0.92", lw=0.5)
    ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"): ax.spines[sp].set_color("0.6")
    ax.tick_params(labelsize=8, color="0.6")
axes[-1].set_xlabel("tiempo (s)", fontsize=9)
axes[0].legend(loc="lower left", fontsize=7.5, frameon=False, ncol=2,
               bbox_to_anchor=(0.0, 1.02))
fig.suptitle(r"Pilot with %.1f s delay and first-order engine — $V_0=%.2f\\,V_s$"
             % (TAU_H, V0), fontsize=10, y=0.988)
fig.tight_layout(rect=[0, 0, 1, 0.955])
out = main.RESULTS_DIR / ("piloto_realista_v%03d.png" % round(V0 * 100))
fig.savefig(out, dpi=200)
print("-> %s" % out)
'''
open("piloto_realista.py", "w").write(s)
print("listo")
