s = open("fig_paper.py").read()
# 1. las anotaciones de los instantes van dentro del panel, no sobre el (a)
s = s.replace('''    axs[0].annotate(r"$t_{det}$", xy=(E.T_DP, 1.02), xycoords=("data", "axes fraction"),
                    fontsize=10, ha="center", va="bottom", color="0.35")
    axs[0].annotate(r"$t_{pilot}$", xy=(E.T_PIL, 1.02), xycoords=("data", "axes fraction"),
                    fontsize=10, ha="center", va="bottom", color="0.35")''',
'''    axs[0].annotate(r"$t_{det}$", xy=(E.T_DP, 0.06), xycoords=("data", "axes fraction"),
                    fontsize=9, ha="right", va="bottom", color="0.35",
                    xytext=(-3, 0), textcoords="offset points")
    axs[0].annotate(r"$t_{pilot}$", xy=(E.T_PIL, 0.06), xycoords=("data", "axes fraction"),
                    fontsize=9, ha="left", va="bottom", color="0.35",
                    xytext=(3, 0), textcoords="offset points")''')
# 2. solo el panel de abajo lleva xlabel; los de arriba lo tenian tapado
s = s.replace('''    for ax in (axs[4], axs[5], axs[6]):
        ax.set_xlabel("$t$ (s)")''',
'''    axs[6].set_xlabel("$t$ (s)")''')
# 3. FAA mas gruesa por debajo, para que se vea que comparten el elevador
s = s.replace('''            if estilo == "step":
                ax.step(t, y, color=col, lw=1.6, ls=ls, where="post", label=lab)''',
'''            if estilo == "step":
                ancho = 2.6 if lab.startswith("FAA") else 1.6
                ax.step(t, y, color=col, lw=ancho, ls=ls, where="post",
                        label=lab, alpha=0.9 if ancho > 2 else 1.0,
                        zorder=2 if ancho > 2 else 3)''')
open("fig_paper.py", "w").write(s)
print("layout corregido")
