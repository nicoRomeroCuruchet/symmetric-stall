s = open("familia.py").read()
a = '''    ax.set_ylabel(et, fontsize=9); ax.grid(True, color="0.92", lw=0.5)'''
b = '''    if k == "dt_ctrl":
        ax.set_ylim(-0.02, 1.05)          # si no, matplotlib hace zoom en el ULP
        ax.annotate("1.000 en las cuatro", xy=(0.99, 1.0),
                    xycoords=("axes fraction", "data"), ha="right", va="top",
                    fontsize=7.5, color="0.35")
    ax.set_ylabel(et, fontsize=9); ax.grid(True, color="0.92", lw=0.5)'''
assert a in s
s = s.replace(a, b)
# etiquetas de h_min: alternar el offset vertical para que no se pisen
s = s.replace('xytext=(5, -9),', 'xytext=(5, -9 - 11 * V0S.index(v0)),')
open("familia.py", "w").write(s)
print("figura corregida")
