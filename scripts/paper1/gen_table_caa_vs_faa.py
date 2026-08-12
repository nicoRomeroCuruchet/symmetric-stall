"""Generate table_caa_vs_faa.tex from results/paper/procedures.json.

The table was in the paper but no script produced it: it was written by hand
once (commit dbd7f6f) and then left orphaned. The column mapping was verified
against the published values, cell by cell.

    Optimal  = e3b_switch_delay["0"]                 (pure policy)
    tau=0.5  = e1_power_delay.instant["0.5"]
    tau=1    = e1_power_delay.instant["1"]
    PD       = e1_power_delay.ramp2s["2"]            (Gratton power-delayed)
    gated    = e1_power_delay.gated_alpha14_ramp2s
"""
import json
from pathlib import Path

OUT = Path("results/paper")
ALPHAS = (16, 18, 20)
VNORMS = (0.90, 0.95, 1.00)

CAB = (r"\begin{table}[hbt!]", r"    \centering",
       r"    \caption{Altitude loss $\Delta h$ (m) of the optimal DP policy (CAA-like:",
       r"    simultaneous nose-down elevator + full power, then pull-up) vs.\ the same",
       r"    optimal pitch schedule with delayed power (FAA-like), across initial",
       r"    conditions ($\gamma_0=0$, $q_0=0$). PD = Gratton's power-delayed",
       r"    ($\tau=2$\,s + 2\,s ramp); gated = power withheld until $\alpha<14^\circ$.}",
       r"    \label{tab:caa_vs_faa}",
       r"    \begin{tabular}{ccccccc}",
       r"        \hline",
       r"        $\alpha_0$ & $V_0/V_s$ & Optimal & $\tau{=}0.5$ s & "
       r"$\tau{=}1$ s & PD & gated \\",
       r"        \hline")


def celda(entrada):
    """-9.09 or -2.75^t if it did not return to level flight within the horizon."""
    h = entrada["h"]
    marca = "" if entrada.get("status") == "recovered" else r"\textsuperscript{t}"
    return f"{h:.2f}{marca}"


def main():
    rep = json.loads((OUT / "procedures.json").read_text())
    cols = [
        rep["e3b_switch_delay"]["0"],
        rep["e1_power_delay"]["instant"]["0.5"],
        rep["e1_power_delay"]["instant"]["1"],
        rep["e1_power_delay"]["ramp2s"]["2"],
        rep["e1_power_delay"]["gated_alpha14_ramp2s"],
    ]
    lineas = list(CAB)
    for a in ALPHAS:
        for v in VNORMS:
            k = f"a{a}_v{v:.2f}"
            vals = " & ".join(celda(c[k]) for c in cols)
            lineas.append(f"        {a} & {v:.2f} & {vals} " + r"\\")
    lineas += [
        r"        \hline",
        r"        \multicolumn{7}{l}{\footnotesize\textsuperscript{t}\,No return "
        r"to level flight within 15 s (shallow powered descent).}\\",
        r"    \end{tabular}",
        r"\end{table}",
    ]
    (OUT / "table_caa_vs_faa.tex").write_text("\n".join(lineas) + "\n")
    print(f"[+] {OUT/'table_caa_vs_faa.tex'} escrita ({len(ALPHAS)*len(VNORMS)} filas)")


if __name__ == "__main__":
    main()
