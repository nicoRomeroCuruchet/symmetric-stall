"""Verifica que todo artefacto que main.tex usa este al dia.

Dos chequeos, por dos fallas distintas:

  1. ARTEFACTO vs POLITICA. Lee main.tex, extrae cada \\includegraphics y cada
     \\input, y compara la fecha contra results/SymmetricStall_policy.npz. Lo
     que aparezca como VIEJO no refleja la politica instalada.

  2. FIGURA vs SU JSON. El chequeo 1 no alcanza. El pipeline escribe algunos
     json DESPUES de dibujar las figuras que los consumen: paper_procedures
     agrega el brazo e3d_held_pull a procedures.json en un paso posterior al
     que genera fig_pilot_sensitivity. Una figura puede entonces ser mas nueva
     que la politica -- y pasar el chequeo 1 -- y aun asi haberse dibujado sin
     datos que su json ya tiene. Eso paso de verdad: la curva de lazo abierto
     (elevador clavado) desaparecio de la figura 12 sin que nada lo detectara,
     porque `report.get("e3d_held_pull", {})` devolvio vacio y el bloque se
     salteo en silencio.

El mapa de dependencias es explicito y AUTOVERIFICADO: si aparece un json en
results/paper que nadie declara, o una figura sin entrada, el script lo
reporta. Un mapa que envejece en silencio es exactamente el modo de falla que
esto intenta cerrar.

    python verificar_paper.py
"""
import re
import sys
from pathlib import Path

PAPER = Path("stall-paper")
TEX = PAPER / "main.tex"
POLICY = Path("results/SymmetricStall_policy.npz")
GEN = Path("results/paper")          # donde se GENERAN, antes de sync_paper.sh
TOLERANCIA_S = 10.0                  # ver verificar_figuras_vs_json

# figura/tabla -> json del que se deriva. La figura tiene que ser mas nueva.
DEPENDE_DE = {
    "fig_pilot_sensitivity":      "procedures.json",
    "fig_procedures":             "procedures.json",
    "fig_ic_optimum":             "ic_gamma_alpha.json",
    "fig_ic_procedures":          "ic_gamma_alpha.json",
    "fig_robustness_matrix":      "robustness.json",
    "table_caa_vs_faa":           "procedures.json",
    "table_maneuvers":            "maneuvers.json",
    "table_montecarlo":           "ic_montecarlo.json",
    "table_robustness_cg_gap":    "robustness.json",
}

# artefactos que no derivan de ningun json (se calculan al vuelo desde la
# politica y el modelo, o son de otro caso de estudio)
SIN_JSON = {
    "fig_trajectories_procedures",   # corre los rollouts en el momento
    "riley_coefficients",            # sale de las tablas del modelo
    "riley_symmetric_stall_heatmaps",
    "banked_glider_", "profiling_table_", "combined_alt_loss_contours",
}

# json que ningun artefacto del paper consume directamente: alimentan numeros
# citados en el texto, no figuras ni tablas
JSON_SOLO_TEXTO = {
    "robustness_steady_state.json",  # gamma_ss y tasas de hundimiento
    "robustness_feasibility.json",   # V*
    "normalization_gap.json",        # ahorro monotono 1.97/3.75/5.34
    "thrust_sensitivity.json",
    "mca_comparison.json",
    "ic_heatmap_dense.json",
    "ic_cuts.json",
}


def main():
    if not POLICY.exists():
        print(f"no existe {POLICY}")
        return 1
    t_pol = POLICY.stat().st_mtime
    texto = TEX.read_text()

    figuras = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", texto)
    inputs = re.findall(r"\\input\{([^}]+)\}", texto)

    objetivos = []
    for f in figuras:
        objetivos.append(("figura", PAPER / f))
    for i in inputs:
        p = PAPER / i
        objetivos.append(("tabla", p if p.suffix else p.with_suffix(".tex")))

    viejos, faltantes, ok = [], [], []
    for tipo, p in objetivos:
        if not p.exists():
            faltantes.append((tipo, p))
        elif p.stat().st_mtime < t_pol:
            viejos.append((tipo, p))
        else:
            ok.append((tipo, p))

    print(f"politica instalada: {POLICY}")
    print(f"  mtime = {t_pol:.0f}\n")
    print(f"AL DIA   : {len(ok)}")
    print(f"VIEJOS   : {len(viejos)}   <- NO reflejan la politica instalada")
    print(f"FALTANTES: {len(faltantes)}\n")

    for etiqueta, lista in (("VIEJOS", viejos), ("FALTANTES", faltantes)):
        if lista:
            print(f"--- {etiqueta} ---")
            for tipo, p in sorted(lista, key=lambda x: str(x[1])):
                print(f"  [{tipo}] {p}")
            print()

    # las figuras del banked glider son de otro modelo: no dependen de esta
    # politica y que aparezcan como viejas es correcto
    # el banked glider y el contorno combinado son del benchmark 3-DOF de
    # Bunge2018: otro modelo, no dependen de esta politica
    AJENOS = ("banked_glider", "profiling_table", "combined_alt_loss_contours")
    otros = [p for _, p in viejos if any(x in p.name for x in AJENOS)]
    if otros:
        print(f"nota: {len(otros)} de los VIEJOS son del banked glider / "
              f"profiling, que no dependen de esta politica.")

    pendientes = [p for _, p in viejos
                  if not any(x in p.name for x in AJENOS)]

    rancias, sin_mapear = verificar_figuras_vs_json(objetivos)

    print(f"\nPENDIENTES REALES: {len(pendientes)}")
    for p in sorted(pendientes, key=str):
        print(f"  {p}")
    return 1 if (faltantes or pendientes or rancias or sin_mapear) else 0


def verificar_figuras_vs_json(objetivos):
    """Chequeo 2: cada figura mas nueva que el json del que se deriva.

    Compara en results/paper (donde se generan), no en stall-paper/img: la
    copia se hace con `cp -p`, asi que conserva el mtime del original, pero el
    original es el que manda.
    """
    stems = {p.stem for _, p in objetivos}
    rancias, sin_mapear = [], []

    for stem in sorted(stems):
        if any(stem.startswith(x) or x in stem for x in SIN_JSON):
            continue
        js = DEPENDE_DE.get(stem)
        if js is None:
            sin_mapear.append(stem)
            continue
        fig = next((GEN / f"{stem}{e}" for e in (".png", ".pdf", ".tex")
                    if (GEN / f"{stem}{e}").exists()), None)
        j = GEN / js
        if fig is None or not j.exists():
            continue
        # Tolerancia: varios artefactos se escriben en la MISMA corrida que su
        # json (main_maneuvers escribe tabla y json seguidos), y el orden
        # dentro de la corrida da diferencias de milisegundos que no son un
        # problema. Lo que interesa es haber sido dibujada en una corrida
        # ANTERIOR, que son minutos u horas.
        atraso = j.stat().st_mtime - fig.stat().st_mtime
        if atraso > TOLERANCIA_S:
            rancias.append((stem, js, atraso))

    # el mapa se autoverifica: un json nuevo que nadie declara es un agujero
    declarados = set(DEPENDE_DE.values()) | JSON_SOLO_TEXTO
    huerfanos = sorted(p.name for p in GEN.glob("*.json")
                       if p.name not in declarados)

    print("\n--- figura vs su json ---")
    if rancias:
        print(f"RANCIAS: {len(rancias)}  <- dibujadas ANTES que sus datos")
        for stem, js, dt in rancias:
            print(f"  {stem}  es {dt:.0f} s mas vieja que {js}")
    else:
        print("RANCIAS: 0")
    if sin_mapear:
        print(f"SIN MAPEAR: {len(sin_mapear)}  <- agregalas a DEPENDE_DE "
              f"o a SIN_JSON")
        for s in sin_mapear:
            print(f"  {s}")
    if huerfanos:
        print(f"JSON NO DECLARADOS: {len(huerfanos)}  <- agregalos a "
              f"DEPENDE_DE o a JSON_SOLO_TEXTO")
        for h in huerfanos:
            print(f"  {h}")
    return rancias, sin_mapear


if __name__ == "__main__":
    sys.exit(main())
