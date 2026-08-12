"""Diff table: every published number against its new value.

Compares the json files in results/paper/ (freshly regenerated with the
MCA + drag-fix + filled policy) against those committed at HEAD,
guardados previamente en PUB.
"""
import json
from pathlib import Path

PUB = Path("/tmp/claude-1000/-home-nromero-dev-research-stall-spin-recovery-dp/publicado")
NEW = Path("results/paper")


def carga(nombre):
    p, n = PUB / nombre, NEW / nombre
    a = json.loads(p.read_text()) if p.exists() else None
    b = json.loads(n.read_text()) if n.exists() else None
    return a, b


def fila(etiqueta, a, b, unidad="m"):
    if a is None or b is None:
        return f"{etiqueta:<42s}{'—':>11}{'—':>11}{'(falta)':>11}"
    d = b - a
    pct = (d / abs(a) * 100) if a else float("nan")
    return f"{etiqueta:<42s}{a:11.3f}{b:11.3f}{d:+9.3f} {pct:+6.1f}%"


def walk(pa, pb, path, out, depth=0):
    """Walk two dicts in parallel and emit the numeric leaves."""
    if depth > 3:
        return
    for k in pa:
        if k not in pb:
            continue
        va, vb = pa[k], pb[k]
        if isinstance(va, dict) and isinstance(vb, dict):
            walk(va, vb, f"{path}.{k}" if path else k, out, depth + 1)
        elif isinstance(va, (int, float)) and isinstance(vb, (int, float)) \
                and not isinstance(va, bool):
            out.append(((f"{path}.{k}" if path else k), va, vb))


def seccion(titulo, nombre, filtro=None):
    a, b = carga(nombre)
    print(f"\n{'='*80}\n{titulo}  ({nombre})\n{'='*80}")
    if a is None or b is None:
        print(f"  (falta: publicado={a is not None} nuevo={b is not None})")
        return
    hojas = []
    walk(a, b, "", hojas)
    if filtro:
        hojas = [h for h in hojas if filtro(h[0])]
    if not hojas:
        print("  (sin hojas numericas comparables)")
        return
    print(f"{'metrica':<42s}{'publicado':>11}{'nuevo':>11}{'cambio':>17}")
    cambiados = 0
    for path, va, vb in hojas:
        if abs(vb - va) > 1e-9:
            cambiados += 1
            print(fila(path, va, vb))
    print(f"\n  {cambiados} de {len(hojas)} valores cambiaron")


if __name__ == "__main__":
    print("DIFF: published paper  vs  regenerated with MCA + drag fix + filled")
    seccion("Procedimientos y sensibilidad del piloto", "procedures.json")
    seccion("Maniobras CAA / FAA", "maneuvers.json")
    seccion("Comparacion de esquemas", "mca_comparison.json")
    seccion("Robustez", "robustness.json")
    seccion("Barrido de IC (gamma, alpha)", "ic_gamma_alpha.json")
