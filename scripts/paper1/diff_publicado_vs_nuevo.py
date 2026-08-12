"""Tabla de diff: cada numero publicado contra su nuevo valor.

Compara los json de results/paper/ (recien regenerados con la politica
MCA + fix de drag + filled) contra los que estaban commiteados en HEAD,
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


def recorre(pa, pb, ruta, salida, prof=0):
    """Recorre dos dicts en paralelo y emite las hojas numericas."""
    if prof > 3:
        return
    for k in pa:
        if k not in pb:
            continue
        va, vb = pa[k], pb[k]
        if isinstance(va, dict) and isinstance(vb, dict):
            recorre(va, vb, f"{ruta}.{k}" if ruta else k, salida, prof + 1)
        elif isinstance(va, (int, float)) and isinstance(vb, (int, float)) \
                and not isinstance(va, bool):
            salida.append(((f"{ruta}.{k}" if ruta else k), va, vb))


def seccion(titulo, nombre, filtro=None):
    a, b = carga(nombre)
    print(f"\n{'='*80}\n{titulo}  ({nombre})\n{'='*80}")
    if a is None or b is None:
        print(f"  (falta: publicado={a is not None} nuevo={b is not None})")
        return
    hojas = []
    recorre(a, b, "", hojas)
    if filtro:
        hojas = [h for h in hojas if filtro(h[0])]
    if not hojas:
        print("  (sin hojas numericas comparables)")
        return
    print(f"{'metrica':<42s}{'publicado':>11}{'nuevo':>11}{'cambio':>17}")
    cambiados = 0
    for ruta, va, vb in hojas:
        if abs(vb - va) > 1e-9:
            cambiados += 1
            print(fila(ruta, va, vb))
    print(f"\n  {cambiados} de {len(hojas)} valores cambiaron")


if __name__ == "__main__":
    print("DIFF: paper publicado  vs  regenerado con MCA + fix de drag + filled")
    seccion("Procedimientos y sensibilidad del piloto", "procedures.json")
    seccion("Maniobras CAA / FAA", "maneuvers.json")
    seccion("Comparacion de esquemas", "mca_comparison.json")
    seccion("Robustez", "robustness.json")
    seccion("Barrido de IC (gamma, alpha)", "ic_gamma_alpha.json")
