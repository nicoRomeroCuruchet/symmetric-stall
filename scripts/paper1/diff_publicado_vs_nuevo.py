"""Diff table: every published number against its new value.

Compares the json files in results/paper/ (freshly regenerated with the
MCA + drag-fix + filled policy) against those committed at HEAD,
guardados previamente en PUB.
"""
import json
from pathlib import Path

PUB = Path("/tmp/claude-1000/-home-nromero-dev-research-stall-spin-recovery-dp/publicado")
NEW = Path("results/paper")


def load(name):
    p, n = PUB / name, NEW / name
    a = json.loads(p.read_text()) if p.exists() else None
    b = json.loads(n.read_text()) if n.exists() else None
    return a, b


def row(label, a, b, unit="m"):
    if a is None or b is None:
        return f"{label:<42s}{'—':>11}{'—':>11}{'(missing)':>11}"
    d = b - a
    pct = (d / abs(a) * 100) if a else float("nan")
    return f"{label:<42s}{a:11.3f}{b:11.3f}{d:+9.3f} {pct:+6.1f}%"


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


def section(title, name, filt=None):
    a, b = load(name)
    print(f"\n{'='*80}\n{title}  ({name})\n{'='*80}")
    if a is None or b is None:
        print(f"  (missing: published={a is not None} new={b is not None})")
        return
    leaves = []
    walk(a, b, "", leaves)
    if filt:
        leaves = [h for h in leaves if filt(h[0])]
    if not leaves:
        print("  (sin leaves numericas comparables)")
        return
    print(f"{'metrica':<42s}{'publicado':>11}{'nuevo':>11}{'cambio':>17}")
    changed = 0
    for path, va, vb in leaves:
        if abs(vb - va) > 1e-9:
            changed += 1
            print(row(path, va, vb))
    print(f"\n  {changed} of {len(leaves)} values changed")


if __name__ == "__main__":
    print("DIFF: published paper  vs  regenerated with MCA + drag fix + filled")
    section("Procedures and pilot sensitivity", "procedures.json")
    section("Maniobras CAA / FAA", "maneuvers.json")
    section("Comparacion de esquemas", "mca_comparison.json")
    section("Robustez", "robustness.json")
    section("Barrido de IC (gamma, alpha)", "ic_gamma_alpha.json")
