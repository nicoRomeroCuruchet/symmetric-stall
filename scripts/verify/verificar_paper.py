"""Check that every artefact main.tex uses is up to date.

Two checks, for two different failure modes:

  1. ARTEFACT vs POLICY. Reads main.tex, extracts every \\includegraphics and
     every \\input, and compares the timestamp against the trained policy.
     Anything reported as STALE does not reflect the installed policy.

  2. FIGURE vs ITS JSON. Check 1 is not enough. The pipeline writes some json
     files AFTER drawing the figures that consume them: procedures.py adds the
     e3d_held_pull arm to procedures.json in a step later than the one that
     generates fig_pilot_sensitivity. A figure can therefore be newer than the
     policy -- and pass check 1 -- and still have been drawn without data its
     json already contains. That actually happened: the open-loop curve (held
     elevator) disappeared from figure 12 with nothing detecting it, because
     `report.get("e3d_held_pull", {})` returned empty and the block was
     silently skipped.

The dependency map is explicit and SELF-CHECKED: if a json shows up in
results/paper that nobody declares, or a figure has no entry, the script
reports it. A map that goes stale in silence is exactly the failure mode this
is meant to close.

    python verificar_paper.py
"""
import re
import sys
from pathlib import Path

PAPER = Path("stall-paper")
TEX = PAPER / "main.tex"
POLICY = Path("results/SymmetricStall_policy.npz")
GEN = Path("results/paper")          # where they are GENERATED, before sync_paper.sh
TOLERANCE_S = 10.0                   # see check_figures_vs_json

# figure/table -> the json it derives from. The figure must be newer.
DERIVES_FROM = {
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

# artefacts that derive from no json (computed on the fly from the policy and
# the model, or belonging to a different case study)
NO_JSON = {
    "fig_trajectories_procedures",   # runs the rollouts on the spot
    "riley_coefficients",            # comes from the model tables
    "riley_symmetric_stall_heatmaps",
    "banked_glider_", "profiling_table_", "combined_alt_loss_contours",
}

# json files no paper artefact consumes directly: they feed numbers quoted in
# the text, not figures or tables
JSON_TEXT_ONLY = {
    "robustness_steady_state.json",  # gamma_ss and sink rates
    "robustness_feasibility.json",   # V*
    "normalization_gap.json",        # monotone saving 1.97/3.75/5.34
    "thrust_sensitivity.json",
    "mca_comparison.json",
    "ic_heatmap_dense.json",
    "ic_cuts.json",
}


def main():
    if not POLICY.exists():
        print(f"{POLICY} does not exist")
        return 1
    t_pol = POLICY.stat().st_mtime
    text = TEX.read_text()

    figures = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", text)
    inputs = re.findall(r"\\input\{([^}]+)\}", text)

    targets = []
    for f in figures:
        targets.append(("figura", PAPER / f))
    for i in inputs:
        p = PAPER / i
        targets.append(("tabla", p if p.suffix else p.with_suffix(".tex")))

    stale, missing, ok = [], [], []
    for kind, p in targets:
        if not p.exists():
            missing.append((kind, p))
        elif p.stat().st_mtime < t_pol:
            stale.append((kind, p))
        else:
            ok.append((kind, p))

    print(f"installed policy: {POLICY}")
    print(f"  mtime = {t_pol:.0f}\n")
    print(f"UP TO DATE: {len(ok)}")
    print(f"STALE     : {len(stale)}   <- do NOT reflect the installed policy")
    print(f"MISSING   : {len(missing)}\n")

    for label, group in (("STALE", stale), ("MISSING", missing)):
        if group:
            print(f"--- {label} ---")
            for kind, p in sorted(group, key=lambda x: str(x[1])):
                print(f"  [{kind}] {p}")
            print()

    # the banked-glider figures belong to a different model: they do not
    # depend on this policy, and showing up as stale is correct. The banked
    # glider and the combined contour come from the Bunge2018 3-DOF benchmark.
    FOREIGN = ("banked_glider", "profiling_table", "combined_alt_loss_contours")
    foreign = [p for _, p in stale if any(x in p.name for x in FOREIGN)]
    if foreign:
        print(f"note: {len(foreign)} of the STALE ones are banked glider / "
              f"profiling, which do not depend on this policy.")

    pending = [p for _, p in stale
               if not any(x in p.name for x in FOREIGN)]

    outdated, unmapped = check_figures_vs_json(targets)

    print(f"\nREAL PENDING: {len(pending)}")
    for p in sorted(pendientes, key=str):
        print(f"  {p}")
    return 1 if (missing or pendientes or rancias or sin_mapear) else 0


def check_figures_vs_json(targets):
    """Check 2: every figure newer than the json it derives from.

    Compares inside results/paper (where they are generated), not in
    stall-paper/img: the copy is made with `cp -p`, so it preserves the
    original mtime, but the original is what counts.
    """
    stems = {p.stem for _, p in targets}
    outdated, unmapped = [], []

    for stem in sorted(stems):
        if any(stem.startswith(x) or x in stem for x in NO_JSON):
            continue
        js = DERIVES_FROM.get(stem)
        if js is None:
            unmapped.append(stem)
            continue
        fig = next((GEN / f"{stem}{e}" for e in (".png", ".pdf", ".tex")
                    if (GEN / f"{stem}{e}").exists()), None)
        j = GEN / js
        if fig is None or not j.exists():
            continue
        # Tolerance: several artefacts are written in the SAME run as their
        # json (main_maneuvers writes table and json back to back), and the
        # ordering within a run gives millisecond differences that are not a
        # problem. What matters is having been drawn in an EARLIER run, which
        # means minutes or hours.
        lag = j.stat().st_mtime - fig.stat().st_mtime
        if lag > TOLERANCE_S:
            outdated.append((stem, js, lag))

    # the map self-checks: a new json nobody declares is a hole
    declared = set(DERIVES_FROM.values()) | JSON_TEXT_ONLY
    orphans = sorted(p.name for p in GEN.glob("*.json")
                     if p.name not in declared)

    print("\n--- figure vs its json ---")
    if outdated:
        print(f"OUTDATED: {len(outdated)}  <- drawn BEFORE their data")
        for stem, js, dt in outdated:
            print(f"  {stem}  is {dt:.0f} s older than {js}")
    else:
        print("OUTDATED: 0")
    if unmapped:
        print(f"UNMAPPED: {len(unmapped)}  <- add them to DERIVES_FROM "
              f"or to NO_JSON")
        for s in unmapped:
            print(f"  {s}")
    if orphans:
        print(f"UNDECLARED JSON: {len(orphans)}  <- add them to "
              f"DERIVES_FROM or to JSON_TEXT_ONLY")
        for h in orphans:
            print(f"  {h}")
    return outdated, unmapped


if __name__ == "__main__":
    sys.exit(main())
