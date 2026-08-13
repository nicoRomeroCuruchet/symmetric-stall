"""Runner for the paper-1 experiment suite against the corrected Riley model.

Paper 1's experiments were conceptually right but were solved on a plant whose
propeller was a constant-`Kt` stand-in calibrated for level flight at 2 Vs.
Riley's actual thrust (Appendix A) decays with airspeed, so the two models
disagree most exactly where the stall lives. This re-runs the same experiments,
unchanged, against the corrected plant.

The thrust model and the CG must be set BEFORE the plant is imported — the
aircraft reads them at class-body level — which is why every experiment is
dispatched from here rather than from each script's own `__main__`.

    python scripts/paper1/run.py --list
    python scripts/paper1/run.py maneuvers trajectories
    python scripts/paper1/run.py --thrust paper1 --out results/x procedures
"""
from __future__ import annotations

import argparse
import logging
import sys

from symmetric_stall import runconfig

#: name -> (module attribute path, one-line description, rough cost)
EXPERIMENTS = {
    # --- procedures.py: pilot-procedure comparison and sensitivity ---------
    "procedures":     ("procedures:main",
                       "E1/E3b/E3c: power delay, switch delay, partial pull", "min"),
    "maneuvers":      ("procedures:main_maneuvers",
                       "CAA vs FAA scripted manoeuvres at the canonical IC", "s"),
    "trajectories":   ("procedures:make_trajectory_comparison_figure",
                       "time-domain DP vs scripted CAA vs FAA", "s"),
    "held_pull":      ("procedures:compute_held_pull",
                       "open-loop held-pull sweep", "min"),
    "procedure_figs": ("procedures:_procedure_figs_cmd",
                       "redraw fig_procedures + fig_pilot_sensitivity", "s"),
    "ic_figs":        ("procedures:_ic_figs_cmd",
                       "redraw every IC-plane map from the cache", "s"),
    "ic_heatmap":     ("procedures:_ic_heatmap_cmd",
                       "dense IC-plane sweep (~600 rollouts) + maps", "min"),
    "caa_ramp":       ("procedures:_caa_ramp_cmd",
                       "realistic-CAA arm added to the dense IC map", "min"),
    "switch_heatmap": ("procedures:_switch_heatmap_cmd",
                       "dense switch-delay sweep (~350 rollouts)", "min"),
    # --- paper_robustness.py: off-nominal aircraft -------------------------
    "robustness":     ("paper_robustness:main",
                       "mass x CG degradation matrix of the nominal policy", "~10 min"),
    "rob_steady":     ("paper_robustness:characterize_steady_state",
                       "steady descent rates feeding the prose", "s"),
    "rob_feasible":   ("paper_robustness:level_flight_feasibility",
                       "level-flight feasibility and V* per mass", "s"),
    "cg_reach":       ("cg_reach:main",
                       "push the CG aft past the divergence boundary", "min"),
    # --- model-only figures (no policy) ------------------------------------
    "riley_coeffs":   ("paper_fig_riley_coeffs:main",
                       "Riley coefficient tables (model only)", "s"),
    "caa_vs_faa":     ("gen_table_caa_vs_faa:main",
                       "table_caa_vs_faa.tex from procedures.json", "s"),
}

#: experiment -> the artifact it reads, and the experiment that writes it.
#: Checked before anything runs, so a missing input is a one-line refusal
#: rather than a traceback after several minutes of rollouts.
REQUIRES = {
    "caa_vs_faa": ("procedures.json", "procedures"),
    "procedure_figs": ("procedures.json", "procedures"),
    "held_pull": ("procedures.json", "procedures"),
    "caa_ramp": ("ic_heatmap_dense.json", "ic_heatmap"),
    "ic_figs": ("ic_heatmap_dense.json", "ic_heatmap"),
    "switch_heatmap": ("ic_heatmap_dense.json", "ic_heatmap"),
}

#: Experiments that cannot run yet, and what each is waiting on.
BLOCKED = {
    "mca":        "needs an MCA-timestep policy retrained on the Riley grid",
    "dp_vs_ppo":  "needs a PPO baseline retrained (stable_baselines3 not installed)",
    "q_values":   "needs the same PPO baseline",
    "cg_sweep":   "needs 7 GPU re-solves on the Riley grid",
    "gamma_alpha": "12,000 rollouts; generator lives in attic/regen_orphans.py",
    "montecarlo": "4-D Monte Carlo; generator lives in attic/regen_orphans.py",
}


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("experiments", nargs="*", help="names to run (see --list)")
    p.add_argument("--list", action="store_true", help="show the suite and exit")
    p.add_argument("--thrust", default=runconfig.DEFAULT_THRUST,
                   choices=runconfig.THRUST_MODELS)
    p.add_argument("--cg-aft", type=float, default=0.0, metavar="M")
    p.add_argument("--cg-right", type=float, default=0.0, metavar="M")
    p.add_argument("--cg-below", type=float, default=0.0, metavar="M")
    p.add_argument("--policy", default=None,
                   help="policy .npz (default: the only one in data/policies)")
    p.add_argument("--out", default=None, help="output directory")
    return p


def show_list():
    print(f"{'experiment':<16} {'cost':<9} what it produces")
    print("-" * 78)
    for name, (_, desc, cost) in EXPERIMENTS.items():
        print(f"{name:<16} {cost:<9} {desc}")
    print()
    print("blocked:")
    for name, why in BLOCKED.items():
        print(f"  {name:<14} {why}")


def resolve(target):
    """Import `module:attr` from this directory or from the package."""
    module_name, attr = target.split(":")
    if module_name == "procedures":
        module = __import__("symmetric_stall.procedures", fromlist=[attr])
    else:
        sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
        module = __import__(module_name, fromlist=[attr])
    return getattr(module, attr)


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.list or not args.experiments:
        show_list()
        return 0

    unknown = [e for e in args.experiments if e not in EXPERIMENTS]
    if unknown:
        blocked = [e for e in unknown if e in BLOCKED]
        for e in blocked:
            print(f"error: {e!r} is blocked: {BLOCKED[e]}", file=sys.stderr)
        for e in unknown:
            if e not in BLOCKED:
                print(f"error: unknown experiment {e!r}", file=sys.stderr)
        return 2

    # Before any import that reaches the plant.
    runconfig.apply(thrust=args.thrust, cg_aft=args.cg_aft,
                    cg_right=args.cg_right, cg_below=args.cg_below)

    import os
    if args.policy:
        os.environ["STALL_POLICY"] = args.policy
    if args.out:
        os.environ["STALL_OUT"] = args.out

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s")
    log = logging.getLogger("run")

    from symmetric_stall import paths
    log.info("model: %s", runconfig.describe())
    log.info("policy: %s", paths.policy_path())
    log.info("output: %s", paths.out_dir())

    for name in args.experiments:
        need = REQUIRES.get(name)
        if need and not (paths.out_dir() / need[0]).exists():
            if need[1] in args.experiments[:args.experiments.index(name)]:
                pass                      # produced earlier in this same run
            else:
                log.error("%s needs %s; run %r first", name, need[0], need[1])
                return 1
        target, desc, _ = EXPERIMENTS[name]
        log.info("=== %s — %s", name, desc)
        resolve(target)()
        log.info("=== %s done", name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
