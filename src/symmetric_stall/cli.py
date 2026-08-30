"""cli.py — training entry point (`symstall-train`).

Why the CLI lives here and not in train.py: `aircraft/grumman.py` reads
CG_AFT_M / CG_RIGHT_M / CG_BELOW_M at class level, i.e. AT IMPORT TIME. If
argument parsing happened inside train.py, the plant would already have been
imported with the old CG and the flags would have no effect. Here we parse,
set the environment via runconfig.apply(), and only THEN import train.
"""
from __future__ import annotations

import argparse

from symmetric_stall import runconfig


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="symstall-train",
        description="Train (or load) the stall-recovery policy by GPU policy "
                    "iteration.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  symstall-train                      # riley grid, riley thrust (the paper's)\n"
            "  symstall-train --grid paper1        # paper-1 grid, riley thrust\n"
            "  symstall-train --thrust paper1      # paper-1 thrust (K_t*delta_t)\n"
        ),
    )
    p.add_argument(
        "--grid", choices=("riley", "paper1"), default="riley",
        help="State-grid preset. riley = 56x81x80x41 (14,878,080 states, "
             "~4h18 on a 3090), the paper's. paper1 = 56x41x60x41 (5,648,160 "
             "states, ~43 min). Default: %(default)s.",
    )
    p.add_argument(
        "--thrust", choices=runconfig.THRUST_MODELS, default=runconfig.DEFAULT_THRUST,
        help="Thrust model, for the plant AND the CUDA kernel. NOTE: the bare "
             "code default is 'paper1'; here the default is '%(default)s' "
             "because that is the paper's.",
    )
    p.add_argument(
        "--mass-factor", type=float, default=1.0,
        help="Multiplier on Riley's published mass (715.3152 kg). Perturbs the "
             "aircraft the policy is SOLVED FOR: the stall speed and the "
             "throttle calibration follow, and the CUDA kernel is compiled "
             "with it. The .npz name carries it (massNNN), so a retrained "
             "policy can never overwrite the nominal one. Default: %(default)s.",
    )
    p.add_argument("--cg-aft", type=float, default=0.0,
                   help="CG offset towards the tail [m]. Default: %(default)s.")
    p.add_argument("--cg-right", type=float, default=0.0,
                   help="CG offset towards the right wing [m]. Default: %(default)s.")
    p.add_argument("--cg-below", type=float, default=0.0,
                   help="CG offset downwards [m]. Default: %(default)s.")
    p.add_argument(
        "--out", type=str, default=None,
        help="Output .npz path. By default it is built automatically under "
             "data/policies/, encoding grid, thrust model and CG.",
    )
    p.add_argument(
        "--greedy-eval", action="store_true",
        help="Use Approach B (argmax over discrete actions, re-derived "
             "from V at each control step) instead of Approach A "
             "(convex blend of corner actions).",
    )
    p.add_argument(
        "--compare", action="store_true",
        help="Run both A and B back-to-back and print altitude-loss delta.",
    )
    p.add_argument(
        "--warm-start-ppo", type=str, default=None,
        help="Path to a trained PPO best_model.zip — seeds PI's discrete "
             "policy from the PPO actor (snap continuous → nearest grid "
             "action). Only used if no cached policy.npz exists.",
    )
    p.add_argument(
        "--actuator-tau", type=float, default=0.0,
        help="EMA filter time constant (s) applied to PI policy output. "
             "0.0 = disabled (raw passthrough). Larger τ = more smoothing. "
             "Models actuator bandwidth — does NOT change the PI optimum.",
    )
    p.add_argument(
        "--mca", action="store_true",
        help="Train/load with the state-driven MCA endogenous timestep "
             "instead of the fixed macro dt (uses a separate policy cache "
             "and output prefix; see PolicyIterationStallConfig).",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    # BEFORE importing the plant. See the module docstring.
    runconfig.apply(
        thrust=args.thrust,
        mass_factor=args.mass_factor,
        cg_aft=args.cg_aft,
        cg_right=args.cg_right,
        cg_below=args.cg_below,
    )

    from symmetric_stall import train  # noqa: E402  (deferred import on purpose)

    train.run(args)


if __name__ == "__main__":  # pragma: no cover
    main()
