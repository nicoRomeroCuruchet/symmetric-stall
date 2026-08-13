"""paths.py — where the experiments find the trained policy and put their results.

The paper-1 experiment suite hardcoded `results/SymmetricStall_policy.npz` in
eight different files, and got its thrust model from whatever happened to be in
the environment. Both are how the published numbers ended up being produced by
a propeller model nobody had chosen: `grumman.py` falls back to `paper1` when
`THRUST_MODEL` is unset, so a script run without it is silently wrong rather
than loudly broken.

`load_policy()` is the single choke point every experiment goes through, and it
refuses to load anything until the thrust model has been chosen explicitly.
"""
from __future__ import annotations

import os
from pathlib import Path

from symmetric_stall import runconfig

#: Repository root, resolved from this file (src/symmetric_stall/paths.py).
REPO_ROOT = Path(__file__).resolve().parents[2]

POLICY_DIR = REPO_ROOT / "data" / "policies"

#: Where the paper-1 experiments, re-run against the corrected model, write to.
DEFAULT_OUT_DIR = REPO_ROOT / "results" / "5_paper1_repro"


def out_dir() -> Path:
    """Output directory for the experiment artifacts. `$STALL_OUT` overrides."""
    d = Path(os.environ.get("STALL_OUT", DEFAULT_OUT_DIR))
    d.mkdir(parents=True, exist_ok=True)
    return d


def policy_path(explicit: str | Path | None = None) -> Path:
    """Resolve the trained policy to evaluate.

    In order: the argument, `$STALL_POLICY`, or — if `data/policies` holds
    exactly one `.npz` — that one. Ambiguity is an error rather than a guess:
    picking the wrong policy produces plausible numbers, which is the worst
    kind of wrong.
    """
    if explicit is not None:
        p = Path(explicit)
    elif "STALL_POLICY" in os.environ:
        p = Path(os.environ["STALL_POLICY"])
    else:
        candidates = sorted(POLICY_DIR.glob("*.npz"))
        if not candidates:
            raise FileNotFoundError(
                f"no policy in {POLICY_DIR}. Train one with `symstall-train`, "
                f"or point $STALL_POLICY at an existing .npz."
            )
        if len(candidates) > 1:
            names = "\n  ".join(c.name for c in candidates)
            raise RuntimeError(
                f"{len(candidates)} policies in {POLICY_DIR}; say which one "
                f"through $STALL_POLICY:\n  {names}"
            )
        p = candidates[0]

    if not p.exists():
        raise FileNotFoundError(p)
    return p


def assert_thrust_configured() -> str:
    """Fail unless the thrust model was chosen on purpose.

    Riley's propeller (Appendix A) and the paper-1 constant-`Kt` stand-in do
    not merely differ in magnitude: Riley's thrust decays with airspeed, so the
    two disagree most exactly where the stall lives. Defaulting silently is how
    paper 1 got its numbers.
    """
    if "THRUST_MODEL" not in os.environ:
        raise RuntimeError(
            "THRUST_MODEL is unset, so the plant would fall back to the code "
            f"default ({runconfig.CODE_DEFAULT_THRUST!r}) and NOT to the "
            "paper's model. Call runconfig.apply(thrust='riley') before "
            "importing the plant, or export THRUST_MODEL=riley."
        )
    return os.environ["THRUST_MODEL"].lower()


def load_policy(explicit: str | Path | None = None, env=None):
    """Load a trained policy, with the thrust model checked against its metadata.

    A policy trained under one propeller model and evaluated under another is
    a silent error — the rollouts converge, the figures render, and every
    number is wrong. When the `.npz` records its configuration (everything
    trained since the migration does), the mismatch is caught here.
    """
    from symmetric_stall.aircraft.symmetric_stall import SymmetricStall
    from symmetric_stall.policy_iteration import PolicyIterationStall

    thrust = assert_thrust_configured()
    path = policy_path(explicit)
    pi = PolicyIterationStall.load(path, env=env if env is not None else SymmetricStall())

    trained = getattr(pi, "run_metadata", None) or {}
    trained_thrust = trained.get("thrust_model")
    if trained_thrust is not None and str(trained_thrust).lower() != thrust:
        raise RuntimeError(
            f"{path.name} was trained with THRUST_MODEL={trained_thrust!r} but "
            f"this process is running {thrust!r}. Evaluating a policy under a "
            f"plant it was not solved for silently invalidates every number."
        )
    return pi


def ensure_states_space(pi):
    """Materialise `pi.states_space`, which `load()` deliberately leaves empty.

    The heatmap and Q-value figures index it directly. Rebuilding it costs
    `prod(grid_shape) x n_dims` floats — 476 MB on the Riley grid — so it stays
    opt-in rather than being paid by every rollout script.
    """
    import numpy as np

    if getattr(pi, "states_space", None) is not None and np.size(pi.states_space):
        return pi.states_space

    shape = tuple(int(x) for x in pi.grid_shape)
    axes = [
        np.linspace(float(pi.bounds_low[i]), float(pi.bounds_high[i]), shape[i])
        for i in range(len(shape))
    ]
    pi.states_space = np.stack(
        np.meshgrid(*axes, indexing="ij"), -1
    ).reshape(-1, len(shape))
    return pi.states_space
