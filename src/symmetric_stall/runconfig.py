"""runconfig.py — the configuration that used to travel hidden in the environment.

Two model parameters do not enter as arguments but as environment variables,
and both the Python plant and the CUDA kernel read them:

    THRUST_MODEL   read by grumman.py:_thrust() AND by policy_iteration.py
                   (the kernel's `#define THRUST_RILEY`). Defaults to "paper1"
                   -- run without setting it and the numbers are NOT the
                   paper's (the canonical trajectory gives -13.3 m instead of
                   -6.8).

    CG_AFT_M       centre-of-gravity offset [m]. grumman.py reads these AT
    CG_RIGHT_M     CLASS LEVEL, i.e. AT IMPORT TIME: setting them after the
    CG_BELOW_M     plant has been imported has no effect whatsoever.

    MASS_FACTOR    multiplies Riley's 715.3152 kg. Read at import time by the
                   plant and compiled into the CUDA kernel, so it perturbs the
                   aircraft the policy is SOLVED FOR, not merely the one it is
                   flown on. That is the difference from the THRUST_SCALE and
                   the `env.airplane.MASS *= f` of paper_robustness.py, which
                   are evaluation-only. Everything grumman.py derives from the
                   mass in __init__ follows: the stall speed, hence the V/Vs
                   grid axis, hence the throttle calibration. Use it to answer
                   "would a policy retrained for this mass do better?"; use the
                   evaluation-only knobs to answer "how does the nominal policy
                   degrade?". They are different questions.

    ELEVATOR_TAU_S elevator-channel time constant [s]. Same role as
                   ENGINE_TAU_S and read the same way, per rollout, but it is
                   NOT one of Riley's constants -- he models no elevator
                   dynamics. See engine.DEFAULT_ELEVATOR_TAU before quoting a
                   value. It tags filenames as `de###`, deliberately NOT as
                   `tau###`: that tag already means the engine, and two
                   channels sharing it would make `..._tau085.png` ambiguous.

    ENGINE_TAU_S   Riley's engine-response time constant [s], eq. (A4). Unlike
                   the three above it is read per ROLLOUT rather than at import,
                   because it belongs to the evaluation and not to the solved
                   plant: the policy is always solved against an ideal engine.
                   0 disables the lag. See engine.py.

Hence `apply()` must run BEFORE importing `symmetric_stall.train` or anything
that pulls in `aircraft.grumman`. The CLI does exactly that.

`describe()` returns the effective configuration so it can be stamped inside
the .npz — the missing piece that made trained policies anonymous — and inside
every result JSON, so no artifact is anonymous about which engine produced it.
"""
from __future__ import annotations

import os

THRUST_MODELS = ("paper1", "riley")
DEFAULT_THRUST = "riley"

#: The CODE's default (grumman.py) is "paper1"; the PAPER's is "riley".
CODE_DEFAULT_THRUST = "paper1"

#: Engine lag off unless asked for. Every result published before the constant
#: was identified was produced this way, so the default has to keep reproducing
#: them exactly rather than silently switching plant underneath old commands.
CODE_DEFAULT_ENGINE_TAU = 0.0

#: Elevator lag off unless asked for, for the same reason as the engine: every
#: published result was produced with an instantaneous elevator.
CODE_DEFAULT_ELEVATOR_TAU = 0.0

#: Riley's aircraft at its published mass. Anything else is a perturbation and
#: has to say so in the filename -- see slug().
CODE_DEFAULT_MASS_FACTOR = 1.0


def apply(
    thrust: str = DEFAULT_THRUST,
    cg_aft: float = 0.0,
    cg_right: float = 0.0,
    cg_below: float = 0.0,
    engine_tau: float = CODE_DEFAULT_ENGINE_TAU,
    mass_factor: float = CODE_DEFAULT_MASS_FACTOR,
    elevator_tau: float = CODE_DEFAULT_ELEVATOR_TAU,
) -> None:
    """Set the model's environment variables.

    Must be called BEFORE importing the plant (see the module docstring).
    `engine_tau` is the exception: it is read per rollout, so it may be set
    later, but it is routed through here so one call configures the whole run.
    """
    if thrust not in THRUST_MODELS:
        raise ValueError(
            f"unknown thrust model: {thrust!r}. Options: {list(THRUST_MODELS)}"
        )
    if float(engine_tau) < 0.0:
        raise ValueError(f"engine_tau must be >= 0, got {engine_tau!r}")
    if float(elevator_tau) < 0.0:
        raise ValueError(f"elevator_tau must be >= 0, got {elevator_tau!r}")
    if float(mass_factor) <= 0.0:
        raise ValueError(f"mass_factor must be > 0, got {mass_factor!r}")
    os.environ["THRUST_MODEL"] = thrust
    os.environ["CG_AFT_M"] = repr(float(cg_aft))
    os.environ["CG_RIGHT_M"] = repr(float(cg_right))
    os.environ["CG_BELOW_M"] = repr(float(cg_below))
    os.environ["ENGINE_TAU_S"] = repr(float(engine_tau))
    os.environ["MASS_FACTOR"] = repr(float(mass_factor))
    os.environ["ELEVATOR_TAU_S"] = repr(float(elevator_tau))


def engine_tau() -> float:
    """Riley's tau_e for THIS run, in seconds. 0 means the ideal engine."""
    return float(os.environ.get("ENGINE_TAU_S", CODE_DEFAULT_ENGINE_TAU))


def elevator_tau() -> float:
    """Elevator time constant for THIS run, in seconds. 0 means instantaneous."""
    return float(os.environ.get("ELEVATOR_TAU_S", CODE_DEFAULT_ELEVATOR_TAU))


def mass_factor() -> float:
    """Multiplier on Riley's published mass for THIS run. 1.0 is his aircraft."""
    return float(os.environ.get("MASS_FACTOR", CODE_DEFAULT_MASS_FACTOR))


def describe() -> dict[str, str]:
    """The effective configuration, as the plant and the kernel see it."""
    return {
        "thrust_model": os.environ.get("THRUST_MODEL", CODE_DEFAULT_THRUST).lower(),
        "cg_aft_m": os.environ.get("CG_AFT_M", "0.0"),
        "cg_right_m": os.environ.get("CG_RIGHT_M", "0.0"),
        "cg_below_m": os.environ.get("CG_BELOW_M", "0.0"),
        "engine_tau_s": repr(engine_tau()),
        "mass_factor": repr(mass_factor()),
        "elevator_tau_s": repr(elevator_tau()),
    }


def engine_label() -> str:
    """One phrase naming BOTH lagged channels, for logs and figure stamps.

    Both are named even when one is off, because "which plant flew this?" is
    the question the stamp exists to answer, and silence about a channel reads
    as "not modelled" rather than "ideal".
    """
    tau = engine_tau()
    eng = f"Riley (A4) lag, tau_e = {tau:g} s" if tau > 0.0 else "ideal engine (no lag)"
    de = elevator_tau()
    elev = f"elevator tau = {de:g} s" if de > 0.0 else "instantaneous elevator"
    return f"{eng} | {elev}"


def slug() -> str:
    """Filename fragment identifying the model. Bare thrust tag if all default."""
    cfg = describe()
    parts = [f"thrust-{cfg['thrust_model']}"]
    cg = (float(cfg["cg_aft_m"]), float(cfg["cg_right_m"]), float(cfg["cg_below_m"]))
    if any(cg):
        parts.append("cg-{:g}_{:g}_{:g}".format(*cg))
    if engine_tau() > 0.0:
        parts.append(f"tau{round(engine_tau() * 100):03d}")
    if mass_factor() != CODE_DEFAULT_MASS_FACTOR:
        # Without this the retrained policies of the mass study would all land
        # on the reference policy's filename, and train.py would either load it
        # instead of training (silently answering the wrong question) or
        # overwrite four hours of GPU with a perturbed aircraft.
        parts.append(f"mass{round(mass_factor() * 100):03d}")
    if elevator_tau() > 0.0:
        parts.append(f"de{round(elevator_tau() * 100):03d}")
    return "_".join(parts)


def warn_if_code_default(logger) -> None:
    """Warn when running on the paper-1 thrust model without having asked for it."""
    if "THRUST_MODEL" not in os.environ:
        logger.warning(
            "[!] THRUST_MODEL is unset: falling back to the code default (%s), "
            "NOT the paper's (riley). The numbers will not match.",
            CODE_DEFAULT_THRUST,
        )
