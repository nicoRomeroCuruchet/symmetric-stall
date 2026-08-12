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

Hence `apply()` must run BEFORE importing `symmetric_stall.train` or anything
that pulls in `aircraft.grumman`. The CLI does exactly that.

`describe()` returns the effective configuration so it can be stamped inside
the .npz — the missing piece that made trained policies anonymous.
"""
from __future__ import annotations

import os

THRUST_MODELS = ("paper1", "riley")
DEFAULT_THRUST = "riley"

#: The CODE's default (grumman.py) is "paper1"; the PAPER's is "riley".
CODE_DEFAULT_THRUST = "paper1"


def apply(
    thrust: str = DEFAULT_THRUST,
    cg_aft: float = 0.0,
    cg_right: float = 0.0,
    cg_below: float = 0.0,
) -> None:
    """Set the model's environment variables.

    Must be called BEFORE importing the plant (see the module docstring).
    """
    if thrust not in THRUST_MODELS:
        raise ValueError(
            f"unknown thrust model: {thrust!r}. Options: {list(THRUST_MODELS)}"
        )
    os.environ["THRUST_MODEL"] = thrust
    os.environ["CG_AFT_M"] = repr(float(cg_aft))
    os.environ["CG_RIGHT_M"] = repr(float(cg_right))
    os.environ["CG_BELOW_M"] = repr(float(cg_below))


def describe() -> dict[str, str]:
    """The effective configuration, as the plant and the kernel see it."""
    return {
        "thrust_model": os.environ.get("THRUST_MODEL", CODE_DEFAULT_THRUST).lower(),
        "cg_aft_m": os.environ.get("CG_AFT_M", "0.0"),
        "cg_right_m": os.environ.get("CG_RIGHT_M", "0.0"),
        "cg_below_m": os.environ.get("CG_BELOW_M", "0.0"),
    }


def slug() -> str:
    """Filename fragment identifying the model. Bare thrust tag if all default."""
    cfg = describe()
    parts = [f"thrust-{cfg['thrust_model']}"]
    cg = (float(cfg["cg_aft_m"]), float(cfg["cg_right_m"]), float(cfg["cg_below_m"]))
    if any(cg):
        parts.append("cg-{:g}_{:g}_{:g}".format(*cg))
    return "_".join(parts)


def warn_if_code_default(logger) -> None:
    """Warn when running on the paper-1 thrust model without having asked for it."""
    if "THRUST_MODEL" not in os.environ:
        logger.warning(
            "[!] THRUST_MODEL is unset: falling back to the code default (%s), "
            "NOT the paper's (riley). The numbers will not match.",
            CODE_DEFAULT_THRUST,
        )
