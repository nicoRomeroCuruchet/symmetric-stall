"""
paper_switching_stats.py — the switching-surface statistics quoted in
Sec. V.D (Optimal Policy).

Definition: per (gamma, V) column of the converged policy at the q = 0
slice, the switching angle is the zero crossing of the commanded
elevator along alpha (linear interpolation between the two grid nodes
where the command changes sign, searched below alpha = 25 deg so the
gamma ~ 0 tie bands at high alpha do not pollute the statistic).
Columns: -40 deg <= gamma <= 0, V <= 1.2 Vs.

The lower tail of the distribution is genuine, not noise: columns
whose angle of attack is already low command little or no push, so
their crossing sits well below the stall boundary.

Usage:
    STALL_POLICY=data/policies/SymmetricStall_riley_....npz \
        python scripts/paper1/paper_switching_stats.py
"""
import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_POLICY = ("data/policies/"
                  "SymmetricStall_riley_56x81x80x41_thrust-riley.npz")


def main():
    path = os.environ.get("STALL_POLICY", DEFAULT_POLICY)
    d = np.load(path)
    P = d["policy"].reshape(d["grid_shape"])
    A = d["action_space"]
    lo, hi, shape = d["bounds_low"], d["bounds_high"], np.array(d["grid_shape"])
    gamma = np.linspace(lo[0], hi[0], shape[0])
    vn = np.linspace(lo[1], hi[1], shape[1])
    alpha = np.rad2deg(np.linspace(lo[2], hi[2], shape[2]))
    qb = np.linspace(lo[3], hi[3], shape[3])
    de_of = np.rad2deg(A[:, 0])
    q0 = int(np.argmin(np.abs(qb)))

    gm = (gamma >= np.deg2rad(-40.0)) & (gamma <= 0.0)
    vm = vn <= 1.2
    amask = alpha <= 25.0

    sw = []
    for gi in np.where(gm)[0]:
        for vi in np.where(vm)[0]:
            col = de_of[P[gi, vi, :, q0]]
            neg = np.where((col < 0.0) & amask)[0]
            if len(neg) == 0 or neg[-1] + 1 >= len(alpha):
                continue
            k = neg[-1]
            d0, d1 = col[k], col[k + 1]
            t = d0 / (d0 - d1) if d1 != d0 else 0.5
            sw.append(alpha[k] + t * (alpha[k + 1] - alpha[k]))

    sw = np.array(sw)
    logger.info("policy: %s", path)
    logger.info("columns with a crossing: %d", len(sw))
    logger.info("median = %.1f deg, p5 = %.1f, p95 = %.1f, mean = %.1f",
                np.median(sw), np.percentile(sw, 5),
                np.percentile(sw, 95), sw.mean())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
