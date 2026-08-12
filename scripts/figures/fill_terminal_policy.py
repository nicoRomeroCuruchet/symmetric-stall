"""Fill the policy of terminal cells from their nearest valid neighbour.

Policy improvement returns early on terminal cells, so every cell with
gamma >= 0 keeps whatever action index the array was initialised with. On
this grid that index is (de = -25 deg, da = -15 deg, dt = 0): full pull,
full aileron, engine idle. Nothing in the dynamic program is wrong about
that -- gamma >= 0 is the absorbing success set, the problem is over there
and no policy is defined -- but the barycentric blend used to execute the
policy does not know it. Any state whose cell straddles gamma = 0 mixes
those cells in, weighted by how close it is to level.

On this grid the consequence is not subtle. The gamma axis has 24 bins over
75 deg, so the cell spanning zero runs from -1.52 to +1.74 deg. A trajectory
at gamma = -0.20 deg draws 40% of its command from cells that order the
throttle shut, exactly when completing the level-off is what remains to be
done. The unshaped 6-DOF run settles into a 0.78 m/s descent for this
reason rather than any aerodynamic one.

The fix is to give those cells the policy of the nearest state where the
problem *is* posed: the same (V, alpha, mu, p, q) at the highest gamma bin
below zero. The value function is left untouched -- it is correct there, and
it is what the recursion needs for backups that cross the boundary. Only the
policy array changes, so the dynamic program is not re-solved.

Usage:
    python3 fill_terminal_policy.py results/BankedSpin_policy.npz
"""
import shutil
import sys
from pathlib import Path

import numpy as np


def fill(npz_path: Path) -> None:
    d = dict(np.load(npz_path))
    shape = tuple(int(x) for x in d["grid_shape"])
    gb = np.linspace(float(d["bounds_low"][0]), float(d["bounds_high"][0]),
                     shape[0])
    pol = d["policy"].reshape(shape)

    terminal = gb > 1e-9
    n_term = int(terminal.sum())
    if n_term == 0:
        print("No gamma > 0 bins on this grid; nothing to fill.")
        return
    donor = int(np.where(~terminal)[0].max())

    A = d["action_space"]
    before = pol[terminal].copy()
    idx0 = int(np.bincount(before.ravel()).argmax())
    print(f"terminal gamma bins : {n_term} of {shape[0]} "
          f"({np.rad2deg(gb[terminal]).round(2)} deg)")
    print(f"cells to fill       : {before.size:,}")
    # the action vector is (de, dt) in the 4-DOF model and (de, da, dt) in
    # the 6-DOF one, so describe whatever columns are present
    names = ("de", "dt") if A.shape[1] == 2 else ("de", "da", "dt")
    desc = ", ".join(
        f"{nm} {A[idx0, c] if nm == 'dt' else np.rad2deg(A[idx0, c]):+.2f}"
        for c, nm in enumerate(names))
    print(f"currently uniform   : {100 * np.mean(before == before.flat[0]):.1f}% "
          f"at index {idx0} = {desc}")
    print(f"donor bin           : gamma = {np.rad2deg(gb[donor]):+.3f} deg")

    # Broadcast the donor slice across every terminal gamma bin. The donor is
    # the closest state in which the recovery problem is still posed, so the
    # blend now mixes commands that were actually optimized for something.
    pol[terminal] = pol[donor][None, ...]

    after = pol[terminal]
    i_dt = A.shape[1] - 1          # throttle is the last control in both models
    dt_before = A[before, i_dt]
    dt_after = A[after, i_dt]
    print(f"\nthrottle in those cells: mean {dt_before.mean():.3f} -> "
          f"{dt_after.mean():.3f}, fraction full "
          f"{np.mean(dt_before > 0.99):.3f} -> {np.mean(dt_after > 0.99):.3f}")

    backup = npz_path.with_suffix(".npz.raw")
    if not backup.exists():
        shutil.copy2(npz_path, backup)
        print(f"\noriginal preserved at {backup.name}")
    d["policy"] = pol.reshape(-1)
    np.savez(npz_path, **d)
    print(f"written: {npz_path}")


if __name__ == "__main__":
    p = Path(sys.argv[1] if len(sys.argv) > 1
             else "results/BankedSpin_policy.npz")
    fill(p)
