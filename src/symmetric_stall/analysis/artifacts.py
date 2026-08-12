"""
artifacts.py
------------
Discretization-artifact metrics for the 4-DOF Policy Iteration control maps,
ported from the 3DOF branch (commit 015fcf9) and generalized to N dimensions.

Motivation (see optimal_control.md on the 3DOF branch): a fixed timestep that
is mis-calibrated relative to the grid produces spurious "islands" of
contradictory control in the optimal policy — self-transition stalling where
the step is too small, cost collisions near switching surfaces. These metrics
quantify that fragmentation so timestep schemes can be compared A/B on the
same grid.

All metrics operate on the DISCRETE policy (the raw `policy` action-index
array reshaped to the grid), NOT on interpolated actions, because
interpolation smooths exactly the artifact we want to measure.

Grid axis order (confirmed against the solver): 0=gamma, 1=V/Vs, 2=alpha, 3=q.
Action layout: index = de_i * n_throttle + throttle_i (de-major meshgrid).
"""

import numpy as np


def policy_control_grids(pi):
    """
    Map the flat action-index policy to N-D control maps.

    Returns
    -------
    idx_grid : (*grid_shape) int   — action index per state
    de_grid  : (*grid_shape) float — optimal elevator deflection in deg
    thr_grid : (*grid_shape) float — optimal throttle in [0, 1]
    """
    shape = tuple(int(x) for x in pi.grid_shape)
    idx_grid = np.asarray(pi.policy).reshape(shape)
    actions = np.asarray(pi.action_space)
    de_grid = np.rad2deg(actions[idx_grid, 0]).astype(np.float64)
    thr_grid = actions[idx_grid, 1].astype(np.float64)
    return idx_grid, de_grid, thr_grid


def tv_norm(grid):
    """
    Anisotropic total-variation norm: mean |Δ| between face-adjacent cells,
    averaged over every adjacent pair on every axis. Higher = more fragmented.
    Dimension-agnostic.
    """
    grid = np.asarray(grid, dtype=np.float64)
    total = 0.0
    pairs = 0
    for ax in range(grid.ndim):
        d = np.abs(np.diff(grid, axis=ax))
        total += float(d.sum())
        pairs += d.size
    return total / pairs if pairs else 0.0


def island_count(idx_grid):
    """
    Count interior cells whose value differs from at least 2·ndim − 1 of
    their 2·ndim face-neighbors — isolated "islands" of contradictory
    control. Only cells with a full face-neighborhood are considered, so
    boundary effects are excluded. (ndim=4 → differs from ≥7 of 8.)

    Returns (count, fraction_of_all_cells).
    """
    idx = np.asarray(idx_grid)
    n_neighbors = 2 * idx.ndim
    differing = np.zeros(idx.shape, dtype=np.int32)
    valid = np.zeros(idx.shape, dtype=np.int32)

    for ax in range(idx.ndim):
        for shift in (-1, 1):
            neighbor = np.roll(idx, shift, axis=ax)
            # The rolled-in edge slice has no real neighbor on that side.
            edge = [slice(None)] * idx.ndim
            edge[ax] = 0 if shift == 1 else -1
            edge = tuple(edge)

            mism = (neighbor != idx)
            mism[edge] = False
            differing += mism.astype(np.int32)

            has_neighbor = np.ones(idx.shape, dtype=np.int32)
            has_neighbor[edge] = 0
            valid += has_neighbor

    island = (differing >= n_neighbors - 1) & (valid == n_neighbors)
    count = int(island.sum())
    return count, count / idx.size


def policy_metrics(pi):
    """
    Full artifact fingerprint of a trained policy: global TV of the δe* and
    δt* maps, island count/fraction on the joint action index, and
    per-control-component islands (the two controls respond differently to
    the timestep scheme — 3DOF lesson).
    """
    idx_grid, de_grid, thr_grid = policy_control_grids(pi)

    metrics = {
        "n_states": int(idx_grid.size),
        "TV_de": tv_norm(de_grid),
        "TV_thr": tv_norm(thr_grid),
    }
    count, frac = island_count(idx_grid)
    metrics["island_count"] = count
    metrics["island_frac"] = frac

    # Decompose the joint action index (de-major: index = de_i*n_thr + thr_i)
    # so each control's map is judged separately.
    n_thr = int(np.unique(np.asarray(pi.action_space)[:, 1]).size)
    de_label = idx_grid // n_thr
    thr_label = idx_grid % n_thr
    metrics["island_de"], metrics["island_de_frac"] = island_count(de_label)
    metrics["island_thr"], metrics["island_thr_frac"] = island_count(thr_label)

    return metrics
