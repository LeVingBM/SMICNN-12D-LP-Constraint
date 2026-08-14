#!/usr/bin/env python
"""Generate multi-source 12-D lamination-parameter boundary data.

Each unit ray is intersected with the discretized feasible region by an exact
epigraph linear program.  The output stores the boundary point, radial value,
and the optimal polar normal required for exact affine supports.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


SOURCE_NAMES = np.asarray(
    [
        "dense_random",
        "sparse_random",
        "coordinate_axes",
        "coordinate_planes_66",
        "random_coordinate_pairs",
        "random_2d_planes",
        "support_atom_projected",
    ]
)

_LP = None


def normalize_rows(x: np.ndarray) -> np.ndarray:
    """Normalize nonzero row vectors."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x.reshape(0, 12).astype(np.float32)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(~np.isfinite(norm)) or np.any(norm <= 0.0):
        raise ValueError("A generated direction is zero or non-finite.")
    return (x / norm).astype(np.float32)


def trig_basis(theta_deg: np.ndarray) -> np.ndarray:
    """Return [cos(2 theta), cos(4 theta), sin(2 theta), sin(4 theta)]."""
    theta = np.deg2rad(np.asarray(theta_deg, dtype=np.float64))
    return np.stack(
        [np.cos(2 * theta), np.cos(4 * theta), np.sin(2 * theta), np.sin(4 * theta)],
        axis=1,
    )


def layer_weights(z_edges: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute normalized A-, B-, and D-type thickness weights."""
    z0, z1 = z_edges[:-1], z_edges[1:]
    h = float(z_edges[-1] - z_edges[0])
    return (
        (z1 - z0) / h,
        2.0 * (z1**2 - z0**2) / h**2,
        4.0 * (z1**3 - z0**3) / h**3,
    )


def build_epigraph_lp(z_edges: np.ndarray, theta_deg: np.ndarray):
    """Build the polar epigraph LP used by every radial query."""
    basis = trig_basis(theta_deg)
    wa, wb, wd = layer_weights(z_edges)
    layers, angles = len(wa), len(theta_deg)
    rows, cols, data = [], [], []
    base_rows = np.arange(layers * angles, dtype=np.int32)
    layer_id = np.repeat(np.arange(layers, dtype=np.int32), angles)
    atoms = np.concatenate(
        [
            wa[:, None, None] * basis[None, :, :],
            wb[:, None, None] * basis[None, :, :],
            wd[:, None, None] * basis[None, :, :],
        ],
        axis=2,
    ).reshape(layers * angles, 12)
    for column in range(12):
        rows.append(base_rows)
        cols.append(np.full(layers * angles, column, dtype=np.int32))
        data.append(atoms[:, column])
    rows.append(base_rows)
    cols.append(12 + layer_id)
    data.append(np.full(layers * angles, -1.0))
    rows.append(np.full(layers, layers * angles, dtype=np.int32))
    cols.append(12 + np.arange(layers, dtype=np.int32))
    data.append(np.ones(layers))
    matrix = coo_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(layers * angles + 1, 12 + layers),
    ).tocsc()
    rhs = np.zeros(layers * angles + 1)
    rhs[-1] = 1.0
    return matrix, rhs, [(None, None)] * (12 + layers)


def dense_random(n: int, rng: np.random.Generator):
    return normalize_rows(rng.standard_normal((n, 12)))


def sparse_random(n: int, rng: np.random.Generator):
    """Generate rays with two to four nonzero coordinates."""
    out = np.zeros((n, 12), dtype=np.float64)
    for row in range(n):
        k = int(rng.integers(2, 5))
        idx = rng.choice(12, k, replace=False)
        out[row, idx] = rng.standard_normal(k)
    return normalize_rows(out)


def coordinate_axes(n: int):
    """Return the 24 positive and negative coordinate-axis rays."""
    if n not in (0, 24):
        raise ValueError("The coordinate-axis source count must be 0 or 24.")
    if n == 0:
        return np.empty((0, 12), dtype=np.float32)
    eye = np.eye(12, dtype=np.float32)
    return np.concatenate([eye, -eye], axis=0)


def coordinate_planes(n: int, rng: np.random.Generator):
    """Sweep all 66 coordinate planes and retain slice metadata."""
    pairs = [(i, j) for i in range(12) for j in range(i + 1, 12)]
    if n and n < len(pairs) * 8:
        raise ValueError("Use at least 528 coordinate-plane rays (8 per plane).")
    base, remainder = divmod(n, len(pairs))
    rays, slice_ids, angles = [], [], []
    for sid, (i, j) in enumerate(pairs):
        count = base + int(sid < remainder)
        if count == 0:
            continue
        phase = rng.uniform(0.0, 2.0 * np.pi)
        angle = phase + 2.0 * np.pi * np.arange(count) / count
        block = np.zeros((count, 12), dtype=np.float32)
        block[:, i], block[:, j] = np.cos(angle), np.sin(angle)
        rays.append(block)
        slice_ids.append(np.full(count, sid, dtype=np.int32))
        angles.append(np.mod(angle, 2.0 * np.pi).astype(np.float32))
    if not rays:
        return np.empty((0, 12), np.float32), np.empty(0, np.int32), np.empty(0, np.float32)
    return np.concatenate(rays), np.concatenate(slice_ids), np.concatenate(angles)


def random_coordinate_pairs(n: int, rng: np.random.Generator):
    pairs = np.asarray([(i, j) for i in range(12) for j in range(i + 1, 12)])
    selected = pairs[rng.integers(0, len(pairs), size=n)]
    angle = rng.uniform(0.0, 2.0 * np.pi, size=n)
    out = np.zeros((n, 12), dtype=np.float32)
    out[np.arange(n), selected[:, 0]] = np.cos(angle)
    out[np.arange(n), selected[:, 1]] = np.sin(angle)
    return out


def random_2d_planes(n: int, rng: np.random.Generator, angles_per_plane: int):
    """Generate angular sweeps in random orthonormal 2-D planes."""
    if n and n < angles_per_plane:
        raise ValueError("The random-plane count must cover at least one full plane.")
    blocks, slice_ids, angles = [], [], []
    plane_counts = [angles_per_plane] * (n // angles_per_plane)
    remainder = n % angles_per_plane
    if remainder >= 8:
        plane_counts.append(remainder)
    elif remainder:
        plane_counts[-1] += remainder
    sid = 1000
    for count in plane_counts:
        basis, _ = np.linalg.qr(rng.standard_normal((12, 2)), mode="reduced")
        phase = rng.uniform(0.0, 2.0 * np.pi)
        angle = phase + 2.0 * np.pi * np.arange(count) / count
        block = np.cos(angle)[:, None] * basis[:, 0] + np.sin(angle)[:, None] * basis[:, 1]
        blocks.append(normalize_rows(block))
        slice_ids.append(np.full(count, sid, dtype=np.int32))
        angles.append(np.mod(angle, 2.0 * np.pi).astype(np.float32))
        sid += 1
    if not blocks:
        return np.empty((0, 12), np.float32), np.empty(0, np.int32), np.empty(0, np.float32)
    return np.concatenate(blocks), np.concatenate(slice_ids), np.concatenate(angles)


def support_atom_projected(
    n: int,
    rng: np.random.Generator,
    z_edges: np.ndarray,
    theta_deg: np.ndarray,
):
    """Project random support normals onto exposed layer-wise angular atoms."""
    basis = trig_basis(theta_deg)
    wa, wb, wd = layer_weights(z_edges)
    out = np.empty((n, 12), dtype=np.float64)
    for start in range(0, n, 512):
        stop = min(n, start + 512)
        normal = normalize_rows(rng.standard_normal((stop - start, 12))).astype(np.float64)
        effective = (
            normal[:, None, 0:4] * wa[None, :, None]
            + normal[:, None, 4:8] * wb[None, :, None]
            + normal[:, None, 8:12] * wd[None, :, None]
        )
        score = np.einsum("bli,mi->blm", effective, basis, optimize=True)
        chosen = basis[np.argmax(score, axis=2)]
        out[start:stop] = np.concatenate(
            [
                np.sum(wa[None, :, None] * chosen, axis=1),
                np.sum(wb[None, :, None] * chosen, axis=1),
                np.sum(wd[None, :, None] * chosen, axis=1),
            ],
            axis=1,
        )
    return normalize_rows(out)


def generate_rays(counts, seed, z_edges, theta_deg, plane_angles):
    """Generate all seven ray sources and aligned metadata arrays."""
    rng = np.random.default_rng(seed)
    blocks, sources, slices, angles = [], [], [], []

    def add(block, source, slice_id=None, angle=None):
        blocks.append(np.asarray(block, dtype=np.float32))
        sources.append(np.full(len(block), source, dtype=np.uint8))
        slices.append(np.full(len(block), -1, dtype=np.int32) if slice_id is None else slice_id)
        angles.append(np.full(len(block), np.nan, dtype=np.float32) if angle is None else angle)

    add(dense_random(counts[0], rng), 0)
    add(sparse_random(counts[1], rng), 1)
    add(coordinate_axes(counts[2]), 2)
    block, sid, angle = coordinate_planes(counts[3], rng)
    add(block, 3, sid, angle)
    add(random_coordinate_pairs(counts[4], rng), 4)
    block, sid, angle = random_2d_planes(counts[5], rng, plane_angles)
    add(block, 5, sid, angle)
    add(support_atom_projected(counts[6], rng, z_edges, theta_deg), 6)

    ray = np.concatenate(blocks)
    source = np.concatenate(sources)
    slice_id = np.concatenate(slices)
    slice_angle = np.concatenate(angles)
    order = rng.permutation(len(ray))
    return ray[order], source[order], slice_id[order], slice_angle[order]


def init_worker(z_edges, theta_deg):
    global _LP
    _LP = build_epigraph_lp(np.asarray(z_edges), np.asarray(theta_deg))


def solve_block(ray_block: np.ndarray):
    """Solve a block of independent radial LPs inside one worker."""
    matrix, rhs, bounds = _LP
    n = len(ray_block)
    r_max = np.full(n, np.nan)
    gamma = np.full(n, np.nan)
    normal = np.full((n, 12), np.nan)
    violation = np.full(n, np.nan)
    status = np.full(n, -1, dtype=np.int16)
    for i, ray in enumerate(ray_block):
        objective = np.zeros(matrix.shape[1])
        objective[:12] = -ray
        result = linprog(
            objective,
            A_ub=matrix,
            b_ub=rhs,
            bounds=bounds,
            method="highs-ds",
            options={"presolve": False},
        )
        status[i] = int(result.status)
        if result.success:
            gamma[i] = -float(result.fun)
            r_max[i] = 1.0 / gamma[i]
            normal[i] = result.x[:12]
            violation[i] = max(0.0, float(np.max(matrix @ result.x - rhs)))
    return r_max, gamma, normal, violation, status


def solve_all(rays, z_edges, theta_deg, workers, chunk_size):
    """Solve all rays sequentially or with process-level parallelism."""
    blocks = [rays[i : i + chunk_size] for i in range(0, len(rays), chunk_size)]
    outputs = []
    start = time.time()
    if workers == 1:
        init_worker(z_edges, theta_deg)
        for block in blocks:
            outputs.append(solve_block(block))
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=init_worker,
            initargs=(z_edges, theta_deg),
        ) as pool:
            for index, result in enumerate(pool.map(solve_block, blocks), start=1):
                outputs.append(result)
                if index == len(blocks) or index % max(1, len(blocks) // 20) == 0:
                    print(f"Solved {min(index * chunk_size, len(rays)):,}/{len(rays):,} rays", flush=True)
    joined = [np.concatenate([item[k] for item in outputs], axis=0) for k in range(5)]
    return (*joined, time.time() - start)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/multisource_demo.npz")
    parser.add_argument("--counts", default="1000,300,24,1056,300,720,300")
    parser.add_argument("--nz", type=int, default=64)
    parser.add_argument("--ntheta", type=int, default=181)
    parser.add_argument("--plane-angles", type=int, default=72)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    parser.add_argument("--chunk-size", type=int, default=32)
    args = parser.parse_args()

    counts = tuple(int(value) for value in args.counts.split(","))
    if len(counts) != 7:
        raise ValueError("--counts must contain seven comma-separated integers.")
    z_edges = np.linspace(-0.5, 0.5, args.nz + 1, dtype=np.float64)
    theta_deg = np.linspace(-90.0, 90.0, args.ntheta, dtype=np.float64)
    rays, source_id, slice_id, slice_angle = generate_rays(
        counts, args.seed, z_edges, theta_deg, args.plane_angles
    )
    r_max, gamma, normal, violation, status, elapsed = solve_all(
        rays, z_edges, theta_deg, args.workers, args.chunk_size
    )
    if np.any(status != 0) or np.any(~np.isfinite(r_max)):
        bad = int(np.sum((status != 0) | ~np.isfinite(r_max)))
        raise RuntimeError(f"StrictLP failed for {bad} rays.")
    boundary = (rays.astype(np.float64) * r_max[:, None]).astype(np.float32)
    identity_error = np.abs(np.einsum("ij,ij->i", normal, boundary) - 1.0)
    metadata = {
        "seed": args.seed,
        "nz": args.nz,
        "ntheta": args.ntheta,
        "counts": dict(zip(SOURCE_NAMES.tolist(), counts)),
        "ray_count": int(len(rays)),
        "elapsed_seconds": elapsed,
        "max_lp_violation": float(np.max(violation)),
        "max_support_identity_error": float(np.max(identity_error)),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        V_boundary=boundary,
        rays_12d=rays,
        r_max=r_max.astype(np.float32),
        gamma=gamma.astype(np.float32),
        support_normal=normal.astype(np.float32),
        lp_violation=violation.astype(np.float32),
        source_id=source_id,
        source_names=SOURCE_NAMES,
        slice_id=slice_id,
        slice_angle=slice_angle,
        z_edges=z_edges,
        theta_deg=theta_deg,
        meta_json=np.asarray(json.dumps(metadata)),
    )
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    print(f"Saved: {output.resolve()}")


if __name__ == "__main__":
    main()
