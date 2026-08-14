#!/usr/bin/env python
"""Validate an SMICNN checkpoint with the metrics defined in the paper.

Reported groups:
- fitting/classification: BT_MSE, BT_bias, BT_p95, BT_p99, BT_max, T_MSE,
  Sign, and NearSign;
- 2-D geometry: signed bias, absolute MAE/p95/max, and outward expansion;
- deployment safety: false-feasible rate and one-sided 99% exact upper bound;
- gradients: N_ig, N_mg, N_ng, Euler MAE/p95/p99, and e_FD.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from scipy.stats import beta

try:  # Support both `python src/code3_...py` and `python -m src.code3_...`.
    from .code2_train_smicnn import SmoothMaxICNN
except ImportError:
    from code2_train_smicnn import SmoothMaxICNN


FIT_THRESHOLDS = {
    "BT_MSE": ("max", 8e-5),
    "abs_BT_bias": ("max", 2e-3),
    "BT_p99": ("max", 0.025),
    "T_MSE": ("max", 1e-4),
    "Sign": ("min", 0.99),
    "NearSign": ("min", 0.98),
}

DEPLOYMENT_LIMITS = {
    1.001: 0.0050,
    1.002: 0.0030,
    1.005: 0.0010,
    1.010: 0.0005,
    1.020: 0.00005,
}

GRADIENT_THRESHOLDS = {
    "N_ig": ("max", 0),
    "N_mg": ("max", 0),
    "N_ng": ("max", 0),
    "Euler_MAE": ("max", 0.05),
    "Euler_p95": ("max", 0.10),
    "Euler_p99": ("max", 0.10),
    "e_FD_p99": ("max", 0.01),
}


def load_model(checkpoint_path: str, device: torch.device):
    """Recreate the concise model from checkpoint metadata."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    architecture = checkpoint["architecture"]
    model = SmoothMaxICNN(
        width=int(architecture["width"]),
        depth=int(architecture["depth"]),
        pieces=int(architecture["pieces"]),
        output_pieces=int(architecture["output_pieces"]),
        tau=float(architecture["tau"]),
        adaptive_pieces=int(architecture["adaptive_pieces"]),
        exact_pieces=int(architecture["exact_pieces"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    x_scale = torch.as_tensor(checkpoint["x_scale"], dtype=torch.float32, device=device)
    return model, x_scale, architecture


def load_data(path: str, sample_size: int, seed: int):
    """Load one independent StrictLP data set and optionally take a fixed subset."""
    data = np.load(path, allow_pickle=False)
    arrays = {key: data[key] for key in data.files}
    n = len(arrays["V_boundary"])
    if sample_size > 0 and sample_size < n:
        index = np.random.default_rng(seed).choice(n, sample_size, replace=False)
        for key in ("V_boundary", "rays_12d", "r_max", "source_id", "slice_id", "slice_angle"):
            if key in arrays:
                arrays[key] = arrays[key][index]
    return arrays


@torch.no_grad()
def predict(model, x_scale, physical_x: np.ndarray, device, batch_size: int):
    """Evaluate physical coordinates in bounded batches."""
    output = []
    for start in range(0, len(physical_x), batch_size):
        x = torch.as_tensor(physical_x[start : start + batch_size], dtype=torch.float32, device=device)
        output.append(model(x / x_scale).squeeze(1).cpu().numpy())
    return np.concatenate(output) if output else np.empty(0)


def pass_check(value, rule):
    direction, threshold = rule
    return bool(value <= threshold) if direction == "max" else bool(value >= threshold)


def fitting_metrics(model, x_scale, boundary, args, device):
    """Compute the seven fitting and classification indicators in Section 4.4.1."""
    boundary_value = predict(model, x_scale, boundary, device, args.batch_size)
    boundary_error = boundary_value - 1.0
    squared_sum, total_count = 0.0, 0
    sign_correct, sign_count = 0, 0
    near_correct, near_count = 0, 0
    t_values = np.asarray([float(v) for v in args.t_values.split(",")], dtype=np.float32)
    for t in t_values:
        value = predict(model, x_scale, boundary * t, device, args.batch_size)
        squared_sum += float(np.sum((value - t) ** 2))
        total_count += len(value)
        distance = abs(float(t) - 1.0)
        correct = np.sign(value - 1.0) == np.sign(float(t) - 1.0)
        if distance > 0.02:
            sign_correct += int(np.sum(correct))
            sign_count += len(correct)
        if 0.02 < distance < 0.05:
            near_correct += int(np.sum(correct))
            near_count += len(correct)
    metrics = {
        "BT_MSE": float(np.mean(boundary_error**2)),
        "BT_bias": float(np.mean(boundary_error)),
        "BT_p95": float(np.quantile(np.abs(boundary_error), 0.95)),
        "BT_p99": float(np.quantile(np.abs(boundary_error), 0.99)),
        "BT_max": float(np.max(np.abs(boundary_error))),
        "T_MSE": squared_sum / max(total_count, 1),
        "Sign": sign_correct / max(sign_count, 1),
        "NearSign": near_correct / max(near_count, 1),
    }
    tested = {**metrics, "abs_BT_bias": abs(metrics["BT_bias"])}
    return {
        "metrics": metrics,
        "requirements": {
            name: {"rule": rule[0], "threshold": rule[1], "pass": pass_check(tested[name], rule)}
            for name, rule in FIT_THRESHOLDS.items()
        },
    }


def one_sided_upper_99(k: int, n: int) -> float:
    """One-sided 99% Clopper-Pearson upper confidence bound."""
    if n <= 0:
        return float("nan")
    if k >= n:
        return 1.0
    return float(beta.ppf(0.99, k + 1, n - k))


def deployment_metrics(model, x_scale, boundary, rho, args, device):
    """Evaluate the five exterior shells and their one-sided 99% bounds."""
    result = []
    for outer_scale, limit in DEPLOYMENT_LIMITS.items():
        value = predict(model, x_scale, boundary * outer_scale, device, args.batch_size)
        false_feasible = value <= rho
        count = int(np.sum(false_feasible))
        upper = one_sided_upper_99(count, len(value))
        result.append(
            {
                "outer_scale": outer_scale,
                "rho": rho,
                "false_feasible_count": count,
                "sample_count": int(len(value)),
                "false_feasible_rate": count / max(len(value), 1),
                "one_sided_99_upper": upper,
                "requirement": limit,
                "pass": bool(upper <= limit),
            }
        )
    return result


def gradient_metrics(model, x_scale, boundary, args, device):
    """Compute the seven gradient indicators in Section 4.4.3."""
    rng = np.random.default_rng(args.seed + 1)
    count = min(args.gradient_samples, len(boundary))
    index = rng.choice(len(boundary), count, replace=False)
    t = rng.choice(np.asarray([0.98, 1.0, 1.02], dtype=np.float32), count)
    physical = (boundary[index] * t[:, None]).astype(np.float32)
    x = torch.tensor(physical, device=device, requires_grad=True)
    phi = model(x / x_scale).squeeze(1)
    gradient = torch.autograd.grad(phi.sum(), x)[0]
    gradient_np = gradient.detach().cpu().numpy()
    phi_np = phi.detach().cpu().numpy()
    finite = np.isfinite(gradient_np).all(axis=1) & np.isfinite(phi_np)
    norm = np.linalg.norm(gradient_np, axis=1)
    radial = np.einsum("ij,ij->i", physical, gradient_np)
    euler = np.abs(radial[finite] - phi_np[finite])
    n_ig = int(np.sum(~finite))
    n_mg = int(np.sum(finite & (norm <= args.tiny_gradient)))
    n_ng = int(np.sum(finite & (radial <= 0.0)))

    fd_count = min(args.fd_samples, int(np.sum(finite)))
    fd_points = physical[np.flatnonzero(finite)[:fd_count]]
    ad_gradient = gradient_np[np.flatnonzero(finite)[:fd_count]]
    fd_gradient = np.empty_like(ad_gradient)
    for dimension in range(12):
        plus, minus = fd_points.copy(), fd_points.copy()
        plus[:, dimension] += args.fd_step
        minus[:, dimension] -= args.fd_step
        fd_gradient[:, dimension] = (
            predict(model, x_scale, plus, device, args.batch_size)
            - predict(model, x_scale, minus, device, args.batch_size)
        ) / (2.0 * args.fd_step)
    numerator = np.linalg.norm(ad_gradient - fd_gradient, axis=1)
    denominator = np.maximum.reduce(
        [
            np.linalg.norm(ad_gradient, axis=1),
            np.linalg.norm(fd_gradient, axis=1),
            np.full(fd_count, 1e-8),
        ]
    )
    e_fd = numerator / denominator
    metrics = {
        "N_ig": n_ig,
        "N_mg": n_mg,
        "N_ng": n_ng,
        "Euler_MAE": float(np.mean(euler)),
        "Euler_p95": float(np.quantile(euler, 0.95)),
        "Euler_p99": float(np.quantile(euler, 0.99)),
        "Euler_max": float(np.max(euler)),
        "e_FD_p99": float(np.quantile(e_fd, 0.99)),
    }
    return {
        "metrics": metrics,
        "requirements": {
            name: {"rule": rule[0], "threshold": rule[1], "pass": pass_check(metrics[name], rule)}
            for name, rule in GRADIENT_THRESHOLDS.items()
        },
    }


@torch.no_grad()
def radial_roots(model, x_scale, rays, true_radius, rho, args, device):
    """Find phi(r u)=rho by vectorized bisection."""
    low = np.zeros(len(rays), dtype=np.float32)
    high = np.maximum(1.5 * true_radius, 1.0).astype(np.float32)
    for _ in range(8):
        value = predict(model, x_scale, rays * high[:, None], device, args.batch_size)
        mask = value < rho
        if not np.any(mask):
            break
        high[mask] *= 2.0
    for _ in range(args.bisection_iterations):
        middle = 0.5 * (low + high)
        value = predict(model, x_scale, rays * middle[:, None], device, args.batch_size)
        inside = value <= rho
        low[inside], high[~inside] = middle[inside], middle[~inside]
    return 0.5 * (low + high)


def periodic_area(angle: np.ndarray, radius: np.ndarray) -> float:
    """Compute a polar area on a closed angular sweep."""
    order = np.argsort(angle)
    theta, r = angle[order], radius[order]
    theta = np.concatenate([theta, theta[:1] + 2.0 * np.pi])
    r = np.concatenate([r, r[:1]])
    return float(0.5 * np.trapezoid(r**2, theta))


def slice_group_metrics(model, x_scale, arrays, source_name, rho, args, device):
    """Aggregate the manuscript's relative radial/area slice metrics."""
    names = [str(value) for value in arrays["source_names"]]
    if source_name not in names:
        return None
    source_index = names.index(source_name)
    mask = (arrays["source_id"] == source_index) & (arrays["slice_id"] >= 0)
    ids = np.unique(arrays["slice_id"][mask])
    radial_error, area_error, expanded_slices = [], [], 0
    for sid in ids:
        selected = mask & (arrays["slice_id"] == sid)
        ray = arrays["rays_12d"][selected].astype(np.float32)
        true_radius = arrays["r_max"][selected].astype(np.float32)
        angle = arrays["slice_angle"][selected].astype(np.float64)
        if len(ray) < 8:
            continue
        predicted = radial_roots(model, x_scale, ray, true_radius, rho, args, device)
        relative_radial = 100.0 * (predicted - true_radius) / true_radius
        radial_error.extend(relative_radial)
        expanded_slices += int(np.any(relative_radial > args.radial_expansion_tolerance))
        true_area = periodic_area(angle, true_radius)
        predicted_area = periodic_area(angle, predicted)
        area_error.append(100.0 * (predicted_area - true_area) / true_area)
    radial_signed = np.asarray(radial_error)
    area_signed = np.asarray(area_error)
    radial_abs = np.abs(radial_signed)
    area_abs = np.abs(area_signed)
    if not len(radial_abs) or not len(area_abs):
        return None
    metrics = {
        "rho": rho,
        "slice_count": int(len(area_abs)),
        "relative_radial_bias_percent": float(np.mean(radial_signed)),
        "relative_radial_MAE_percent": float(np.mean(radial_abs)),
        "relative_radial_p95_percent": float(np.quantile(radial_abs, 0.95)),
        "relative_radial_max_percent": float(np.max(radial_abs)),
        "relative_radial_max_outward_percent": float(np.max(radial_signed)),
        "radial_expansion_tolerance_percent": args.radial_expansion_tolerance,
        "outward_direction_fraction": float(
            np.mean(radial_signed > args.radial_expansion_tolerance)
        ),
        "slices_with_outward_expansion": expanded_slices,
        "slice_outward_expansion_fraction": expanded_slices / max(len(area_abs), 1),
        "relative_area_bias_percent": float(np.mean(area_signed)),
        "relative_area_MAE_percent": float(np.mean(area_abs)),
        "relative_area_p95_percent": float(np.quantile(area_abs, 0.95)),
        "relative_area_max_percent": float(np.max(area_abs)),
    }
    if source_name == "coordinate_planes_66":
        reference = {
            "relative_radial_MAE_percent": 1.5,
            "relative_area_MAE_percent": 5.0,
            "relative_area_p95_percent": 10.0,
        }
    else:
        reference = {"relative_radial_MAE_percent": 2.0}
    metrics["reference_requirements"] = {
        key: {"threshold": limit, "pass": bool(metrics[key] <= limit)}
        for key, limit in reference.items()
    }
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="reports/metrics.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--sample-size", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--rho", type=float, default=0.98)
    parser.add_argument("--slice-rhos", default="1,0.99,0.98,0.97,0.96")
    parser.add_argument("--t-values", default="0,0.3,0.5,0.7,0.85,0.95,0.97,0.98,1.02,1.03,1.05,1.15")
    parser.add_argument("--gradient-samples", type=int, default=1000)
    parser.add_argument("--tiny-gradient", type=float, default=1e-10)
    parser.add_argument("--fd-samples", type=int, default=128)
    parser.add_argument("--fd-step", type=float, default=1e-4)
    parser.add_argument("--bisection-iterations", type=int, default=45)
    parser.add_argument(
        "--radial-expansion-tolerance",
        type=float,
        default=1.0,
        help="Positive relative radial error (%%) counted as meaningful outward expansion.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    model, x_scale, architecture = load_model(args.checkpoint, device)
    arrays = load_data(args.data, args.sample_size, args.seed)
    boundary = np.asarray(arrays["V_boundary"], dtype=np.float32)
    report = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "data": str(Path(args.data).resolve()),
        "architecture": architecture,
        "fitting_and_classification": fitting_metrics(model, x_scale, boundary, args, device),
        "deployment_safety": deployment_metrics(model, x_scale, boundary, args.rho, args, device),
        "gradient_performance": gradient_metrics(model, x_scale, boundary, args, device),
        "fixed_coordinate_slices": [],
        "random_2d_slices": [],
    }
    required_slice_keys = {"source_names", "source_id", "slice_id", "slice_angle", "rays_12d", "r_max"}
    if required_slice_keys.issubset(arrays):
        for rho in [float(v) for v in args.slice_rhos.split(",")]:
            fixed = slice_group_metrics(
                model, x_scale, arrays, "coordinate_planes_66", rho, args, device
            )
            random_slice = slice_group_metrics(
                model, x_scale, arrays, "random_2d_planes", rho, args, device
            )
            if fixed is not None:
                report["fixed_coordinate_slices"].append(fixed)
            if random_slice is not None:
                report["random_2d_slices"].append(random_slice)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved: {output.resolve()}")


if __name__ == "__main__":
    main()
