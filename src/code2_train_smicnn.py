#!/usr/bin/env python
"""Train a support-augmented Smooth-Max ICNN on radial boundary data.

The concise four-stage workflow is:
1. train the deep ICNN backbone;
2. initialize trainable adaptive planes from model gradients;
3. append frozen StrictLP support planes for hard boundary rays;
4. mine a separate data set (if supplied), append more exact planes, and refine.

The final output is the normalized Smooth-Max value itself; no output Softplus,
hinge wrapper, or safety head is used.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Paper-scale weights. Closely related L1/L2 variants are consolidated so that
# the public script remains readable while preserving all six loss groups.
W = {
    "T": 16.0,
    "Y": 0.002,
    "BT": 1100.0,
    "BT_L1": 40.0,
    "NT": 350.0,
    "NT_L1": 32.0,
    "TOP": 800.0,
    "TAIL_POS": 800.0,
    "TAIL_NEG": 6000.0,
    "BIAS": 160000.0,
    "SIGN": 1.0,
    "SIGN_HINGE": 8.0,
    "SLOPE": 3.0,
    "EULER_T": 3.0,
    "HOM": 1.5,
    "EULER_BOUNDARY": 6.0,
    "RADIAL_POSITIVE": 2.0,
    "ROOT": 4.0,
    "HARD_BOUNDARY": 4000.0,
    "HARD_INSIDE": 500.0,
    "HARD_OUTSIDE": 6000.0,
    "HARD_TOP": 6000.0,
}

def smooth_max(values: torch.Tensor, dim: int, tau: float) -> torch.Tensor:
    """Normalized log-mean-exp; it converges to max as tau approaches zero."""
    if tau <= 0.0:
        return values.max(dim=dim).values
    return tau * torch.logsumexp(values.float() / tau, dim=dim) - tau * math.log(values.shape[dim])


class NonNegativeLinear(nn.Module):
    """Bias-free linear layer with elementwise-squared nonnegative weights."""

    def __init__(self, in_features: int, out_features: int, initial_weight: float = 5e-4):
        super().__init__()
        center = math.sqrt(initial_weight)
        self.raw_weight = nn.Parameter(center + 0.01 * torch.randn(out_features, in_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.raw_weight.square())


class SmoothICNNLayer(nn.Module):
    """One convex hidden state with input skip connections."""

    def __init__(self, input_dim: int, width: int, pieces: int, tau: float):
        super().__init__()
        self.width, self.pieces, self.tau = width, pieces, tau
        self.z_map = NonNegativeLinear(width, width * pieces)
        self.x_map = nn.Linear(input_dim, width * pieces, bias=False)
        nn.init.xavier_uniform_(self.x_map.weight)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        candidates = self.z_map(z).view(batch, self.width, self.pieces)
        candidates = candidates + self.x_map(x).view(batch, self.width, self.pieces)
        return smooth_max(candidates, dim=2, tau=self.tau)


class SmoothMaxICNN(nn.Module):
    """Deep ICNN candidates plus trainable and exact direct affine supports."""

    def __init__(
        self,
        width: int = 2560,
        depth: int = 4,
        pieces: int = 8,
        output_pieces: int = 32,
        tau: float = 1e-4,
        adaptive_pieces: int = 0,
        exact_pieces: int = 0,
    ):
        super().__init__()
        self.width = width
        self.depth = depth
        self.pieces = pieces
        self.output_pieces = output_pieces
        self.tau = tau
        self.input_map = nn.Linear(12, width * pieces, bias=False)
        nn.init.xavier_uniform_(self.input_map.weight)
        self.layers = nn.ModuleList(
            [SmoothICNNLayer(12, width, pieces, tau) for _ in range(depth - 1)]
        )
        self.output_z = NonNegativeLinear(width, output_pieces)
        self.output_x = nn.Linear(12, output_pieces, bias=False)
        nn.init.xavier_uniform_(self.output_x.weight)
        self.adaptive_weight = nn.Parameter(torch.empty(adaptive_pieces, 12))
        if adaptive_pieces:
            nn.init.xavier_uniform_(self.adaptive_weight)
        self.register_buffer("exact_weight", torch.empty(exact_pieces, 12))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        z = smooth_max(self.input_map(x).view(batch, self.width, self.pieces), 2, self.tau)
        for layer in self.layers:
            z = layer(x, z)
        candidates = [self.output_z(z) + self.output_x(x)]
        if self.adaptive_weight.numel():
            candidates.append(F.linear(x, self.adaptive_weight))
        if self.exact_weight.numel():
            candidates.append(F.linear(x, self.exact_weight))
        return smooth_max(torch.cat(candidates, dim=1), 1, self.tau).unsqueeze(1)

    def add_adaptive(self, weight: torch.Tensor):
        """Append trainable affine planes and rebuild the parameter tensor."""
        if weight.numel() == 0:
            return
        merged = torch.cat([self.adaptive_weight.detach(), weight.detach()], dim=0)
        self.adaptive_weight = nn.Parameter(merged)

    def add_exact(self, weight: torch.Tensor):
        """Append frozen exact StrictLP planes, removing rounded duplicates."""
        if weight.numel() == 0:
            return
        merged = torch.cat([self.exact_weight.detach().cpu(), weight.detach().cpu()], dim=0)
        rounded = np.round(merged.numpy(), decimals=7)
        _, keep = np.unique(rounded, axis=0, return_index=True)
        self.exact_weight = merged[torch.from_numpy(np.sort(keep))].to(self.exact_weight.device)


def load_boundary(path: str):
    """Load boundary points and optional exact polar normals."""
    data = np.load(path, allow_pickle=False)
    boundary = np.asarray(data["V_boundary"], dtype=np.float32)
    normal = np.asarray(data["support_normal"], dtype=np.float32) if "support_normal" in data else None
    if boundary.ndim != 2 or boundary.shape[1] != 12:
        raise ValueError("V_boundary must have shape [N, 12].")
    return boundary, normal


@torch.no_grad()
def boundary_errors(model, boundary, x_scale, device, batch_size=4096):
    """Return signed model errors on physical boundary points."""
    values = []
    model.eval()
    for start in range(0, len(boundary), batch_size):
        x = torch.from_numpy(boundary[start : start + batch_size]).to(device)
        values.append((model(x / x_scale).squeeze(1) - 1.0).cpu())
    return torch.cat(values).numpy()


def adaptive_planes(model, boundary, x_scale, count, device):
    """Initialize planes from gradients on the largest boundary residuals."""
    if count <= 0:
        return torch.empty(0, 12, device=device)
    error = np.abs(boundary_errors(model, boundary, x_scale, device))
    index = np.argsort(error)[-min(count, len(boundary)) :]
    x = torch.from_numpy(boundary[index]).to(device) / x_scale
    x.requires_grad_(True)
    model.eval()
    value = model(x)
    gradient = torch.autograd.grad(value.sum(), x)[0]
    denominator = (gradient * x).sum(dim=1, keepdim=True)
    valid = torch.isfinite(gradient).all(dim=1) & (denominator.squeeze(1).abs() > 1e-8)
    return (gradient[valid] / denominator[valid]).detach()


def exact_planes(model, boundary, normal, x_scale, count, device):
    """Select exact StrictLP supports at the currently hardest boundary rays."""
    if count <= 0:
        return torch.empty(0, 12, device=device)
    if normal is None:
        raise ValueError("Exact-support stages require support_normal in the NPZ file.")
    error = np.abs(boundary_errors(model, boundary, x_scale, device))
    index = np.argsort(error)[-min(count, len(boundary)) :]
    # The model input is V / x_scale, so normalized weights are c * x_scale.
    weight = torch.from_numpy(normal[index]).to(device) * x_scale
    return weight.detach()


def sample_t(batch: int, values: torch.Tensor) -> torch.Tensor:
    index = torch.randint(0, len(values), (batch,), device=values.device)
    return values[index]


def training_loss(model, boundary, hard_boundary, x_scale, args, device, t_values):
    """Compute the six manuscript loss groups in a compact implementation."""
    n = len(boundary)
    idx = torch.randint(0, n, (args.batch,), device=device)
    vb = torch.from_numpy(boundary[idx.cpu().numpy()]).to(device)
    t = sample_t(args.batch, t_values)
    phi = model(vb * t[:, None] / x_scale).squeeze(1)
    loss_t = F.mse_loss(phi, t)
    loss_y = F.mse_loss(phi.square(), t.square())
    loss_value = W["T"] * loss_t + W["Y"] * loss_y

    bidx = torch.randint(0, n, (args.boundary_batch,), device=device)
    xb = torch.from_numpy(boundary[bidx.cpu().numpy()]).to(device) / x_scale
    boundary_error = model(xb).squeeze(1) - 1.0
    squared = boundary_error.square()
    top_count = max(1, int(math.ceil(args.top_fraction * len(squared))))
    positive_tail = F.relu(boundary_error - args.positive_tail_margin).square().mean()
    negative_tail = F.relu(-boundary_error - args.negative_tail_margin).square().mean()
    bias_error = F.relu(
        (boundary_error.mean() - args.bias_target).abs() - args.bias_band
    ).square()
    near_mask = (t - 1.0).abs() <= 0.05
    near_mse = (phi[near_mask] - t[near_mask]).square().mean() if near_mask.any() else phi.new_zeros(())
    near_l1 = (phi[near_mask] - t[near_mask]).abs().mean() if near_mask.any() else phi.new_zeros(())
    loss_boundary = (
        W["BT"] * squared.mean()
        + W["BT_L1"] * boundary_error.abs().mean()
        + W["NT"] * near_mse
        + W["NT_L1"] * near_l1
        + W["TOP"] * squared.topk(top_count).values.mean()
        + W["TAIL_POS"] * positive_tail
        + W["TAIL_NEG"] * negative_tail
        + W["BIAS"] * bias_error
    )

    sign_mask = (t - 1.0).abs() > args.sign_margin
    if sign_mask.any():
        logit = args.logit_scale * (phi[sign_mask] - 1.0)
        target = (t[sign_mask] > 1.0).float()
        loss_bce = F.binary_cross_entropy_with_logits(logit, target)
        signed_phi = phi[sign_mask]
        signed_t = t[sign_mask]
        inside = signed_t < 1.0
        outside = signed_t > 1.0
        hinge_inside = (
            F.relu(signed_phi[inside] - (1.0 - args.sign_hinge_margin)).square().mean()
            if inside.any()
            else phi.new_zeros(())
        )
        hinge_outside = (
            F.relu((1.0 + args.sign_hinge_margin) - signed_phi[outside]).square().mean()
            if outside.any()
            else phi.new_zeros(())
        )
        loss_sign = W["SIGN"] * loss_bce + 0.5 * W["SIGN_HINGE"] * (
            hinge_inside + hinge_outside
        )
    else:
        loss_sign = phi.new_zeros(())

    loss_gradient = phi.new_zeros(())
    loss_root = phi.new_zeros(())
    if args.gradient_batch > 0:
        gidx = torch.randint(0, n, (args.gradient_batch,), device=device)
        vg = torch.from_numpy(boundary[gidx.cpu().numpy()]).to(device) / x_scale
        gradient_t = torch.tensor([0.98, 1.0, 1.02], device=device)
        tg = sample_t(args.gradient_batch, gradient_t)
        xg = (vg * tg[:, None]).requires_grad_(True)
        pg = model(xg).squeeze(1)
        grad = torch.autograd.grad(pg.sum(), xg, create_graph=True)[0]
        radial = (xg * grad).sum(dim=1)
        loss_radial = (radial - tg).square().mean()
        loss_hom = (radial - pg).square().mean()
        loss_positive = F.relu(args.radial_floor - radial).square().mean()
        x_boundary = vg.requires_grad_(True)
        p_boundary = model(x_boundary).squeeze(1)
        g_boundary = torch.autograd.grad(p_boundary.sum(), x_boundary, create_graph=True)[0]
        boundary_radial = (x_boundary * g_boundary).sum(dim=1)
        loss_boundary_gradient = (boundary_radial - 1.0).square().mean()
        plus = model((1.0 + args.slope_delta) * vg).squeeze(1)
        minus = model((1.0 - args.slope_delta) * vg).squeeze(1)
        finite_slope = (plus - minus) / (2.0 * args.slope_delta)
        loss_slope = (finite_slope - 1.0).square().mean()
        loss_gradient = (
            W["SLOPE"] * loss_slope
            + W["EULER_T"] * loss_radial
            + W["HOM"] * loss_hom
            + W["EULER_BOUNDARY"] * loss_boundary_gradient
            + W["RADIAL_POSITIVE"] * loss_positive
        )
        root_error = (p_boundary - 1.0) / boundary_radial.abs().clamp_min(args.root_floor)
        root_count = max(1, int(math.ceil(args.root_top_fraction * len(root_error))))
        loss_root = W["ROOT"] * (
            root_error.square().mean() + root_error.square().topk(root_count).values.mean()
        )

    loss_hard = phi.new_zeros(())
    if hard_boundary is not None and len(hard_boundary) and args.hard_batch > 0:
        hidx = torch.randint(0, len(hard_boundary), (args.hard_batch,), device=device)
        hv = torch.from_numpy(hard_boundary[hidx.cpu().numpy()]).to(device)
        hard_boundary_value = model(hv / x_scale).squeeze(1)
        hard_error = hard_boundary_value - 1.0
        hard_count = max(1, int(math.ceil(args.hard_top_fraction * len(hard_error))))
        inside_t = sample_t(
            args.hard_batch, torch.tensor([0.97, 0.98, 0.99, 0.995], device=device)
        )
        outside_t = sample_t(
            args.hard_batch,
            torch.tensor([1.001, 1.002, 1.005, 1.01, 1.02], device=device),
        )
        inside_value = model(hv * inside_t[:, None] / x_scale).squeeze(1)
        outside_value = model(hv * outside_t[:, None] / x_scale).squeeze(1)
        loss_hard = (
            W["HARD_BOUNDARY"] * hard_error.square().mean()
            + W["HARD_TOP"] * hard_error.square().topk(hard_count).values.mean()
            + W["HARD_INSIDE"]
            * F.relu(inside_value - (1.0 - args.hard_margin)).square().mean()
            + W["HARD_OUTSIDE"]
            * F.relu((1.0 + args.hard_margin) - outside_value).square().mean()
        )

    total = loss_value + loss_boundary + loss_sign + loss_gradient + loss_root + loss_hard
    return total, {
        "value": float(loss_value.detach()),
        "boundary": float(loss_boundary.detach()),
        "sign": float(loss_sign.detach()),
        "gradient": float(loss_gradient.detach()),
        "root": float(loss_root.detach()),
        "hard": float(loss_hard.detach()),
    }


def train_stage(
    name, model, boundary, validation, x_scale, epochs, args, device, history, hard_boundary=None
):
    """Train one stage and record a concise epoch history."""
    if epochs <= 0:
        return
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    t_values = torch.tensor([float(v) for v in args.t_values.split(",")], device=device)
    for epoch in range(1, epochs + 1):
        model.train()
        running = {
            key: 0.0
            for key in ("total", "value", "boundary", "sign", "gradient", "root", "hard")
        }
        for _ in range(args.steps_per_epoch):
            optimizer.zero_grad(set_to_none=True)
            total, parts = training_loss(
                model, boundary, hard_boundary, x_scale, args, device, t_values
            )
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            running["total"] += float(total.detach())
            for key, value in parts.items():
                running[key] += value
        val_error = boundary_errors(model, validation, x_scale, device)
        record = {
            "recorded_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "training_stage": name,
            "stage": name,
            "epoch": epoch,
            **{key: value / args.steps_per_epoch for key, value in running.items()},
            "val_BT_MSE": float(np.mean(val_error**2)),
            "val_BT_p95": float(np.quantile(np.abs(val_error), 0.95)),
        }
        history.append(record)
        print(json.dumps(record), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--hard-data", default="")
    parser.add_argument("--output", default="checkpoints/smicnn.pt")
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--width", type=int, default=2560)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--pieces", type=int, default=8)
    parser.add_argument("--output-pieces", type=int, default=32)
    parser.add_argument("--tau", type=float, default=1e-4)
    parser.add_argument("--stage1-epochs", type=int, default=30)
    parser.add_argument("--stage2-epochs", type=int, default=20)
    parser.add_argument("--stage3-epochs", type=int, default=10)
    parser.add_argument("--stage4-epochs", type=int, default=10)
    parser.add_argument("--adaptive-supports", type=int, default=24576)
    parser.add_argument("--exact-supports", type=int, default=128)
    parser.add_argument("--hard-supports", type=int, default=1024)
    parser.add_argument("--validation-fraction", type=float, default=0.02)
    parser.add_argument("--steps-per-epoch", type=int, default=160)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--boundary-batch", type=int, default=1024)
    parser.add_argument("--gradient-batch", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-6)
    parser.add_argument("--grad-clip", type=float, default=3.0)
    parser.add_argument(
        "--t-values",
        default="0.3,0.5,0.7,0.85,0.95,0.975,0.985,0.99,0.995,0.998,0.999,1,1.001,1.002,1.005,1.01,1.015,1.025,1.05,1.15",
    )
    parser.add_argument("--top-fraction", type=float, default=0.30)
    parser.add_argument("--sign-margin", type=float, default=0.002)
    parser.add_argument("--logit-scale", type=float, default=250.0)
    parser.add_argument("--sign-hinge-margin", type=float, default=0.0025)
    parser.add_argument("--positive-tail-margin", type=float, default=0.016)
    parser.add_argument("--negative-tail-margin", type=float, default=0.008)
    parser.add_argument("--bias-target", type=float, default=5e-4)
    parser.add_argument("--bias-band", type=float, default=1e-4)
    parser.add_argument("--slope-delta", type=float, default=0.01)
    parser.add_argument("--radial-floor", type=float, default=0.75)
    parser.add_argument("--root-floor", type=float, default=0.2)
    parser.add_argument("--root-top-fraction", type=float, default=0.35)
    parser.add_argument("--hard-batch", type=int, default=256)
    parser.add_argument("--hard-top-fraction", type=float, default=0.55)
    parser.add_argument("--hard-margin", type=float, default=0.001)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    boundary, normal = load_boundary(args.data)
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(boundary))
    validation_count = max(1, int(round(len(boundary) * args.validation_fraction)))
    val_index, train_index = order[:validation_count], order[validation_count:]
    train_boundary = boundary[train_index]
    val_boundary = boundary[val_index]
    train_normal = normal[train_index] if normal is not None else None
    val_normal = normal[val_index] if normal is not None else None
    x_scale_np = np.maximum(np.quantile(np.abs(train_boundary), 0.995, axis=0), 1e-6).astype(np.float32)
    x_scale = torch.from_numpy(x_scale_np).to(device)

    model = SmoothMaxICNN(
        args.width, args.depth, args.pieces, args.output_pieces, args.tau
    ).to(device)
    history = []
    start = time.time()

    train_stage("Deep ICNN", model, train_boundary, val_boundary, x_scale, args.stage1_epochs, args, device, history)
    model.add_adaptive(adaptive_planes(model, train_boundary, x_scale, args.adaptive_supports, device))
    train_stage("Adaptive direct supports", model, train_boundary, val_boundary, x_scale, args.stage2_epochs, args, device, history)
    model.add_exact(exact_planes(model, val_boundary, val_normal, x_scale, args.exact_supports, device))
    train_stage("Hard direct supports", model, train_boundary, val_boundary, x_scale, args.stage3_epochs, args, device, history)

    if args.hard_data:
        hard_boundary, hard_normal = load_boundary(args.hard_data)
    else:
        hard_boundary, hard_normal = val_boundary, val_normal
    model.add_exact(exact_planes(model, hard_boundary, hard_normal, x_scale, args.hard_supports, device))
    train_stage(
        "Hard-mining",
        model,
        train_boundary,
        val_boundary,
        x_scale,
        args.stage4_epochs,
        args,
        device,
        history,
        hard_boundary=hard_boundary,
    )

    architecture = {
        "width": args.width,
        "depth": args.depth,
        "pieces": args.pieces,
        "output_pieces": args.output_pieces,
        "tau": args.tau,
        "adaptive_pieces": int(model.adaptive_weight.shape[0]),
        "exact_pieces": int(model.exact_weight.shape[0]),
        "output_softplus": False,
    }
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "architecture": architecture,
        "x_scale": x_scale_np,
        "history": history,
        "training_seconds": time.time() - start,
        "source_data": str(Path(args.data).resolve()),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    output.with_suffix(".history.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"checkpoint": str(output.resolve()), **architecture}, indent=2))


if __name__ == "__main__":
    main()
