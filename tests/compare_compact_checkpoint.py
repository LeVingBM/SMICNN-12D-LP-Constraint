#!/usr/bin/env python
"""Compare a compact storage checkpoint against the FP32 reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from src import SmoothMaxICNN  # noqa: E402
from src.checkpoint_io import decode_state_dict  # noqa: E402


def build(checkpoint: dict) -> SmoothMaxICNN:
    cfg = checkpoint["architecture"]
    model = SmoothMaxICNN(
        width=int(cfg["width"]),
        depth=int(cfg["depth"]),
        pieces=int(cfg["pieces"]),
        output_pieces=int(cfg["output_pieces"]),
        tau=float(cfg["tau"]),
        adaptive_pieces=int(cfg["adaptive_pieces"]),
        exact_pieces=int(cfg["exact_pieces"]),
    )
    model.load_state_dict(decode_state_dict(checkpoint), strict=True)
    return model.eval()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--compact", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    points = 2.0 * torch.rand(args.samples, 12, generator=generator) - 1.0

    outputs = []
    formats = []
    for path in (args.reference, args.compact):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        formats.append(checkpoint["format_version"])
        model = build(checkpoint).to(device)
        x_scale = torch.as_tensor(checkpoint["x_scale"], dtype=torch.float32, device=device)
        chunks = []
        with torch.no_grad():
            for start in range(0, len(points), args.batch_size):
                x = points[start : start + args.batch_size].to(device)
                chunks.append(model(x / x_scale).squeeze(1).cpu())
        outputs.append(torch.cat(chunks))
        del model, checkpoint
        if device.type == "cuda":
            torch.cuda.empty_cache()

    difference = (outputs[1] - outputs[0]).abs()
    relative = difference / outputs[0].abs().clamp_min(1e-8)
    report = {
        "reference_format": formats[0],
        "compact_format": formats[1],
        "samples": args.samples,
        "device": str(device),
        "max_abs_difference": float(difference.max()),
        "mean_abs_difference": float(difference.mean()),
        "p99_abs_difference": float(torch.quantile(difference, 0.99)),
        "max_relative_difference": float(relative.max()),
        "p99_relative_difference": float(torch.quantile(relative, 0.99)),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
