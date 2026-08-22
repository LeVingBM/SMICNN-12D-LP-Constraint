#!/usr/bin/env python
"""Export a row-wise affine UINT8 storage checkpoint for FP32 inference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def encode_tensor(tensor: torch.Tensor) -> dict:
    if not tensor.is_floating_point() or tensor.ndim < 2 or tensor.numel() == 0:
        return {"kind": "plain", "values": tensor}
    flat = tensor.float().reshape(tensor.shape[0], -1)
    lower = flat.amin(dim=1)
    upper = flat.amax(dim=1)
    scale = (upper - lower) / 255.0
    constant = scale == 0
    safe_scale = torch.where(constant, torch.ones_like(scale), scale)
    values = torch.round((flat - lower[:, None]) / safe_scale[:, None]).clamp_(0, 255)
    values = values.to(torch.uint8)
    scale = torch.where(constant, torch.zeros_like(scale), scale)
    return {
        "kind": "rowwise-affine-uint8",
        "values": values,
        "scale": scale,
        "offset": lower,
        "shape": tuple(tensor.shape),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    source = torch.load(args.source, map_location="cpu", weights_only=False)
    if source.get("format_version") != "smicnn-public-v1":
        raise ValueError("Expected the verified smicnn-public-v1 FP32 checkpoint.")
    quantized_state = {
        name: encode_tensor(tensor) for name, tensor in source["model_state_dict"].items()
    }
    compact = {key: value for key, value in source.items() if key != "model_state_dict"}
    compact.update(
        {
            "format_version": "smicnn-public-v2-uint8-storage",
            "storage_format": "rowwise-affine-uint8",
            "inference_dtype": "float32",
            "quantized_state_dict": quantized_state,
            "reference_fp32_sha256": sha256(args.source),
            "notes": (
                "Frozen base SMICNN with row-wise affine UINT8 weight storage. "
                "Weights are dequantized to FP32 before inference. No Softplus, "
                "safety head, threshold hinge, or external calibration is included."
            ),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(compact, args.output)
    source_bytes = args.source.stat().st_size
    output_bytes = args.output.stat().st_size
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "bytes": output_bytes,
                "size_mib": output_bytes / (1 << 20),
                "reduction_percent": 100.0 * (1.0 - output_bytes / source_bytes),
                "sha256": sha256(args.output),
                "reference_fp32_sha256": compact["reference_fp32_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
