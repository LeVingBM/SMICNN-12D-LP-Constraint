#!/usr/bin/env python
"""Export the frozen research checkpoint to the public, head-free format.

The source checkpoint contains historical metadata and an optional safety head.
This exporter retains only the frozen base SMICNN, removes the final Softplus
from the deployment definition, and converts redundant fixed Smooth-Max biases
to the normalized log-mean-exp representation used by the public code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source = Path(args.source)
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    state = checkpoint["model_state_dict"]
    tau = float(checkpoint["tau"])
    direct_pieces = int(state["direct_x.weight"].shape[0])
    output_pieces = int(state["out_x.weight"].shape[0])
    width = int(state["input_smax_bias"].numel())
    pieces = int(state["input_linear.weight"].shape[0] // width)
    depth = 1 + len({key.split(".")[1] for key in state if key.startswith("layers.")})

    expected_hidden = -tau * math.log(pieces)
    hidden_biases = [state["input_smax_bias"]] + [
        state[f"layers.{index}.smax_bias"] for index in range(depth - 1)
    ]
    if any(float((bias - expected_hidden).abs().max()) > 1e-12 for bias in hidden_biases):
        raise ValueError("Hidden Smooth-Max biases are not fixed normalization constants.")
    expected_output = -tau * math.log(direct_pieces + output_pieces)
    if abs(float(state["output_smax_bias"]) - expected_output) > 1e-10:
        raise ValueError("Output Smooth-Max bias is not the expected normalization constant.")

    scale = float(torch.nn.functional.softplus(state["raw_cal_scale"]) + 1e-6)
    shift = float(state["cal_shift"])
    if abs(scale - 1.0) > 1e-7 or abs(shift) > 1e-12:
        raise ValueError("The public no-calibration format requires scale=1 and shift=0.")

    public_state = {
        "input_map.weight": state["input_linear.weight"],
        "output_z.raw_weight": state["out_z.raw_weight"],
        "output_x.weight": state["out_x.weight"],
        "adaptive_weight": state["direct_x.weight"],
        "exact_weight": torch.empty(0, 12, dtype=state["direct_x.weight"].dtype),
    }
    for index in range(depth - 1):
        public_state[f"layers.{index}.z_map.raw_weight"] = state[
            f"layers.{index}.z_pos.raw_weight"
        ]
        public_state[f"layers.{index}.x_map.weight"] = state[
            f"layers.{index}.x_linear.weight"
        ]

    public = {
        "format_version": "smicnn-public-v1",
        "model_state_dict": public_state,
        "architecture": {
            "width": width,
            "depth": depth,
            "pieces": pieces,
            "output_pieces": output_pieces,
            "tau": tau,
            "adaptive_pieces": direct_pieces,
            "exact_pieces": 0,
            "output_softplus": False,
            "safety_head": False,
            "calibration_scale": 1.0,
            "calibration_shift": 0.0,
        },
        "x_scale": np.asarray(checkpoint["x_scale"], dtype=np.float32).reshape(12),
        "training_epoch": int(checkpoint.get("epoch", -1)),
        "source_checkpoint_sha256": sha256(source),
        "notes": (
            "Frozen base SMICNN only. The optional safety head and final output "
            "Softplus are intentionally excluded."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(public, output)
    print(
        json.dumps(
            {
                "output": str(output.resolve()),
                "bytes": output.stat().st_size,
                "sha256": sha256(output),
                "architecture": public["architecture"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
