"""Minimal integrity test for the public no-Softplus/no-safety-head checkpoint."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from src import SmoothMaxICNN  # noqa: E402
from src.checkpoint_io import decode_state_dict  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPOSITORY_ROOT
        / "models"
        / "smicnn_12d_v1_compact_uint8_storage.pt",
    )
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
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
    model.load_state_dict(decode_state_dict(checkpoint), strict=True)
    model.eval()

    with torch.no_grad():
        phi_at_origin = float(model(torch.zeros(1, 12)).item())

    assert checkpoint["format_version"] in {
        "smicnn-public-v1",
        "smicnn-public-v1-fp16-storage",
        "smicnn-public-v2-uint8-storage",
    }
    assert architecture["output_softplus"] is False
    assert architecture["safety_head"] is False
    assert math.isfinite(phi_at_origin)
    assert abs(phi_at_origin) < 1e-6
    print(
        {
            "format_version": checkpoint["format_version"],
            "phi_at_origin": phi_at_origin,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "status": "passed",
        }
    )


if __name__ == "__main__":
    main()
