"""Load full-precision and compact SMICNN public checkpoints."""

from __future__ import annotations

from pathlib import Path

import torch

try:  # Support package imports and direct script execution.
    from .code2_train_smicnn import SmoothMaxICNN
except ImportError:
    from code2_train_smicnn import SmoothMaxICNN


def decode_state_dict(checkpoint: dict) -> dict[str, torch.Tensor]:
    """Return an FP32 state dict from any supported public storage format."""
    if "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if checkpoint.get("storage_format") != "rowwise-affine-uint8":
        raise ValueError("Unsupported SMICNN checkpoint storage format.")

    state: dict[str, torch.Tensor] = {}
    for name, entry in checkpoint["quantized_state_dict"].items():
        if entry["kind"] == "rowwise-affine-uint8":
            values = entry["values"].float()
            scale = entry["scale"].float().reshape(-1, 1)
            offset = entry["offset"].float().reshape(-1, 1)
            state[name] = (values * scale + offset).reshape(entry["shape"])
        elif entry["kind"] == "plain":
            state[name] = entry["values"]
        else:
            raise ValueError(f"Unsupported tensor encoding for {name!r}.")
    return state


def load_smicnn_checkpoint(path: str | Path, device: str | torch.device = "cpu"):
    """Load a public checkpoint and return ``(model, x_scale, metadata)``."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
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
    target = torch.device(device)
    model.to(target).eval()
    x_scale = torch.as_tensor(checkpoint["x_scale"], dtype=torch.float32, device=target)
    return model, x_scale, checkpoint
