# Frozen model

The recommended compact checkpoint is distributed in GitHub release `v1.0.1`
because its size exceeds GitHub's 100 MiB limit for ordinary Git objects.

- Asset: `smicnn_12d_v1_compact_uint8_storage.pt`
- Download: <https://github.com/LeVingBM/SMICNN-12D-LP-Constraint/releases/download/v1.0.1/smicnn_12d_v1_compact_uint8_storage.pt>
- SHA-256: `30ef5e8ea0d657ed64e70c3ffbc798054c2bd79be1f691473c5734a6514784af`
- Size: 188,837,138 bytes (180.09 MiB)
- Reduction from FP32: 73.17%

The asset contains row-wise affine UINT8 weight storage, architecture,
`x_scale`, provenance, and format metadata. `src.checkpoint_io` reconstructs
FP32 tensors before inference. It contains no optimizer state, Softplus output
layer, threshold-hinge wrapper, or safety-head parameters.

The archival full-precision checkpoint remains in release `v1.0.0` for exact
reproduction of the original frozen weights.
