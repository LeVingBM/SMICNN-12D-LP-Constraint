# Frozen model

The FP32 checkpoint is distributed in the GitHub release `v1.0.0` because its
size exceeds GitHub's 100 MiB limit for ordinary Git objects.

- Asset: `smicnn_12d_v1_no_softplus_no_safety_head.pt`
- Download: <https://github.com/LeVingBM/SMICNN-12D-LP-Constraint/releases/download/v1.0.0/smicnn_12d_v1_no_softplus_no_safety_head.pt>
- SHA-256: `f693c22c4d818965bd3aae40188beb575c8ccb6571bea1420d40ef70bf474927`
- Size: 703,795,379 bytes

The asset contains `model_state_dict`, `architecture`, `x_scale`, provenance,
and format metadata. It contains no optimizer state, Softplus output layer,
threshold-hinge wrapper, or safety-head parameters.

