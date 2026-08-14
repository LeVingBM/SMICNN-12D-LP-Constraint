# Data generation and NPZ schema

## Seven ray sources

| Source | Definition | Purpose |
|---|---|---|
| Dense random | Unit-normalized Gaussian vectors in 12-D | Global directional coverage |
| Sparse random | Rays with 2-4 nonzero coordinates | Low-order interactions and axis-adjacent regions |
| Coordinate axes | Positive/negative canonical basis vectors | Coordinate extrema |
| 66 coordinate planes | Angular sweeps over all coordinate pairs | Complete fixed 2-D slice coverage |
| Random coordinate pairs | Random directions in randomly selected coordinate planes | De-bias fixed angular grids |
| Random 2-D planes | Angular sweeps in random orthonormal 12-D planes | Off-axis geometric generalization |
| Support-atom projected | Exposed atomic points induced by random support normals | High-curvature/support-critical directions |

## StrictLP label

For a unit ray `d`, the epigraph LP evaluates the polar support
`gamma(d)=max_c c^T d` subject to layerwise atomic support constraints. The true
radial boundary is `V_boundary=d/gamma(d)`, and the optimal polar vector `c`
satisfies `c^T V_boundary=1`. It can therefore be reused as an exact affine
support plane.

## NPZ fields

| Field | Shape | Meaning |
|---|---|---|
| `V_boundary` | `[N,12]` | StrictLP boundary points |
| `rays_12d` | `[N,12]` | Unit ray directions |
| `r_max` | `[N]` | Boundary radii |
| `gamma` | `[N]` | Reference gauge on unit rays |
| `support_normal` | `[N,12]` | Optimal polar support normals |
| `lp_violation` | `[N]` | Maximum LP inequality violation |
| `source_id` | `[N]` | Integer ray-source identifier |
| `source_names` | `[7]` | Identifier-to-name mapping |
| `slice_id` | `[N]` | Plane identifier, or `-1` |
| `slice_angle` | `[N]` | Polar angle for slice samples |
| `z_edges` | `[N_z+1]` | Thickness discretization |
| `theta_deg` | `[N_theta]` | Candidate angle grid |
| `meta_json` | scalar | Generation configuration and quality checks |

Training/validation/test splits should be made by ray, not by radial copies of
the same boundary point. Blind test seeds must not be reused for hard mining,
threshold selection, or checkpoint selection.

