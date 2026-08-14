# Model card: `smicnn_12d_v1_no_softplus_no_safety_head.pt`

## Intended use

The model approximates the Minkowski gauge of a discretized 12-D lamination-
parameter feasible domain and supplies a smooth convex inequality for continuous
gradient-based structural optimization.

## Architecture

| Item | Value |
|---|---:|
| Input dimension | 12 |
| Width | 2560 |
| Depth | 4 |
| Smooth-Max pieces per hidden unit | 8 |
| Deep output pieces | 32 |
| Direct affine supports | 1,466,303 |
| Smooth-Max parameter `tau` | 1e-4 |
| Final Softplus | No |
| Safety head | No |
| External hinge wrapper | No |
| Calibration | scale=1, shift=0 |

The deep hidden-to-hidden weights are constrained to be nonnegative by squared
parameterization. Consequently, `phi(V)` is convex in `V`. Smooth-Max uses a
normalized log-mean-exp operator, so `phi(0)` is numerically zero and the model
is near positively homogeneous over the validated working region.

## Reference discretization

- normalized thickness coordinate: `[-0.5, 0.5]`;
- thickness intervals: `N_z=64`;
- candidate angles: `N_theta=181` uniformly spanning `[-90°, 90°]`;
- label solver: exact epigraph linear programming (`HiGHS dual simplex`).

## Independent test evidence

On three independent one-million-ray data sets, removing the final Softplus did
not change boundary values because all tested boundary outputs were above the
Softplus linear threshold. The no-Softplus base model gave approximately:

| Metric | Test V | Test VI | Test VII |
|---|---:|---:|---:|
| BT MSE | 5.0908e-5 | 5.1042e-5 | 5.1035e-5 |
| BT bias | 7.4461e-4 | 7.4599e-4 | 7.2869e-4 |
| BT p99 | 0.023053 | 0.023021 | 0.023148 |
| BT max | 0.055935 | 0.065183 | 0.058050 |

The original no-hinge evaluation also found zero non-finite, tiny, or non-
positive-radial gradients in the sampled protocols. These empirical results do
not extend to a proof over all continuous directions.

## Deployment guidance

Use `g_rho(V)=phi(V)-rho<=0`. The choice of `rho` is a deployment policy, not a
trained parameter. It trades usable design volume against false-feasible risk.
Random-ray tests can support a probabilistic statement for their fixed sampling
distribution, but optimizer-induced directions are not i.i.d. random rays.

Recommended engineering workflow:

1. choose `rho` on an independent validation distribution;
2. report false-feasible counts and the one-sided 99% confidence upper bound;
3. run optimizer-induced directional attacks;
4. verify the final design with the exact StrictLP oracle.

## Known limitations

- `rho=0.98` was not uniformly safe under all later optimizer-induced attack
  protocols, although it performed well on the preregistered random-ray tests.
- More conservative thresholds improved safety but reduced feasible volume.
- At ordinary independent boundary points, automatic-differentiation and
  finite-difference gradients agreed well at `h=1e-4`; at some optimizer
  endpoints, the finite-difference tail was larger because of local high
  curvature/support switching.
- The model represents the `N_z=64`, `N_theta=181` discretized reference domain,
  not the exact continuous-angle domain.
- No claim is made that finite tests certify global nonnegativity or safety on
  all of `R^12`.

## Integrity

The release checkpoint contains only the frozen base network and input scale.
Historical optimizer state, the optional positive homogeneous safety head,
threshold-hinge parameters, and final output Softplus are excluded. Numerical
conversion from the archived base checkpoint was verified bit-for-bit on an
independent random input batch (`max_abs_difference=0`).

