# Evaluation metrics

## Boundary fitting and classification

For boundary errors `e_i=phi(V_i^B)-1`, the validator reports BT MSE, signed BT
bias, absolute BT p95/p99, and BT max. Along radial samples `t V_i^B`, it reports
T MSE, Sign outside `|t-1|<=0.02`, and NearSign for
`0.02<|t-1|<=0.05`.

| Metric | Acceptance value |
|---|---:|
| BT MSE | <= 8e-5 |
| absolute BT bias | <= 0.002 |
| BT p99 | <= 0.025 |
| T MSE | <= 1e-4 |
| Sign | >= 0.99 |
| NearSign | >= 0.98 |

BT p95 and BT max are reported diagnostics without replacing the p99 criterion.

## Two-dimensional geometry

For fixed coordinate and random 2-D slices, the signed relative errors are

```text
e_R = 100 (R_pred - R_ref) / R_ref,
e_A = 100 (A_pred - A_ref) / A_ref.
```

The report contains signed bias, absolute MAE/p95/max, the outward-direction
fraction above the selected tolerance, and the number of slices containing at
least one outward direction.

## Deployment safety

An exterior point `gamma V^B` is false feasible when
`phi(gamma V^B)<=rho`. In addition to the observed rate `k/n`, the validator
reports the one-sided 99% Clopper-Pearson upper confidence bound.

| gamma | U99 limit |
|---:|---:|
| 1.001 | 0.5% |
| 1.002 | 0.3% |
| 1.005 | 0.1% |
| 1.010 | 0.05% |
| 1.020 | 0.005% |

## Gradient quality

`N_ig`, `N_mg`, and `N_ng` count non-finite gradients, gradients with norm at
most `1e-10`, and non-positive radial derivatives, respectively. Euler error is
`|V^T grad(phi)-phi(V)|`. `e_FD` is the relative L2 discrepancy between the
automatic-differentiation gradient and a full 12-D central finite-difference
gradient at `h=1e-4`.

| Metric | Acceptance value |
|---|---:|
| N_ig / N_mg / N_ng | 0 / 0 / 0 |
| Euler MAE | <= 0.05 |
| Euler p95 | <= 0.10 |
| Euler p99 | <= 0.10 |
| e_FD p99 | <= 0.01 |

