# Cavitation bubble above a rigid wall

`WALL-CAVITATION.FOR` is a flat-wall variant of the legacy bilge-opening
boundary-element solver. The original `BILGE-HOLE.FOR` is left unchanged.

## Model

- Axisymmetric, incompressible, inviscid and irrotational liquid.
- Initially spherical bubble of radius `R0D * RM`.
- Flat rigid wall at `z = 0` with zero normal velocity,
  `d(phi)/dn = 0`.
- The numerical wall is a finite disk. The supplied case extends to
  `10 R0`; increasing its radius and panel count should be used for a
  wall-truncation convergence check.
- Gravity and surface tension are omitted from the Bernoulli update in this
  simple inertial-collapse case.
- The run stops when the two axial bubble endpoints approach within
  `CRITER`, before jet impact/topology change. It does not model the
  post-impact toroidal phase.

## Build and run

```sh
gfortran -std=legacy -ffixed-line-length-none \
  -fallow-argument-mismatch -fno-align-commons -O2 \
  WALL-CAVITATION.FOR -o wall-cavitation

mkdir -p wall-output
cp WALL-DADE.TXT wall-output/WALL-DADE.TXT
(cd wall-output && ../wall-cavitation)
python3 make_wall_movie.py
```

The solver writes `wall-output/WALL-CAVITATION.plt` and scalar histories.
The Python script creates `wall-output/wall-cavitation.avi`.

## Input

The first record of `WALL-DADE.TXT` is:

```text
M NWALL RM R0D PC PINF D T ADFD HPRIME WALLR P1INF EPS NMTE CRITER
```

Key fields:

- `M`: bubble panels.
- `NWALL`: one-sided wall panels; the output mirrors them about the axis.
- `RM`: physical maximum-radius scale.
- `R0D`: initial bubble radius divided by `RM` (normally `1`).
- `HPRIME`: initial center height divided by `RM`; it must exceed `R0D`.
- `WALLR`: numerical wall radius divided by `RM`.
- `ADFD`: adaptive time-step scale.
- `EPS`: initial non-condensable-gas pressure fraction.
- `NMTE`: number of time steps between smoothing operations.
- `CRITER`: axial endpoint-gap stopping tolerance, normalized by `RM`.

The remaining six records contain 12-, 10-, and 16-point Gauss-Legendre
nodes and weights used by the legacy integration routines.

## Supplied demonstration

The supplied case uses 40 bubble panels, 80 one-sided wall panels, and a
stand-off ratio `gamma = HPRIME / R0D = 1.5`. It reaches the axis-jet stop
after 317 accepted steps. The bubble volume centroid moves from `1.500 R0`
to `1.128 R0`, showing migration toward the wall.
