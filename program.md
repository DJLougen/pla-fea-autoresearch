# Autoresearch contract

The research loop seeks the simplest admissible linear-elastic material law that
generalizes from calibration coupons to validation coupons and a completely
held-out component.

## Agent-editable surface

An automated agent may edit only `configs/research_space.yaml` and create new
experiment YAML files under `configs/proposals/`. A proposal may change the
model class, parameter tying, starting priors and bounds, optimizer, mesh
schedule, curve weighting, or regularization strength.

## Fixed surface

The agent must not edit `src/prepare.py`, `src/metrics.py`,
`src/validation.py`, `src/evaluate.py`, or `src/results.py`; processed
partitions; held-out specimens; unit conversion; failure/invalidity penalties;
solver timeout enforcement; or the append-only logging schema.

Every hypothesis has a fixed numerical budget, is scored on validation rather
than calibration alone, and is accepted only when its complexity-adjusted
validation score beats the incumbent. Held-out component data are reported
after selection and never drive selection.

The fixed axial/force curve score uses shape, initial slope, peak, and
location-at-peak terms. If a tensile specimen includes measured transverse
strain, 20% of that specimen's score is reserved for the relative error in its
transverse-versus-axial elastic slope. Missing transverse strain is never
fabricated and leaves the original axial score unchanged.

Nonlinear plasticity, damage, fracture, and cohesive laws are out of scope until
this complete elastic workflow has been validated against real experiments.
