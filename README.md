# PLA FEA Autoresearch

> **Research preview:** this project is open for replication, measured-data
> contributions, and attempts to falsify its predictions. It is not a
> certified material model and must not be used for safety-critical design.

[![Tests](https://github.com/DJLougen/pla-fea-autoresearch/actions/workflows/tests.yml/badge.svg)](https://github.com/DJLougen/pla-fea-autoresearch/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository calibrates the simplest useful **linear-elastic** constitutive
law for anisotropic FDM-printed PLA. CalculiX (`ccx`) is the production physics
simulator, measured tensile/flexural curves are ground truth, SciPy Differential
Evolution is the inner numerical optimizer, and a constrained experiment loop
compares isotropic, transversely isotropic, and fully orthotropic hypotheses.

The fixed inverse-calibration and model-selection core intentionally stops at
elasticity. A provenance-separated post-core workflow derives an exploratory
pre-peak isotropic plastic card from the public Bambu tensile curves; it does
not alter elastic model selection and does not infer damage, fracture, or
cohesive behavior.

## What this can do today

- Replay and compare ordinary and experimental infill patterns under matched
  mass, mesh, loading, and material assumptions.
- Generate printable pattern meshes and frozen physical-test capture sheets.
- Propagate uncertainty in published PLA properties, print direction, nozzle
  temperature, print speed, and dimensional tolerance.
- Keep published evidence, simulated evidence, and physical measurements
  provenance-separated.

The current evidence audit reports **34 proven, 25 partial, and 2 missing**
claims. The two missing items are physical measurements: six matched
graded-hourglass versus triangular bending curves and matched local-axis
calibration/validation curves. Until those exist, unusual infill rankings are
hypotheses to test—not proven real-world improvements.

## A useful first contribution

If you have a printer and a force gauge or load cell, start with the frozen
six-specimen bending comparison in
`results/custom_infill/graded_hourglass/first_physical_test_plan.csv`. Keep the
material lot, drying, slicer profile, bead orientation, dimensions, and test
fixture identical between patterns. See [CONTRIBUTING.md](CONTRIBUTING.md) for
the submission format and quality gates.

The repository includes extracted CSV curves from two CC BY 4.0 datasets.
Original workbooks, papers, images, CAD, generated solver runs, and G-code are
not committed; checksum-pinned importers and provenance records show how to
obtain or rebuild them.

## Architecture

1. `src.prepare` validates raw CSVs, converts force/displacement when geometry
   permits, preserves source hashes, and writes deterministic processed files.
2. `src.materials` expands model parameters into nine engineering constants,
   builds the compliance matrix, and rejects inadmissible candidates before a
   solver launch.
3. `src.calculix` renders isolated input decks, runs `ccx` with a hard timeout,
   preserves failures, and returns structured statuses.
4. `src.evaluate` is the fixed scoring path. It compares full curves on a common
   normalized abscissa, not just ultimate values.
5. `src.optimize` calibrates continuous parameters with cached, resumable,
   reproducible Differential Evolution.
6. `src.autoresearch` evaluates a deterministic hypothesis list, uses
   complexity-adjusted validation error for selection, and never selects on
   calibration or held-out-component error.
7. `src.report` summarizes the accepted model, expanded constants, bounds,
   errors, failures, evaluation counts, and complexity tradeoff.

The production path is:

```text
CSV curves → deterministic processing → candidate material law
           → CalculiX solid model → parsed curve → fixed validation score
           → constrained model-class comparison → held-out report
```

## Installation

Python 3.9 or newer is supported.

```bash
cd pla-fea-autoresearch
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Install CalculiX separately and confirm that `ccx` is on `PATH`:

```bash
ccx -v
```

Alternatively, set `solver.executable` in the production YAML files to an
absolute executable path. `src.calculix.find_ccx` accepts either an explicit
path or a command name, and automatically detects Homebrew-style versioned
executables such as `ccx_2.23`.

The optional 3D infill fidelity workflow uses Gmsh and MeshIO:

```bash
uv sync --extra dev --extra solid
```

## Coordinate and unit conventions

All analysis uses MPa, N, mm, and tonnes. Density is
`1.24e-9 tonnes/mm³`. Since `1 N/mm² = 1 MPa`, no hidden stress conversion is
performed.

Material coordinates are independent of global CAD axes:

- local 1: extrusion/bead direction;
- local 2: in-layer direction transverse to the bead;
- local 3: build direction/layer normal.

Every solid section uses `*ORIENTATION`. The tensile templates map global
coupon loading to local direction 1, 2, or 3 independently. Do not remove that
mapping when replacing the coarse meshes with production meshes.

## Data schema

Each CSV represents one specimen. Column aliases without unit suffixes (such as
`Force`, `Displacement`, `width`) are accepted, but values must use the fixed
units below.

| Field | Unit | Requirement |
|---|---|---|
| `force_N`, `displacement_mm` | N, mm | Required together if stress/strain are absent |
| `engineering_stress_MPa`, `engineering_strain` | MPa, dimensionless | Required together if force/displacement are absent |
| `transverse_strain` | dimensionless | Optional for tensile curves; when supplied it enters the fixed Poisson-sensitive score |
| `specimen_width_mm`, `specimen_thickness_mm`, `gauge_length_mm` | mm | Required for tensile force/displacement conversion |
| `test_id` | — | Recommended; filename stem is the fallback |
| `partition` | — | `calibration`, `validation`, or `heldout`; defaults to calibration |
| `test_type` | — | `tensile`, `flexural`, or `component` |
| `print_direction` | — | local `1`, `2`, `3`, off-axis `12_45`, `13_45`, `23_45`, or `mixed` |
| time, raster angle, layer height, nozzle temperature, print speed, infill | documented column units | Optional and preserved |

The preprocessor removes duplicate rows, sorts stably by strain, rejects
obvious unit mistakes, and never fabricates a missing measurement. Raw files
are not edited. The manifest stores their absolute paths and SHA-256 hashes.
Real flexural data should ideally include already reduced flexural
stress/strain or retain force/displacement for direct FEA comparison.

Add real data by copying one CSV per specimen to `data/raw/`, explicitly
assigning partitions, then run:

```bash
python -m src.prepare --input data/raw --output data/processed
```

Never tune partition labels after seeing model results. A component geometry
must appear only in the `heldout` partition.

## Downloaded public reference curves

Two checksum-pinned importers are included:

```bash
python scripts/import_mendeley_8jxw5533py.py
python scripts/import_zenodo_16935914.py
```

The Mendeley source is the closer material match: Bambu PLA Basic printed on a
Bambu X1C. It yields 22 literal raw tensile curve blocks. Its article claims 21
specimens, however, and the raw block labels/counts conflict with both that
claim and several same-label UTS values in the summary workbook. The Zenodo
source yields 72 additional printed-PLA tensile curves, but does not identify
the filament as Bambu PLA Basic.

Run the immutable source reconciliation with:

```bash
python -m src.mendeley_tensile_reconciliation_audit
```

The result is a bounded negative finding, not a guessed repair. All 21 summary
UTS cells can be found among the literal raw peaks, but only 12 point to the raw
curve carrying the same label. The summary reuses only 13 of the 22 raw curves
and omits nine. Consequently, no one-to-one relabeling or directional
calibration partition is released.

Both imports stay under `data/public`, carry source URLs, hashes, licenses, and
quality flags, and remain deliberately `partition=unassigned`. Neither source
provides the complete local-axis mapping, flexural curves, or held-out component
response required by this repository's calibration contract. Review their
dataset-specific READMEs and manifests before promoting any curve to
`data/raw`.

A separate checksum-pinned importer captures published Bambu PLA Basic
three-point-bending evidence at 220 °C:

```bash
uv sync --extra digitization
python -m src.bambu_flexural_digitization
python -m src.published_flexural_validation
```

It transcribes the exact Table 3 modulus/strength means and digitizes the three
visible Figure 4 curves for 0°, 45°, and 90° raster orientations. The image
extraction cross-checks all three curves against the table. A matched
100 × 12.7 × 3.2 mm, 64 mm-span CalculiX model then compares the existing
material card to the reported bending moduli without refitting it and records a
three-level mesh check. Because the paper does not identify each plotted curve
as a mean, median, or representative specimen and does not publish its raw
machine files, these data are external validation only. They are prohibited
from entering calibration, model selection, or fracture-law fitting.

The paper's complete exact Table 3 flexural process matrix can also be
validated and applied to the saved bending fields:

```bash
python -m src.bambu_flexural_process_reference
python -m src.flexural_process_strength_screen
```

The importer checksum-pins Figures 5 and 6 and independently checks all twelve
exact nozzle-temperature/orientation modulus and strength rows. The screen
uses the published 0°/45°/90° strengths and coefficients of variation to
evaluate in-plane principal tensile stress in eight saved bending fields. A
scaled TDS build-Z value is retained as an explicit proxy because the paper
does not publish a build-direction process matrix. This is a bending-reference
screen, not a fracture, tensile, compression, shear, or final-load law.

The same primary source also provides Bambu-specific DMA summary points:

```bash
python -m src.bambu_dma_reference
python -m src.bambu_dynamic_profile
python -m src.dma_storage_sensitivity
```

Figures 7 and 9 independently cross-check the 0°/45°/90° storage moduli and
tan-delta peak temperatures for the 220 °C print condition. The generated
orthotropic sensitivity uses the published 30 °C, 1 Hz storage moduli and
infers G12 from the 45° compliance with the separate exact-brand measured
in-plane ν12 = 0.3459. It is used only as an in-phase storage-stiffness proxy.
The paper does not publish loss
modulus, a frequency sweep, or raw temperature sweeps, so the workflow does
not claim damping or a frequency law from that source. The lowest digitized
DMA Tg, 62.50 °C, is an exclusion boundary—not an allowable service
temperature.

A second exact-brand open dataset supplies the missing small-strain wave
window:

```bash
python -m src.bambu_viscoelastic_wave_reference
python -m src.bambu_short_time_relaxation
python -m src.viscoelastic_pattern_modal_screen
python -m src.frequency_dependent_modal_screen
```

The importer checksum-pins 11 deposited files and reconstructs 85 raw DMA
points at an 18 °C reference temperature over 1 Hz–25 kHz. The study reports
printed density 1144 kg/m³, in-plane flat Poisson ratio 0.3459,
G′ = 714.7 MPa, and G″ = 26.6 MPa for Bambu-X1C PLA Basic samples. Its
interactive outlier clicks are not recoverable, so the published
representative values and the no-click raw reconstruction are stored
separately.

The modal screen replays the frozen Orca toolpaths using an isotropic in-phase
storage proxy, `E′ = 2G′(1+ν)`, and matches the dominant out-of-plane mode by
effective modal mass. At the 80 × 80 × 20 voxel resolution, the triangular
and graded-hourglass modes are 1020.616 Hz and 1115.504 Hz, respectively:
a 1.0930× upward shift for the custom pattern. The 60→80 resolution changes
are 1.43% and 0.70%. The reported loss factor is
0.03722 (small-damping ratio estimate 0.01861). This is a bounded comparative
modal result, not a physically validated absolute resonance, forced-response,
fatigue, or arbitrary-temperature model.

The short-time relaxation command fits positive generalized-Maxwell spectra
with 3, 5, 7, and 9 branches to the checksum-pinned 18 C master curve. It uses
source-stratified five-fold validation and the one-standard-error rule, which
selects the five-branch model. A deterministic six-robust-MAD loss-factor rule
flags two extreme points and retains the other 83.

The storage-modulus fit passes its 3% normalized-RMSE gate; the loss-modulus
fit fails its 35% precision gate. The result is therefore limited to a
small-strain storage and relaxation sensitivity from 6.37 microseconds to
0.159 seconds at 18 C / 40% RH. Six frequency-specific in-phase CalculiX
elastic cards cover 1 Hz to 25 kHz. They are not a native CalculiX Prony
material, do not validate time-domain creep, and cannot be transferred to
arbitrary temperature, humidity, print settings, fatigue, or failure.

The frequency-dependent modal command closes the stiffness-frequency loop for
the triangular control and both promoted weird-kite patterns. It evaluates the
five-branch storage modulus at the current mode, runs the frozen sliced-path
CalculiX eigenproblem, and repeats until the material frequency and solved
mode agree within 0.1%. All 12 solves pass, every 60→80 mesh change is below
1.5%, and the selected out-of-plane mode remains mode 2 throughout. At the
fine mesh, the triangular, uniform-kite, and locally graded-kite modes are
1026.189, 1203.985, and 1190.961 Hz. The candidate/control ratios remain
1.1733 and 1.1606. This validates numerical self-consistency and relative
ranking only—not absolute resonance or damping.

The simulator also enforces the manufacturer service envelope rather than
silently extrapolating the room-temperature cards:

```bash
python -m src.bambu_service_envelope
```

The checksum-pinned Bambu PLA Basic TDS V3.0 reports 60 °C glass transition,
54 °C heat deflection at 1.8 MPa, 57 °C heat deflection at 0.45 MPa, and
0.43% saturated water absorption at 25 °C/55% RH. Short-term monotonic
20–30 °C cases, the exact 30 °C/1 Hz storage-stiffness point, and the exact
18 °C/1 Hz–25 kHz reduced-frequency wave window are admitted with explicit
restrictions. Warm cases below 54 °C are guardrail-only, and structural
extrapolation at or above 54 °C is rejected. Creep, fatigue, humidity
corrections, aging, and dynamic conditions outside the two measured windows
are rejected as unsupported.

The command exports a 90-specimen staged DMA, stress-relaxation, humidity,
directional fatigue, and controlled-aging plan that would turn the most
important rejections into calibrated behavior.
This gate is a hard boundary on the simulator; manufacturer Tg, HDT, and water
uptake are not misrepresented as temperature, lifetime, or moisture
constitutive laws.

Turn that plan into randomized, checksum-frozen capture files and validate a
submission with:

```bash
python -m src.service_law_campaign
python -m src.service_law_campaign --input path/to/filled-csvs
python -m src.service_law_calibration --input path/to/filled-csvs
```

The generated package contains 90 stage-specific templates and separate
release gates for DMA temperature-frequency transfer, 1 s-to-4 h stress
relaxation, humidity-conditioned tensile retention, directional R=0.1 fatigue,
and controlled aging. A stage unlocks only when every frozen specimen in that
stage passes. Fatigue runouts remain right-censored at 100,000 cycles instead
of being converted to invented failures. The current package is
`awaiting_measurements`: all 90 templates are recognized as planned but blank,
and no service law is released.

The calibration command is already wired for bounded log-quadratic DMA
interpolation, positive generalized-Maxwell relaxation spectra, matched
humidity/aging retention factors, and directional right-censored Basquin
fatigue fits. Synthetic recovery gates verify held-out DMA replicates, a
positive relaxation spectrum, mixed fatigue failures/runouts, and tensile
metric extraction. These checks prove the fitting machinery, not the missing
Bambu measurements: the current calibration registry releases 0/5 laws.

Build the canonical condition-aware material selector with:

```bash
python -m src.bambu_digital_twin
```

It consolidates nine ready-to-use CalculiX cards and a checksum inventory of
their evidence. Each requested analysis is routed to an exact-brand model,
a bounded sensitivity, or a hard exclusion. The selector supports defensible
room-temperature pattern studies—including static elasticity, in-plane
pre-peak tension, relative topology compression, flexural fields, interlayer
initiation, and the narrow published ballistic window—without pretending that
one PLA card covers every physical regime. Unsupported heat, humidity, creep,
fatigue, build-direction plasticity, and post-limit strain requests receive no
fallback material card.

The generated `results/bambu_pla_digital_twin/selection_matrix.csv` is the
quick reference for deciding whether a proposed infill experiment can be
simulated now, can be used only as a ranking sensitivity, or needs new
physical measurements first.

An additional exact-brand conditioning matrix can be generated with:

```bash
python -m src.bambu_cryogenic_reference
python -m src.cryogenic_pattern_sensitivity
```

The peer-reviewed source covers 320 Bambu PLA Basic specimens across four
raster angles, four cubic-infill levels, tensile/flexural loading, and
as-printed versus freeze/recovered conditions. It yields 160 exact chart
means and 80 paired conditioning effects. Treated specimens were kept at
-80 °C for 24 hours and then returned to room temperature for 24 hours before
testing, so the result is not a material law for a part operating at -80 °C.

For the graded-hourglass candidate, the observed 16-cell flexural-modulus
correction envelope is applied as a bounded ranking stress test around the
frozen-toolpath FEA. A common factor preserves the 1.2619 custom/control
ratio; even the deliberately adversarial assignment of the worst published
factor to the custom pattern and the best to the triangular control leaves a
1.0788 ratio. This supports a robust screening priority, while raw curves,
numeric dispersion, topology-specific conditioning, repeated cycling, and
absolute stiffness remain unvalidated.

The same command creates an 18-coupon physical handoff by reusing the frozen
six-object 3MF three times: direct as-printed, sealed room-temperature for
48 hours, and -80 °C/24 hours plus 24 hours of recovery. The additional
sealed room-temperature plate separates the freezer effect from the elapsed
48-hour storage effect missing from the publication. A test plan and one
force-displacement capture template per coupon are included.

The same figures also provide a complete manufacturing-process matrix for
210, 220, 230, and 240 °C nozzle settings:

```bash
python -m src.bambu_nozzle_temperature_profiles
python -m src.nozzle_temperature_sensitivity
```

This creates four admissible orthotropic storage profiles and runs triangular
and `sd_c2_o50` through all five solid load modes at each nozzle setting. Ten
of the twelve Figure 7/Figure 9 point pairs pass the strict crossplot
tolerance; the 210 °C and 230 °C 90° discrepancies remain flagged and those
profiles are exploratory. The workflow models changes in the as-printed
30 °C/1 Hz storage response, not the thermal response of a part in service,
extrusion flow, void formation, strength changes, or the safety of a printer
setting.

The canonical room directional card also uses the same checksum-pinned
exact-brand in-plane Poisson measurement:

```bash
python -m src.bambu_tds_profile
python -m src.static_poisson_propagation
```

Profile v2 replaces only the historical `nu12=0.33` prior with the measured
`nu12=0.3459`. `G12` is then derived under the existing in-plane-isotropy
hypothesis; it is not relabeled as a direct shear measurement. A 50-case
matched propagation check reruns five representative patterns under five load
modes with the old and new values. The full rank order is preserved and the
largest individual metric shift is 0.144%. `nu13`, `nu23`, `G13`, and `G23`
remain priors pending the directional campaign.

The literal Bambu curves can still support a provenance-separated elastic
sensitivity profile without guessing their missing local-axis mapping:

```bash
python -m src.bambu_curve_calibration
```

This fits every raw curve over a fixed 0.2–0.6% strain window, reconciles the
same-label summary values, compares isotropic and source-placement groupings
with BIC and leave-one-out error, and writes
`data/reported/bambu_pla_raw_curve_elastic.json`. The current data select the
isotropic model; placement labels are not promoted to orthotropic axes.

The Bambu TDS summary can also be encoded as two separate elastic sensitivity
profiles:

```bash
python -m src.bambu_tds_profile
python -m src.anisotropy_sensitivity
```

The paired 3D run holds E1 = E2 fixed at the reported 2580 MPa and compares an
isotropic assumption with the reported E3 = 2060 MPa engineering-constants
hypothesis. Local axes 1/2/3 are mapped to global X/Y/Z for the flat infill
coupons. Poisson ratios and out-of-plane shear moduli remain clearly labeled
initialization priors, so this is an anisotropy sensitivity—not a claimed full
orthotropic calibration.

For pre-peak material nonlinearity, build the separately identified plastic
sensitivity card:

```bash
python -m src.bambu_plastic_calibration
```

This excludes the three Z-upright curves, aligns the remaining 19 raw curves,
uses only the pre-peak region supported by at least 80% of specimens, applies a
0.2% offset yield definition, and converts the aggregate engineering curve to
a ten-point true-stress/equivalent-plastic-strain card. It writes
`data/reported/bambu_pla_raw_curve_prepeak_plastic.json` and a CalculiX
`*PLASTIC` include. The card is a monotonic in-layer tensile sensitivity model;
it is not a fracture law or a calibrated compression/interlayer model.

## Infill-pattern screening

The reported-property benchmark separates material behavior from structural
infill geometry. It fixes a provenance-tagged Bambu PLA elastic screening law,
generates equal-material beam models for several explicit patterns, runs them
through CalculiX, and writes matching OpenSCAD centerline coupons:

```bash
python -m src.infill_benchmark
```

The library includes grid, triangular, X-braced, checkerboard selective-X,
staggered-diamond, bow-tie, re-entrant, concentric-diamond, and sinusoidal
patterns, plus wall-connected 90° zigzag, 45° zigzag, and order-3 Hilbert
controls. Each equal-nominal-mass pattern is screened in tension, in-plane
shear, out-of-plane three-point bending,
geometrically nonlinear compression with a deterministic imperfection, and
linear eigenvalue buckling. Results are written to
`results/infill_benchmark/`, including force-displacement CSVs, solver decks,
pattern SVGs, printable `.scad` files, material cards, per-load rankings, and a
balanced multi-load index.

Use `--relative-density`, `--strain`, `--patterns`, and `--load-cases` to run
focused comparisons. For example:

```bash
python -m src.infill_benchmark \
  --relative-density 0.25 \
  --patterns x_braced triangular grid \
  --load-cases shear compression buckling
```

This remains a beam-level geometry screen, not a final strength prediction.
Keeping the material card fixed while changing only geometry makes comparisons
interpretable. A promising pattern should next receive a converged 3D solid
mesh, measured nonlinear material behavior, and matched physical coupon tests.

An exact-brand held-out check compares the three added controls with physical
Bambu PLA Basic results from a checksum-pinned 2026 Research Square preprint:

```bash
python -m src.adaptive_infill_external_validation
```

The physical study ranks Hilbert above both zigzags in tension and compression
at nominal 20% and 40% infill. The equal-material elastic beam and 3D solid
models reproduce the 90°-over-45° zigzag ordering but do not reproduce
Hilbert's peak-strength advantage. This negative result is retained rather
than fitted away. It limits the current simulator to elastic stiffness,
buckling, and hotspot screening; it does not validate ultimate load,
interlayer separation, or post-collapse energy. The source has one specimen
per configuration and does not publish complete geometry, G-code, raw curves,
or uncertainty.

New infill concepts can be supplied directly as centerline JSON rather than
added to the built-in Python library:

```bash
python -m src.custom_infill \
  --pattern ideas/graded_hourglass.json \
  --beam-only

python -m src.custom_infill \
  --pattern ideas/graded_hourglass.json \
  --mesh-size 0.9
```

The first command is the inexpensive five-load beam screen. The second
automatically promotes the same source-hashed graph to equal-actual-mass 3D
solids against triangular, performs a three-level bending mesh check, emits
exact-mass printable OpenSCAD, and writes a 30-coupon physical test matrix.
Every run must pass the canonical Bambu PLA digital-twin gate first: the
material profile and generated CalculiX card are checksum-verified, while
unsupported temperature, humidity, and conditioning requests stop before a
solver is launched. The output also includes `promotion_decision.json` and a
focused `first_physical_test_plan.csv`; the latter contains only the
three-replicate custom/control pairs needed to test the specialist advantage
found by the simulation.

Freeze the six specimen IDs and create one raw force-displacement template per
coupon with:

```bash
python -m src.custom_pattern_validation --initialize
```

Initialization also exports two binary STLs directly from the exact C3D6
volume meshes used in the CalculiX study. Both files must be watertight, retain
40 × 40 × 4 mm bounds, and match the planned Bambu PLA mass within 0.1%.
Their source-mesh and STL checksums are frozen into the capture manifest, so
the printed geometry cannot silently drift from the simulated geometry.

When OrcaSlicer is installed, initialization also creates a six-object X1
Carbon plate and slices it to audited G-code. The registered override uses
the official Bambu PLA Basic profile at 220 °C, a 55 °C textured plate,
0.20 mm layers, three walls, 100% rectilinear core, no supports or brim, and a
50 mm/s deposition-speed ceiling. The parser confirms three unscaled copies
of each topology, identical toolpath volume within replicate groups, and less
than 1% custom/control toolpath-mass imbalance. Official profile snapshots,
the experimental overrides, 3MF, G-code, and their checksums are retained
together.

Initialization then audits every positive-extrusion XY segment in that frozen
G-code by object, layer, slicer feature, line width, and bead angle. The
current plate deposits 92.94% of the triangular CAD volume and 93.76% of the
graded-hourglass CAD volume; the difference is 0.82 percentage points and the
custom/control deposited-mass imbalance is 0.88%. Because the admitted Bambu
card is isotropic within the XY print plane, the observed bead-orientation
difference does not change the present linear-elastic ranking. The audit
therefore admits only a restricted relative-stiffness comparison. Absolute
stiffness, bead-scale stress, interlayer failure, fracture, fatigue, and
post-collapse behavior remain outside this continuum model and require
physical data or a calibrated toolpath-resolved reanalysis.

The frozen bead paths are additionally replayed as a mass-calibrated
40/60/80-cell XY voxel sensitivity with all 20 physical layers retained.
Each element receives the admitted Bambu engineering constants scaled by its
fine-raster occupancy fraction. All 6 CalculiX cases converge; at the finest
80 × 80 × 20 level, the graded-hourglass/control bending ratio is 1.262,
compared with 1.198 in the continuum solids. The 60-to-80-cell stiffness
changes are 0.95% for triangular and 0.59% for graded hourglass, while the
ratio changes 1.52%. This confirms that the relative bending advantage
survives the frozen slicer morphology under the stated perfect-bond,
linear-mixture assumptions. It still does not replace the matched physical
coupon test.

The same initialization command exports a support base and loading-nose holder
for a three-point-bending fixture. Three 4.00 mm hardened-steel rods provide
the actual contact lines at a 40.00 mm span. The protocol traces those
locations and the centered 0.400 mm displacement directly to the CalculiX node
sets. A fixture-compliance calibration or local midpoint displacement sensor
is mandatory; submissions above 0.0006 mm/N fixture compliance are rejected.

The curves do not require a universal testing machine. A calibrated force
path can use either a load cell reporting newtons or a digital scale with at
least 50 N (about 5.1 kg) capacity. A dial indicator measures local midpoint
motion; actuator motion is also supported when fixture compliance has been
measured with a stiff bridge. Generate the six low-cost templates with:

```bash
python -m src.manual_bending_capture --initialize
```

Record the continuous 1.2 mm/min ramp at 10 Hz or faster, fill the instrument
calibration IDs and error bounds in every row, then convert the raw readings:

```bash
python -m src.manual_bending_capture \
  --input path/to/manual-capture \
  --output path/to/converted-submission
```

Gram readings are converted with standard gravity, while actuator
displacement is corrected by the measured force times fixture compliance.
The adapter refuses force capacity below 50 N, force uncertainty above
0.25 N, displacement uncertainty above 0.01 mm, displacement resolution worse
than 0.005 mm, sampling gaps longer than 0.1 s, and excessive fixture
compliance. It records source and converted checksums, never clips a bad
reading, and must pass the unchanged canonical six-coupon validator.

After testing, place the six filled CSVs in one directory and run:

```bash
python -m src.custom_pattern_validation --input path/to/submission
```

The importer rejects missing or altered specimen IDs, sparse or nonlinear
stiffness windows, excessive mass error, changed print controls, and test
conditions outside the Bambu PLA service gate. It reports replicate means,
mass-normalized stiffness, a deterministic bootstrap interval for the measured
custom/control ratio, and whether the predicted specialist advantage is
validated, indistinguishable, reversed, or directionally correct but in need
of recalibration. A shared stiffness-scale candidate is reported separately
and never used to hide topology-specific disagreement or silently rewrite the
canonical material card.

Crossing centerlines receive shared structural nodes automatically. The input
gate rejects disconnected graphs, ambiguous overlaps, internal dangling
struts, inadequate fixture boundaries, sub-minimum segments, and a nominal
equal-mass width below the declared printability floor. Acute joints are
reported for slicer inspection. See `ideas/README.md` for the schema.

The staggered-diamond family can be searched over cell count and alternating
row offset, then promoted automatically from beam screening to equal-mass 3D
solids:

```bash
python -m src.parametric_infill_search
python -m src.parametric_infill_search --nonlinear-only
```

The second command applies the raw-curve pre-peak plastic card to the promoted
challenger and triangular control at a conservative 0.8% global strain.

A broader weird-pattern search now varies all four mirrored kite-row apexes,
then carries only distinct trade-off leaders through solid, modal, slicer, and
frozen-toolpath gates:

```bash
python -m src.weird_infill_search
python -m src.weird_infill_toolpath_validation
python -m src.toolpath_first_limit_screen
python -m src.weird_infill_physical_validation
```

The search evaluates 82 kite variants in 415 equal-density beam cases,
promotes six to 35 equal-mass C3D6 load cases and seven continuum modal
solves, then refines the triangular control and two leading specialist
patterns at 0.7 mm. The best specialist is `kite_08_08_08_08`; after replaying
the exact sliced paths at 80 × 80 × 20 cells it retains 1.4084× triangular
bending stiffness and 1.1727× its dominant out-of-plane modal frequency. The
second candidate, `kite_08_08_12_08`, retains ratios of 1.3851 and 1.1601.
All 60→80 static and modal changes are below 1.5%, and both metrics select the
same winner.

The first-limit screen reruns all three frozen sliced patterns at 60 and
80-cell resolution with endpoint strain output. Because the voxel material
uses fill-fraction stiffness, it reconstructs solid-material local stress from
the saved strain field before applying the directional tensile/shear and
interlayer-initiation sensitivities. All six cases pass, and first-limit mesh
changes are at most 4.2%.

At the fine mesh, the conservative/optimistic p99 bending-limit bands are
1.119–1.274 mm for triangular, 1.033–1.253 mm for the uniform kite, and
1.043–1.269 mm for the locally graded kite. The corresponding uniform-kite
limit-load sensitivity is 91.0–110.4 N versus 70.0–79.7 N for triangular.
The registered 0.4 mm physical ramp lies below every conservative band. These
are linear test-planning limits, not crack-initiation or final-load
predictions; matched physical curves and crack-site observations remain
mandatory.

The manufacturing folder contains a registered nine-object X1 Carbon 3MF:
three copies each of triangular and the two candidates. The slicer gate
confirms the frozen Bambu PLA Basic profile and only 1.21% maximum difference
between pattern-mean deposited masses. Test the supported modal response
first, then reuse each coupon for the 0.4 mm bending comparison. The physical
intake module currently reports `awaiting_measurements`; it requires all nine
rows and never treats blank capture cells as passing data.

A separate mechanism-inspired concept tests whether a continuous center spine
and overlapping diagonal fans can carry a center bending load more efficiently
than triangular infill:

```bash
python -m src.custom_infill \
  --pattern ideas/bowtie_fan.json \
  --mesh-size 0.9
```

The `bowtie_fan` is deliberately a bending specialist, not a general-purpose
replacement. At equal CAD mass, its converged C3D6 bending ratio is 1.381
relative to triangular; tension, compression, shear, and buckling are weaker.
The exact frozen Orca toolpath was then replayed at 60, 80, and 120 XY cells
with all 20 printed layers. All six CalculiX cases pass and the finest
toolpath-aware ratio is 1.385; the 80-to-120 stiffness changes are 3.67% for
triangular and 0.09% for the bow-tie, with a 3.62% ratio change.

The slicer predicts 1.35% less deposited mass for the bow-tie than the control.
That difference is retained under an explicit 2% pre-test tolerance and the
physical importer still requires measured-mass normalization. The registered
six-coupon plate, G-code, fixture, and raw force-displacement templates are
under `results/custom_infill/bowtie_fan/physical_validation/`. The package is
`awaiting_measurements`; no simulated value is treated as a physical result.

The printable finalists can also be challenged together against structured
manufacturing and orientation perturbations:

```bash
python -m src.pattern_robustness_campaign
```

This runs 730 CalculiX beam cases across triangular, both promoted kite
variants, and `bowtie_fan`. The 146 variants cover all five load cases,
0°/90° pattern placement, ±10% bead width, and every removable interior
strut. Triangular remains the best general-purpose pattern. In the intended
bending orientation, the beam ratios are 1.623 for the uniform kite, 1.592
for the graded kite, and 1.565 for the bow-tie; their conservative all-load,
all-orientation scores remain below triangular.

A new 12-object 3MF puts three copies of all four finalists in one slicing
job. Its predicted pattern-mean mass spread is 2.56%. The exact combined
G-code is replayed at 60×60×20, 80×80×20, and 120×120×20 voxels. Absolute
stiffness is deliberately not admitted because the two kite stiffnesses move
slightly over 5% from 80 to 120. The matched candidate/control ratios change
by at most 3.63%, however, and every non-control remains above triangular at
all three resolutions, so the complete candidate set—not an individual
winner—is admitted for physical testing.

After measuring the 12 coupons, fill
`results/pattern_robustness_campaign/physical_validation/measurement_capture.csv`
and run:

```bash
python -m src.pattern_robustness_campaign --evaluate-only
```

The importer requires all masses, ambient conditions, and bending stiffnesses,
limits group CV to 10%, and requires measured candidate/control ratios to
agree with the frozen-toolpath predictions within 20%. Blank rows remain
`awaiting_measurements`.

Saved 3D stress fields can be screened in local material axes against the
reported Bambu XY and Z-upright tensile references:

```bash
python -m src.directional_strength_audit
```

This reports peak and node-sampled p99 positive normal-stress utilization for
local 1, 2, and 3. It deliberately does not score compression or shear because
matching directional strengths are absent, and crossing 1.0 is not interpreted
as crack initiation or a predicted failure load. The generated report is under
`results/directional_strength_screening/`.

A second screen includes all three local shear components between two explicit
theory bookends:

```bash
python -m src.multiaxial_failure_sensitivity
```

The conservative bookend uses
`Sij=min(Xti,Xtj)/sqrt(3)` and the optimistic bookend uses
`Sij=min(Xti,Xtj)`. Positive normal and absolute shear terms interact
quadratically. These are not measured Bambu shear strengths or confidence
limits. Normal compression, crack growth, delamination, and post-peak behavior
remain outside the criterion. The workflow screens the parametric and
source-hashed custom-pattern fields together and writes a ranked,
three-replicate physical validation plan under
`results/multiaxial_failure_sensitivity/`.

A peer-reviewed 2026 thin-wall study adds measured, process-conditioned
interface initiation strengths for Bambu PLA Basic:

```bash
python -m src.bambu_interlayer_reference
python -m src.interlayer_initiation_screen
```

The reference retains both the traditional decomposition-seam profile
(35.07 +/- 2.55 MPa vertical interlayer tension; 13.88 +/- 2.98 MPa apparent
tear) and the continuous-layer SMLP profile (47.49 +/- 1.16 MPa; 14.15 +/-
3.17 MPa). The screen combines positive local-3 opening with resultant
local-23/local-31 tear and propagates one-standard-deviation lower strengths
through 42 saved static fields. It uses the measured tear response as an
interface proxy, not as a pure-shear coupon.

This advances initiation screening but does not create a cohesive law. The
paper supplies no critical separation, traction-separation curve, fracture
energy, unloading law, or post-peak data. The source also used a modified
dual-extruder machine, 208 C nozzle, 0.3 mm layers, and single-bead walls, so
both process profiles are sensitivity bookends rather than universal material
constants.

A separate peer-reviewed 2025 Bambu A1 mini study supplies eight exact-brand
print-process groups (30/500 mm/s, X/Y build-plate placement, with/without
ironing), each tested five times:

```bash
python -m src.bambu_print_speed_reference
python -m src.print_speed_infill_sensitivity
```

The checksum-pinned reference digitizes the plotted tensile, Young's modulus,
and impact mean +/- one-standard-deviation bands. It also extracts all eight
representative Figure 8 post-peak traces. Each output curve labels its evidence
class point by point: the post-peak monotone envelope is digitized, the
pre-peak cubic is reconstructed from the plotted group mean modulus/strength
and peak strain, and the terminal zero-stress drop is an explicit numerical
assumption. All eight representative peaks cross-check the corresponding
group mean tensile strengths within 2%.

The sensitivity run applies those eight in-plane process profiles to 42 saved
static infill fields (336 field/profile evaluations). It is a conservative
way to ask whether an infill initiation ranking survives realistic process
scatter. It is not a sparse-infill damage law: the source used 100%
rectilinear ISO 527-2 coupons, did not release the five raw curves per group,
and supplies no speed-conditioned shear/compression strength, crack-growth
law, or fracture energy.

A peer-reviewed exact-brand ballistic study adds specimen-level post-peak
evidence, an author damage model, and three physical targets for each of four
infill patterns:

```bash
python -m src.bambu_ballistic_reference
python -m src.ballistic_pattern_validation
```

The importer checksum-pins the official 4262 × 1579 primary-source Figure 6
PNG and separates four colored tensile and four colored flexural traces. The
paper reports five tested specimens for each response, but a fifth distinct
trace is not independently visible in the high-resolution rendering; it may
be fully overplotted or omitted and is not invented. The four-curve peak
averages cross-check the reported 34.1 MPa tensile and 78 MPa flexural
five-specimen means to +0.79% and -0.40%, respectively. Every interpolated
source-column gap is recorded and the largest is below the fixed 0.5
percentage-point strain continuity gate.

The paper's author model uses 50 MPa impact yield, hardening to 60 MPa at 5%
plastic strain, damage initiation at 5%, linear softening with 2.5 kJ/m2
fracture energy, and deletion. Exported 0.5, 1.0, and 2.0 mm material-point
curves use crack-band regularization and preserve 2.5 N/mm softening energy.
These are author-calibrated model assumptions, not matched fracture-coupon
measurements: the quasi-static specimens yielded at about 35 MPa, and the
paper states that no explicit rate-hardening law was included.

The physical 50% infill ranking is honeycomb, cubic, grid, then gyroid. Mean
penetrations were 110.72, 161.52, 169.21, and greater than 200 mm,
respectively. Honeycomb remains 1.32 times better than grid on the exported
mass-normalized energy-per-depth measure. The paper's Abaqus result predicts
184 mm for a homogenized grid target versus 169.21 mm measured; it does not
mesh the internal grid and therefore cannot predict the four-pattern ranking
from geometry. The empirical velocity sensitivity is restricted to the
reported 354-368 m/s window and is not armor certification.

A separate 2026 exact-brand study contributes real held-out component curves:

```bash
python -m src.bambu_uav_arm_reference
```

It checksum-pins the CC BY paper, XML, geometry figures, and high-resolution
Figure 7 for four Bambu PLA Basic UAV-arm configurations: topology optimized,
honeycomb, central lightening hole, and solid baseline. Three physical
stress-strain traces per configuration are digitized, for twelve curves total.
Their group-mean peaks cross-check Table 3 within 0.35%. Overplotted color
segments remain explicit visibility gaps rather than being reconstructed.

These curves are locked `external_heldout_only` and cannot enter material
calibration or model selection. The authors state that machine data are
available on request, while reconstructed CAD/STEP is not public. The
repository therefore preserves the paper's E=3000 MPa, nu=0.35, 500 N Abaqus
comparison as provenance but does not claim an exact CalculiX geometry replay.

The exact-brand Huang and Zhang dataset provides a stronger geometry-level
check for small-strain dynamics:

```bash
python -m src.bambu_modal_component_validation
```

This checksum-verifies the source workbooks, exact Test2 STEP, and 2 mm / 1 mm
Abaqus solid meshes for a 4 x 50 x 125 mm cantilever. It replays the meshes in
CalculiX, classifies the five common bending/torsion modes, and compares the
frozen raw-curve isotropic profile with five measured frequencies from each of
three Bambu PLA Basic components. The 15 targets are never used for fitting or
profile selection. The frozen model passes the fixed held-out gate with 11.1%
relative RMSE and 12.2% maximum absolute relative error; converged CalculiX
frequencies agree with the authors' converged Abaqus values within 1.84%.

This is independent validation of exact geometry and small-strain modal
stiffness only. The printed specimens mix horizontal and vertical placements,
the source is modeled isotropically, and modal frequency is not a static
force-displacement curve. It therefore does not release local-axis constants
or validate infill failure.

The four ballistic pattern families can also be compared as explicit,
equal-actual-mass 3D cellular geometries rather than empirical pattern labels:

```bash
python -m src.volumetric_pattern_validation
```

This runs 24 CalculiX cases for idealized honeycomb, body-centered cubic,
alternating grid, and gyroid geometries: a 24/32/40-voxel convergence check,
three compression directions, and a common 5% pre-damage plastic-energy
screen. All cases solve, and the maximum 32-to-40-voxel stiffness change is
7.8%. The geometry-only screens reproduce none of the four complete physical
ballistic rankings (10/24 pairwise orders). That negative result is retained:
quasi-static stiffness or pre-damage compression energy is not a substitute
for projectile contact, rate response, fracture, and material deletion.

The same command exports four volume-matched, watertight STLs under
`results/volumetric_pattern_validation/printable_stl/` and a randomized
36-specimen compression plan covering four patterns, three loading axes, and
three repeats. The STL validation table records checked volume, mass, bounds,
triangle count, and non-manifold edges. These are printable isosurface
equivalents of the voxel models, not recovered Bambu Studio toolpaths. Use the
measured compression curves to accept or reject the topology proxy before
promoting a weird infill to more expensive impact or fracture testing.

An exact-brand quasi-static benchmark now makes that acceptance gate concrete:

```bash
python -m src.bambu_common_infill_reference
python -m src.common_infill_compression_validation
```

The checksum-pinned importer preserves the nine aggregate Figure 4
stress-strain curves and the nine three-replicate mass summaries from a
TechRxiv study of purple Bambu PLA Basic. The study tested 30 mm cubes with
grid, gyroid, and cubic infill at 10%, 20%, and 30% nominal density, for 27
physical compression tests in total. The source is a preprint and does not
release the raw specimen curves.

The matched benchmark runs 18 CalculiX cases at 32 and 40 voxels, uses each
condition's measured total mass, loads the build-axis analog, and keeps the
solid material card fixed. Both meshes reproduce the reported
`grid > gyroid > cubic` initial-stiffness ranking at all three densities;
the nine-condition Spearman correlation is 0.933. The worst stiffness mesh
change is 20.2%, so this is a stable ranking result, not a precision modulus
result.

The publication has an internal factor-of-1,000 unit conflict. Read literally,
the raw simulations are 177–1040 times stiffer than the plotted Pa values.
But multiplying the plotted stress by 1,000 makes all nine digitized peaks
match the paper's separately stated MPa yield strengths within 4%; under that
sensitivity, the simulation/source modulus ratio is 0.177–1.040. Both
interpretations are exported, and neither is silently selected as truth.

A single scale fitted only at 20% has 32.1% mean error on the held-out
10%/30% conditions, so use this evidence to screen relative quasi-static
compression rankings only. The model still omits the source coupons' 0.42 mm
outer walls, exact slicer toolpaths, wall/infill interfaces, crushing contact,
and failure.

An independent peer-reviewed metamaterial study adds a second exact-brand
check and a direct route to unusual cellular ideas:

```bash
python -m src.bambu_pvh_reference
python -m src.pvh_material_transfer_sensitivity
```

The checksum-pinned CC BY supplement to *Bio-inspired perturbated
hierarchical mechanical metamaterial for energy absorption* uses Bambu PLA
Basic on a Bambu X1 Carbon. Its three tensile coupons give 2140 MPa modulus
and 36 MPa 0.2%-offset yield. Its 15 x 15-cell, 20%-density hierarchical
honeycomb validation specimen uses 1.0 mm vertex hexagons, 0.5 mm random
perturbations, and nominal 0.4 mm walls. The digitized author
simulation-versus-test curve differs by about -8% at the first collapse peak
and -1.7% in energy through 45% strain. Six separately reported
optimum/baseline modulus and strength comparisons have 1.9%-16.7% FEA-to-test
error.

An exact replay is intentionally blocked: the paper does not release the
random perturbation directions or seed, CAD/Abaqus files, or the post-yield
PLA hardening table. Those inputs control collapse order, so substituting
invented values would only create a similar-looking structure.

Instead, the independent 2140 MPa/36 MPa material card is transferred onto
the four reproducible equal-mass explicit geometries. All eight plastic
CalculiX cases solve. Both the project's 19-curve Bambu card and the
independent paper's card rank 5%-compression energy as
`grid > gyroid > honeycomb > cubic` (Spearman 1.0); the independent card
changes absolute energy by only about 3.7%-5.1%. This shows the candidate
ordering is robust to these two material cards. It does not physically
validate the geometries; that remains the purpose of the exported randomized
36-coupon compression plan.

Material-envelope robustness can be reproduced as independent scenarios:

```bash
python -m src.material_uncertainty_sensitivity --prepare-only
python -m src.material_uncertainty_sensitivity --scenario measured_low
python -m src.material_uncertainty_sensitivity --scenario measured_high
python -m src.material_uncertainty_sensitivity --scenario shear_low
python -m src.material_uncertainty_sensitivity --scenario shear_high
python -m src.material_uncertainty_sensitivity --aggregate
```

The measured scenarios use the reported XY/Z modulus mean ±1 standard
deviation. The shear cases vary unmeasured G13/G23 priors by ±25%; they are
sensitivity bounds, not confidence intervals or probability distributions.

A broader directional interval can be propagated through all four printable
finalists:

```bash
python -m src.directional_interval_robustness
```

This creates 44 admissible cards: the canonical model, every one-parameter
endpoint, three joint anchors, and 24 deterministic Latin-hypercube samples.
`E1`, `E2`, and `E3` use the TDS mean ±1 standard deviation; measured
`nu12=0.3459` is fixed because its source supplies no uncertainty; `nu13`,
`nu23`, and all three static shear moduli use explicitly declared broad
sensitivity ranges. These cards are interval samples, not fitted materials,
confidence intervals, or probabilities.

The B31 ratios are exactly invariant across the interval, demonstrating that
the beam screen cannot resolve these constants and must not choose adverse
cards. All 44 cards therefore run through 880 coarse equal-mass C3D6 cases.
Six adverse cards advance to 120 refined cases. All solve, the maximum
coarse-to-refined candidate/control ratio change is 2.08%, triangular remains
the all-purpose winner, and `bowtie_fan` remains second in every sampled card.
Its refined adverse bending ratio is 1.333× triangular. This is not a proof of
the continuous global worst case and does not validate strength or fracture.

The literal exact-brand specimen scatter can also be propagated separately:

```bash
python -m src.raw_specimen_infill_uncertainty
```

This runs the four frozen printable finalists through all five load fixtures
using the pooled raw-curve isotropic profile, then performs direct CalculiX
checks at the minimum and maximum measured specimen moduli. Thirty solver cases
confirm proportional propagation to better than 0.001%. The 22 raw curves have
a 9.3% modulus coefficient of variation and empirical 5%–95% bounds of
2065–2563 MPa.

Within this pooled scalar-modulus ensemble, all load-case rankings remain
unchanged across 22/22 measured moduli. Triangular is the balanced winner; the
uniform kite is the bending specialist at 1.388× triangular. The exported
`prediction_intervals.csv` reports absolute stiffness and buckling intervals
for every finalist/load pair. This is empirical specimen coverage, not a
population confidence interval, and it does not represent directional
constants, print defects, damage, or environmental effects.

Measured CAD-to-print dimensional variation is propagated independently:

```bash
python -m src.bambu_print_tolerance_infill
```

The checksum-pinned source contains 25 Bambu PLA Basic calibration cubes
printed on the Bambu X1C, with 75 readings per axis. It reproduces the
published mean absolute deviations of 0.016, 0.056, and 0.020 mm in X, Y, and
Z. Both fixed-offset and proportional-scale transfers are retained because a
20 mm cube does not identify the correct size-scaling law.

All 200 beam bending cases pass. Three observed minimum/median/maximum ratio
scenarios advance to 60 C3D6 multi-load cases. Triangular remains the balanced
winner in every solid scenario, while the uniform-kite bending ratio remains
within 1.387–1.389× triangular. This covers whole-geometry dimensional error;
the separate bead-width and missing-strut campaign remains responsible for
local manufacturing defects.

## 3D solid infill validation

The leading beam-screen candidates can be remeshed as their actual printable
capsule unions. The workflow clips every coupon to exactly 40 × 40 × 4 mm,
solves each pattern's strut width to equal **actual** solid volume, meshes the
result with layered six-node continuum prisms, and runs a three-level mesh
convergence study:

```bash
python -m src.solid_infill
```

Outputs under `results/solid_infill_validation/` include Gmsh meshes, CalculiX
decks, corrected-mass OpenSCAD coupons, stiffness uncertainty from the reported
Bambu PLA modulus spread, and a nine-specimen matched tension-validation plan.
This is the preferred printable geometry for validating the beam ranking.

The converged 0.7 mm meshes can then be exercised under all five load modes:

```bash
python -m src.solid_multiload
python -m src.solid_buckling_convergence
```

Any screening command accepts `--properties`. To rerun the continuum matrix
with the robust raw-curve profile:

```bash
python -m src.solid_multiload \
  --properties data/reported/bambu_pla_raw_curve_elastic.json \
  --output results/solid_multiload_raw_calibrated \
  --runs-dir runs/solid_multiload_raw_calibrated
```

`results/solid_multiload/` contains the refined tension, shear, bending,
geometrically nonlinear compression, and unconstrained 3D eigenvalue-buckling
results. Unlike the planar beam screen, the solid model permits the physically
important out-of-plane buckling mode. The companion convergence command checks
that mode at 1.2, 0.9, and 0.7 mm mesh sizes and writes a 45-specimen,
three-replicate physical validation matrix using the corrected-mass OpenSCAD
coupons.

The same refined meshes can be taken into the supported pre-peak plastic range:

```bash
python -m src.nonlinear_infill
```

For each leading pattern this runs a matched incremental geometrically
nonlinear elastic reference and an elastic-plastic model, writes the full
force-displacement curve, detects yield from the first nonzero saved `PEEQ`
state, and reports how much of the calibrated hardening range was consumed.
The default 1.5% global-strain endpoint stays inside the source-supported
hardening range for all three current candidates.

To regenerate the requirement-by-requirement evidence inventory, including the
test suite:

```bash
python -m src.completion_audit
```

The audit deliberately distinguishes a runnable simulator from a fully
validated brand model and lists the real measurements still required.

Generate the randomized, pre-partitioned physical characterization campaign
needed to close those gaps with:

```bash
python -m src.characterization_campaign
```

It writes matching screening CAD, 43 specimen IDs, per-specimen capture CSVs,
and off-axis 12/13/23 tensile orientations that make the three shear moduli
observable. The held-out L-bracket remains blinded by construction. A
scientific campaign workbook is also available under
`outputs/bambu-pla-campaign/`.

Verify that the planned measurements can actually identify the complete
orthotropic card before printing:

```bash
python -m src.characterization_identifiability
```

This differentiates the exact fixed-evaluator equations for the six planned
tensile orientations. Their twelve axial-modulus and transverse-Poisson
observables produce a rank-9 sensitivity matrix for the nine constants
`E1/E2/E3`, `nu12/nu13/nu23`, and `G12/G13/G23`; its normalized condition
number is 9.33. Axial-only data have rank 6, principal-direction axial/lateral
data have rank 5, and removing either the in-plane or build-coupled off-axis
coupons also loses full rank. Therefore lateral strain and all three off-axis
orientations are mandatory, not optional extra data.

The exported linearized uncertainty forecast assumes five repeats, 3% modulus
noise, and 0.01 absolute Poisson noise. Its largest predicted relative
standard error is 5.0%, for `nu23`. Those are campaign-planning assumptions,
not measured confidence intervals.

After testing, place the completed per-specimen CSV files in one directory and
run the strict intake gate before fixed preprocessing:

```bash
python -m src.campaign_ingestion \
  --input results/characterization_campaign/capture_templates
```

The gate checks the campaign checksum, exact specimen set, immutable
partitions, repeated material/process/geometry metadata, usable point count,
monotonic range, duplicate rate, tensile lateral-strain coverage, and
acceptance by the fixed preprocessor. The generated blank templates are
deliberately rejected until real measurements replace their empty data cells.

Once all 43 files pass intake, calibrate and independently gate the complete
local-axis material card with:

```bash
python -m src.directional_material_calibration \
  --input path/to/filled-csvs
```

Only the frozen `r01-r03` tensile coupons enter the nine-constant fit.
The `r04-r05` coupons independently gate release using axial-modulus and
transverse-Poisson errors. A blank, incomplete, inadmissible, or
validation-failing submission emits no CalculiX card. If the coupon gate
passes, the card is frozen before the three L-brackets are solved; their
curves never enter fitting or model selection. The default blank-campaign run
also records a synthetic recovery check for all nine engineering constants,
but that check is software verification and not material evidence.

## Synthetic demonstration

CalculiX is not required for the smoke demonstration:

```bash
bash scripts/run_demo.sh
```

The demo uses `solver.mode: mock`, a deterministic elastic response generator
whose hidden constants differ from the priors. This backend exists only to
test preprocessing, optimization, caching, selection, logging, and reporting
before `ccx` or real experiments are available. It is not an FEA result and
must not be used for material characterization.

The single command writes:

- `results/results.tsv`: append-only record of every hypothesis;
- `results/best/material.inp`: best CalculiX material card;
- `results/best/config.yaml` and `result.json`: reproducibility artifacts;
- `results/report.md`: model-comparison report;
- `runs/<experiment>/checkpoint.json`: resumable candidate cache.

## Production workflow

After preparing real data, select the production configs (their solver mode is
`calculix`):

```bash
python -m src.evaluate --config configs/baseline.yaml
python -m src.optimize --config configs/baseline.yaml --budget 200
python -m src.optimize --config configs/transverse_isotropic.yaml --budget 300
python -m src.optimize --config configs/orthotropic.yaml --budget 450
```

For constrained model comparison, point `configs/research_space.yaml` at the
production configs and run:

```bash
python -m src.autoresearch \
  --program program.md \
  --research-space configs/research_space.yaml \
  --max-experiments 20

python -m src.report --results results/results.tsv
```

Candidate solver jobs have isolated directories. Timeout, non-convergence,
singularity, parse failures, stdout, and stderr are preserved. Missing `ccx`
returns a structured failure and a fixed penalty instead of crashing.

## Inverse calibration and scoring

The fixed per-curve loss includes normalized curve RMSE, initial-slope error,
peak force/stress error, and displacement/strain-at-peak error. When a tensile
curve includes at least two finite `transverse_strain` samples, 20% of that
curve's score is the relative error in transverse-versus-axial elastic slope;
the CalculiX path reads global-Y strain from the coupon field output. Per-test
scores are equally weighted by default. The calibration objective adds a
stated prior regularizer; model selection adds a fixed per-parameter complexity
penalty. Physical invalidity and solver failure have large fixed penalties.
Lower is always better.

Differential Evolution uses a recorded seed, bounded parameters, a maximum
unique-evaluation budget, early stopping, an atomic JSON checkpoint, and a
cache keyed by the parameter vector. Re-running the same configuration resumes
deterministically. `optimization.workers` may be increased when independent
solver capacity is safe. The optimizer interface deliberately rejects unknown
algorithms so future CMA-ES or Bayesian adapters can be added explicitly
without changing the evaluator.

The outer loop accepts a proposal only if its complexity-adjusted **validation**
score improves on the incumbent. Held-out component error is computed and
reported only after fitting; it never drives acceptance.

## Fixed and agent-editable files

`program.md` is the research contract. An automated agent may edit only
`configs/research_space.yaml` and add experiment definitions under
`configs/proposals/`. It may propose model class, parameter tying, priors and
bounds, optimizer adapter, mesh schedule, curve weighting, or regularization.

The agent may not alter data loading, processed partitions, held-out specimens,
unit conversion, `src.metrics`, `src.evaluate`, `src.validation`, failure
penalties, timeout behavior, seed, or the `src.results` schema. The loop checks
the data, solver, seed, and logging contract against the first hypothesis and
records preflight rejections.

## Tests

```bash
pytest
```

Tests cover unit and geometry conversion, deterministic preprocessing, template
rendering, material-card layout, compliance and positive-definiteness checks,
curve interpolation and metrics, append-only logging, timeout and failure
parsing, evaluation caching, local-axis directional and bounded tension-shear
strength screening, and an
end-to-end synthetic partition smoke test. The real-CalculiX integration test
skips cleanly when `ccx` is unavailable.

## Scientific limitations

A full orthotropic law is generally not identifiable from three axial curves
alone. Axial tests constrain `E1`, `E2`, and `E3` well, but Poisson ratios and
all shear moduli require lateral-strain, shear, torsion, off-axis, bending, or
multiaxial measurements. Regularized optimizer output is not proof of a unique
material law; inspect parameter sensitivity and uncertainty before reuse.

Tensile ultimate strength is **not** yield stress. The elastic
inverse-calibration core does not identify either yield or fracture. The
separate pre-peak plastic card uses a 0.2%-offset yield construction but still
does not identify fracture or post-peak damage. A bending modulus also must not
automatically replace an axial Young's modulus: bending can mix tension,
compression, shear, geometry error, and layer-interface effects.
