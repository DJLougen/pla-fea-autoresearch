# Exact-brand UAV-arm component reference

The peer-reviewed source prints Bambu PLA Basic UAV rotary-arm specimens in
four geometries: topology optimized, honeycomb, central lightening hole, and a
solid baseline. Figure 7 contains three physical stress-strain traces for each
geometry. All 12 visible traces are digitized without reconstructing
overplotted segments. Their geometry-level mean peak stresses cross-check
Table 3 within 0.35%.

This is valuable independent component-level evidence because the geometries
were not used to fit this repository's material cards. It is not an exact
simulation validation: the article says underlying data are available on
request, and neither machine-exported curves nor reconstructed CAD/STEP files
are public. The curves therefore remain `external_heldout_only` and cannot
enter material calibration or model selection.

The paper's own Abaqus comparison uses an isotropic E=3000 MPa, nu=0.35 card,
a 500 N hole-to-hole tensile load, and a 3.0/2.0/1.5/0.75 mm mesh schedule.
Those author settings are preserved as provenance, not promoted as the
canonical Bambu PLA constitutive law.
