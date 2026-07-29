# Bambu PLA Basic broadband viscoelastic reference

The exact-brand source reports a printed density of
1144 kg/m3, Poisson ratio 0.3459,
and, at 18 C over
1-25000 Hz,
representative G'=714.7 MPa and
G''=26.6 MPa.

The checksum-pinned raw DMA files produce 85 reduced-frequency
points. Their no-click mean G' is
718.343 MPa
and mean G'' is
23.513 MPa.
The storage mean is within 1% of the reported representative value. The
deposited plotting script removes outliers by interactive clicking, so the
paper's cleaned loss value cannot be reproduced exactly; both versions are
retained rather than silently choosing one.

Quality gate: **passed**.

The CalculiX card is an in-phase isotropic storage proxy derived as
E'=2G'(1+nu). It is restricted to small-strain wave/modal sensitivity at
18 C. It is not a static shear, failure, creep, fatigue, moisture, aging, or
arbitrary-temperature law.
