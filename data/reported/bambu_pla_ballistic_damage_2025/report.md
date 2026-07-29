# Bambu PLA Basic ballistic/damage reference

## Measured evidence

The peer-reviewed source tested five tensile and five flexural specimens made
from black Bambu PLA Basic on a Bambu X1C. Table 4 reports 34.1 MPa tensile
strength, 2600 MPa tensile modulus, 8.8% tensile breaking elongation, and
78 MPa flexural strength. The official high-resolution primary-source PNG
contains four independently separable colored specimen traces for each
response. Their average digitized peaks cross-check the reported five-specimen
averages to +0.79% for tensile
and -0.40% for flexure.
Although the article reports five tested specimens, a fifth distinct trace is
not independently visible in the rendering; it may be fully overplotted or
omitted and is not fabricated by the importer. The continuity gate records
every bridged source-column gap and limits the largest to 0.5 percentage
points of engineering strain.

## Author damage model

The source's Abaqus model uses 50 MPa yield, hardening to 60 MPa at 5%
equivalent plastic strain, damage initiation at 5%, linear energy-regularized
softening with 2.5 kJ/m2 fracture energy, and element deletion. The exported
0.5, 1.0, and 2.0 mm crack-band material-point curves integrate to the same
2.5 N/mm softening energy.

This is useful model evidence, but not direct material identification. The
measured quasi-static specimens yielded at about 35 MPa with little hardening;
50 MPa was selected for the impact model. The paper invokes strain-rate
effects to rationalize that choice but later states that rate hardening was
not actually included. The damage parameters are literature-grounded and
calibrated, not measured in matched exact-brand fracture coupons.

## Validation boundary

The author FE model predicts 184 mm penetration versus 169.2 mm measured for
the grid target, about 9% high, with a qualitatively similar cavity. It uses a
homogenized solid block rather than explicit 50% infill. Therefore this model
is a bounded high-rate damage sensitivity and penetration reference, not a
validated universal sparse-infill fracture law.
