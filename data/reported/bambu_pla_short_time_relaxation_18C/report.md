# Bambu PLA Basic short-time relaxation sensitivity

A positive 5-branch generalized-Maxwell
spectrum was selected from [3, 5, 7, 9] branches by
source-stratified five-fold cross-validation and the one-standard-error rule.
The deterministic robust gate excluded 2 of
85 raw points because their loss factors were more
than six robust MAD from the log-median.

The final storage-modulus normalized RMSE is
**1.88%** and passes the 3% gate.
The loss-modulus normalized RMSE is
**47.14%** and fails the 35% precision
gate. The raw loss scatter is retained; it is not fitted away or replaced by
the paper's representative value.

The law is therefore admitted only as a small-strain storage and relaxation
sensitivity at 18 C / 40% RH from
6.3662e-06 to
0.159155 s. Six frequency-specific
in-phase CalculiX elastic cards are exported. They do not create a native
CalculiX Prony law, and they do not unlock long-term creep, fatigue, failure,
moisture, aging, or temperature extrapolation.
