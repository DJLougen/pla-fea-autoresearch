from pathlib import Path

import pytest

from src.viscoelastic_pattern_modal_screen import (
    build_storage_profile,
    parse_modal_dat,
    select_out_of_plane_mode,
)


def test_storage_profile_is_consistent_with_reported_shear() -> None:
    profile = build_storage_profile()
    constants = profile["screening_engineering_constants"]
    assert constants["G12"] == pytest.approx(714.7)
    assert constants["E1"] == pytest.approx(
        2.0 * 714.7 * (1.0 + 0.3459)
    )
    assert profile["density_tonnes_mm3"] == pytest.approx(1.144e-9)


def test_modal_parser_and_z_mass_selection(tmp_path: Path) -> None:
    path = tmp_path / "model.dat"
    path.write_text(
        """
 E I G E N V A L U E   O U T P U T
 MODE NO    EIGENVALUE                       FREQUENCY
                                     REAL PART            IMAGINARY PART
                           (RAD/TIME)      (CYCLES/TIME     (RAD/TIME)
      1   0.1000000E+07   0.1000000E+04   0.1591549E+03   0.0000000E+00
      2   0.4000000E+07   0.2000000E+04   0.3183099E+03   0.0000000E+00
 P A R T I C I P A T I O N   F A C T O R S
 E F F E C T I V E   M O D A L   M A S S
MODE NO.   X-COMPONENT     Y-COMPONENT     Z-COMPONENT     X-ROTATION      Y-ROTATION      Z-ROTATION
      1   0.1E-06 0.1E-06 0.2E-06 0.1E-06 0.1E-06 0.1E-06
      2   0.1E-06 0.1E-06 0.8E-06 0.1E-06 0.1E-06 0.1E-06
 T O T A L   E F F E C T I V E   M A S S
""",
        encoding="utf-8",
    )
    modes = parse_modal_dat(path)
    assert len(modes) == 2
    selected = select_out_of_plane_mode(modes)
    assert selected["mode"] == 2
    assert selected["frequency_Hz"] == pytest.approx(318.3099)
    assert selected["z_mass_fraction_within_extracted_modes"] == pytest.approx(
        0.8
    )
