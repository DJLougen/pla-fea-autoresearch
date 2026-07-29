from pathlib import Path

import pytest

from src.parsing import parse_frd, parse_frd_scalar_max_history


def test_frd_keeps_latest_field_and_streams_scalar_history(tmp_path: Path):
    path = tmp_path / "model.frd"
    path.write_text(
        """
  100CL  101 2.50000E-02       2
 -4  PE          1    1
 -5  PE          1    1    0    0
 -1         1 0.00000E+00
 -1         2 1.00000E-04
 -3
  100CL  102 5.00000E-02       2
 -4  PE          1    1
 -5  PE          1    1    0    0
 -1         1 2.00000E-04
 -1         2 3.00000E-04
 -3
"""
    )
    latest = parse_frd(path)
    assert latest["frd_pe"].shape == (2, 1)
    assert latest["frd_pe"][-1, 0] == pytest.approx(3.0e-4)
    history = parse_frd_scalar_max_history(path, "PE")
    assert history == pytest.approx({0.025: 1.0e-4, 0.05: 3.0e-4})


def test_frd_ignores_empty_field_blocks(tmp_path: Path):
    path = tmp_path / "model.frd"
    path.write_text(
        """
 -4  DISP        4    1
 -5  D1          1    2    1    0
 -3
 -4  STRESS      6    1
 -1         1 1.00000E+00 2.00000E+00
 -3
"""
    )
    parsed = parse_frd(path)
    assert "frd_disp" not in parsed
    assert parsed["frd_stress"].shape == (1, 2)
