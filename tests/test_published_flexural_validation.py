from pathlib import Path

import pytest

from src.calculix import CalculixRunner, ORIENTATIONS, find_ccx
from src.materials import PRIORS, material_card
from src.published_flexural_validation import (
    DEPTH_MM,
    SPAN_MM,
    TEMPLATE,
    WIDTH_MM,
    build_structured_flexural_template,
)


def test_published_flexural_template_has_matched_geometry():
    text = Path(TEMPLATE).read_text()
    assert "100.,12.7,3.2" in text
    assert "*NSET,NSET=SUPPORTS\n5,6,21,22" in text
    assert "*NSET,NSET=LOAD\n15,16" in text


def test_structured_flexural_mesh_retains_supports_and_depth_refinement():
    text = build_structured_flexural_template(4, 4)
    assert "*ELEMENT,TYPE=C3D8I,ELSET=COUPON" in text
    assert ",18,0,0" in text
    assert ",50,0,3.2" in text
    assert ",82,12.7,0" in text
    element_lines = [
        line
        for line in text.splitlines()
        if line and line[0].isdigit() and line.count(",") == 8
    ]
    assert len(element_lines) == 96


@pytest.mark.integration
def test_real_calculix_published_flexural_deck_returns_positive_modulus(
    tmp_path,
):
    executable = find_ccx()
    if executable is None:
        pytest.skip("CalculiX ccx is not installed")
    runner = CalculixRunner(
        tmp_path / "runs", str(executable), timeout_seconds=30
    )
    result = runner.run(
        TEMPLATE,
        {
            "ORIENTATION": ORIENTATIONS["1"],
            "MATERIAL_CARD": material_card(PRIORS),
            "MAX_DISPLACEMENT": -0.25,
        },
        run_id="published-flexure",
    )
    assert result.ok, result.error
    displacement = abs(float(result.curve["displacement"][-1]))
    force = abs(float(result.curve["reaction_force"][-1]))
    apparent_modulus = (
        force
        / displacement
        * SPAN_MM**3
        / (4.0 * WIDTH_MM * DEPTH_MM**3)
    )
    assert apparent_modulus > 1000.0
