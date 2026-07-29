import os
from pathlib import Path

import numpy as np
import pytest

from src.calculix import CalculixRunner, ORIENTATIONS, find_ccx, render_template
from src.materials import PRIORS, material_card, off_axis_youngs_modulus
from src.parsing import parse_dat


def test_template_rendering_and_orientation():
    template = Path("meshes/tensile_x.inp.template").read_text()
    rendered = render_template(
        template,
        {
            "ORIENTATION": "1.,0.,0.,0.,1.,0.",
            "MATERIAL_CARD": material_card(PRIORS),
            "MAX_DISPLACEMENT": 0.5,
        },
    )
    assert "{{" not in rendered
    assert "*ORIENTATION, NAME=MAT_ORIENTATION" in rendered
    assert "*ELASTIC, TYPE=ENGINEERING CONSTANTS" in rendered


def test_template_rendering_rejects_missing_values():
    with pytest.raises(ValueError, match="missing template value"):
        render_template("value={{A}}", {})


def test_compact_curve_parser(tmp_path):
    dat = tmp_path / "model.dat"
    dat.write_text("CURVE 0 0 0 0 0\nCURVE 1 .5 100 2.5 .01\n")
    curve = parse_dat(dat)
    assert curve["reaction_force"][-1] == pytest.approx(100.0)
    assert curve["strain"][-1] == pytest.approx(0.01)


def test_failed_run_parser(tmp_path):
    dat = tmp_path / "model.dat"
    dat.write_text("no convergence in increment\n")
    with pytest.raises(ValueError, match="non-convergence"):
        parse_dat(dat)


def test_calculix_style_history_parser_includes_fields(tmp_path):
    dat = tmp_path / "model.dat"
    dat.write_text(
        """
 displacements (vx,vy,vz) for set LOAD and time  1.0000000E+00
 2  5.0000000E-01 0.0 0.0
 forces (fx,fy,fz) for set FIXED and time  1.0000000E+00
 1 -1.0000000E+02 0.0 0.0
 stresses (elem, integ.pnt.,sxx,syy,szz,sxy,sxz,syz) for set COUPON and time  1.0000000E+00
 1 1 2.5 0.0 0.0 0.0 0.0 0.0
 strains (elem, integ.pnt.,exx,eyy,ezz,exy,exz,eyz) for set COUPON and time  1.0000000E+00
 1 1 0.01 0.0 0.0 0.0 0.0 0.0
"""
    )
    curve = parse_dat(dat)
    assert curve["reaction_force"][0] == pytest.approx(100.0)
    assert curve["stress"][0] == pytest.approx(2.5)
    assert curve["strain"][0] == pytest.approx(0.01)


def _make_executable(path: Path, body: str) -> Path:
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(path.stat().st_mode | 0o111)
    return path


def test_solver_timeout_is_structured_and_failed_run_is_preserved(tmp_path):
    executable = _make_executable(tmp_path / "slow-ccx", "sleep 2\n")
    template = tmp_path / "model.inp.template"
    template.write_text("*HEADING\n")
    runner = CalculixRunner(tmp_path / "runs", str(executable), timeout_seconds=0.05)
    result = runner.run(template, {}, run_id="timeout-case")
    assert not result.ok
    assert result.status == "timeout"
    assert result.run_dir.exists()
    assert (result.run_dir / "model.inp").exists()


def test_solver_failure_is_structured(tmp_path):
    executable = _make_executable(
        tmp_path / "bad-ccx", "echo 'job failed' >&2\nexit 2\n"
    )
    template = tmp_path / "model.inp.template"
    template.write_text("*HEADING\n")
    # macOS can take more than two seconds to dispatch a newly created
    # executable while the full CalculiX integration suite is under load.
    runner = CalculixRunner(tmp_path / "runs", str(executable), timeout_seconds=5)
    result = runner.run(template, {}, run_id="failure-case")
    assert not result.ok
    assert result.status == "solver_failed"
    assert result.return_code == 2


def test_recoverable_nonlinear_iteration_is_not_solver_failure(tmp_path):
    executable = _make_executable(
        tmp_path / "recovering-ccx",
        (
            "echo 'no convergence'\n"
            "echo 'Job finished'\n"
            "cat > model.dat <<'EOF'\n"
            "CURVE 1 .5 100 2.5 .01\n"
            "EOF\n"
        ),
    )
    template = tmp_path / "model.inp.template"
    template.write_text("*HEADING\n")
    runner = CalculixRunner(tmp_path / "runs", str(executable), timeout_seconds=5)
    result = runner.run(template, {}, run_id="recovered-case")
    assert result.ok
    assert result.curve["reaction_force"][-1] == pytest.approx(100.0)


def test_terminal_nonconvergence_is_solver_failure(tmp_path):
    executable = _make_executable(
        tmp_path / "nonconverging-ccx", "echo 'no convergence'\n"
    )
    template = tmp_path / "model.inp.template"
    template.write_text("*HEADING\n")
    runner = CalculixRunner(tmp_path / "runs", str(executable), timeout_seconds=5)
    result = runner.run(template, {}, run_id="nonconverged-case")
    assert not result.ok
    assert result.status == "solver_failed"


@pytest.mark.integration
def test_real_calculix_is_detectable_when_installed():
    executable = find_ccx()
    if executable is None:
        pytest.skip("CalculiX ccx is not installed")
    assert executable.is_file()


@pytest.mark.integration
def test_real_calculix_solves_orthotropic_coupon(tmp_path):
    executable = find_ccx()
    if executable is None:
        pytest.skip("CalculiX ccx is not installed")
    runner = CalculixRunner(tmp_path / "runs", str(executable), timeout_seconds=30)
    result = runner.run(
        Path("meshes/tensile_x.inp.template"),
        {
            "ORIENTATION": ORIENTATIONS["1"],
            "MATERIAL_CARD": material_card(PRIORS),
            "MAX_DISPLACEMENT": 0.5,
        },
        run_id="real-coupon",
    )
    assert result.ok, result.error
    assert result.curve["reaction_force"][-1] == pytest.approx(1032.0, rel=1e-5)
    assert result.curve["stress"][-1] == pytest.approx(25.8, rel=1e-5)
    strain = result.curve["frd_tostrain"]
    assert np.mean(strain[:, 1]) / np.mean(strain[:, 0]) == pytest.approx(
        -PRIORS["nu12"], rel=1e-5
    )


@pytest.mark.integration
def test_real_calculix_solves_off_axis_and_heldout_geometries(tmp_path):
    executable = find_ccx()
    if executable is None:
        pytest.skip("CalculiX ccx is not installed")
    runner = CalculixRunner(
        tmp_path / "runs", str(executable), timeout_seconds=30
    )
    off_axis = runner.run(
        Path("meshes/tensile_x.inp.template"),
        {
            "ORIENTATION": ORIENTATIONS["12_45"],
            "MATERIAL_CARD": material_card(PRIORS),
            "MAX_DISPLACEMENT": 0.5,
        },
        run_id="off-axis-12",
    )
    assert off_axis.ok, off_axis.error
    expected_force = off_axis_youngs_modulus(PRIORS, "12") * 0.01 * 40.0
    assert off_axis.curve["reaction_force"][-1] == pytest.approx(
        expected_force, rel=1e-5
    )
    heldout = runner.run(
        Path("meshes/heldout_component.inp.template"),
        {
            "ORIENTATION": ORIENTATIONS["mixed"],
            "MATERIAL_CARD": material_card(PRIORS),
            "MAX_DISPLACEMENT": 0.5,
        },
        run_id="heldout-l",
    )
    assert heldout.ok, heldout.error
    assert abs(heldout.curve["reaction_force"][-1]) > 0.0


@pytest.mark.integration
def test_real_calculix_outputs_lateral_strain_for_y_and_z_coupons(
    tmp_path,
):
    executable = find_ccx()
    if executable is None:
        pytest.skip("CalculiX ccx is not installed")
    runner = CalculixRunner(
        tmp_path / "runs", str(executable), timeout_seconds=30
    )
    expected = {
        "2": PRIORS["nu12"] * PRIORS["E2"] / PRIORS["E1"],
        "3": PRIORS["nu13"] * PRIORS["E3"] / PRIORS["E1"],
    }
    for direction, poisson in expected.items():
        result = runner.run(
            Path(f"meshes/tensile_{'y' if direction == '2' else 'z'}.inp.template"),
            {
                "ORIENTATION": ORIENTATIONS[direction],
                "MATERIAL_CARD": material_card(PRIORS),
                "MAX_DISPLACEMENT": 0.5,
            },
            run_id=f"lateral-{direction}",
        )
        assert result.ok, result.error
        strain = result.curve["frd_tostrain"]
        measured = -float(
            np.mean(strain[:, 1]) / np.mean(strain[:, 0])
        )
        assert measured == pytest.approx(poisson, rel=1e-5)
