from pathlib import Path

from src.custom_pattern_print_mesh import write_print_meshes


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLAN = (
    PROJECT_ROOT
    / "results/custom_infill/graded_hourglass/"
    "first_physical_test_plan.csv"
)


def test_exact_simulated_meshes_export_as_watertight_equal_mass_stls(
    tmp_path: Path,
):
    manifest = write_print_meshes(tmp_path, PLAN)
    assert manifest["all_print_mesh_gates_passed"] is True
    assert manifest["mesh_count"] == 2
    assert manifest["watertight_mesh_count"] == 2
    for row in manifest["meshes"]:
        assert row["print_mesh_gate_passed"] is True
        assert row["bounds_gate_passed"] is True
        assert row["nonmanifold_edge_count"] == 0
        assert abs(row["mass_relative_error"]) <= 0.001
        assert Path(row["printable_stl"]).exists()


def test_print_mesh_export_is_checksum_deterministic(tmp_path: Path):
    first = write_print_meshes(tmp_path, PLAN)
    first_hashes = {
        row["pattern"]: row["printable_stl_sha256"]
        for row in first["meshes"]
    }
    second = write_print_meshes(tmp_path, PLAN)
    second_hashes = {
        row["pattern"]: row["printable_stl_sha256"]
        for row in second["meshes"]
    }
    assert first_hashes == second_hashes
