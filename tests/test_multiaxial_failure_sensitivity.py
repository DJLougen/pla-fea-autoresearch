import json
from pathlib import Path

from src.multiaxial_failure_sensitivity import (
    _physical_priorities,
    audit_summaries,
)


def test_audit_marks_buckling_not_applicable(tmp_path: Path):
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            [
                {
                    "pattern": "idea",
                    "load_case": "buckling",
                    "status": "ok",
                    "run_directory": str(tmp_path / "run"),
                }
            ]
        ),
        encoding="utf-8",
    )
    properties = tmp_path / "properties.json"
    properties.write_text(
        json.dumps(
            {
                "reported_strength_reference": {
                    "overall_tensile_UTS_MPa": 35.0,
                    "overall_tensile_UTS_standard_deviation_MPa": 4.0,
                    "z_upright_tensile_UTS_MPa": 31.0,
                    "z_upright_tensile_UTS_standard_deviation_MPa": 3.0,
                }
            }
        ),
        encoding="utf-8",
    )
    rows = audit_summaries([("test", summary)], properties)
    assert rows[0]["screen_status"] == "not_applicable_eigenvalue_buckling"


def test_physical_priority_reduces_first_displacement():
    rows = [
        {
            "study": "test",
            "pattern": "idea",
            "load_case": "shear",
            "screen_status": "ok",
            "p99_von_mises_shear_bookend_utilization": 1.2,
            "maximum_von_mises_shear_bookend_utilization": 1.5,
            "maximum_equal_tensile_shear_bookend_utilization": 0.9,
            "applied_displacement_mm": 0.4,
            "governing_component": "|tau12|",
        }
    ]
    priorities = _physical_priorities(rows)
    assert priorities[0]["linear_screen_scale"] == 0.5
    assert priorities[0]["recommended_first_displacement_mm"] == 0.2
    assert priorities[0]["replicates"] == 3
