from src.weird_infill_toolpath_validation import (
    evaluate_toolpath_gate,
    select_representative_objects,
)


def test_representative_object_selection_requires_three_replicates() -> None:
    parsed = {
        f"{pattern}.stl_{replicate}": []
        for pattern in (
            "triangular",
            "kite_08_08_08_08",
            "kite_08_08_12_08",
        )
        for replicate in range(1, 4)
    }
    selected = select_representative_objects(parsed)
    assert selected["triangular"] == "triangular.stl_1"
    assert selected["kite_08_08_08_08"] == "kite_08_08_08_08.stl_1"


def test_toolpath_gate_requires_convergence_and_both_advantages() -> None:
    rows = []
    values = {
        "triangular": (100.0, 1000.0),
        "kite_08_08_08_08": (130.0, 1150.0),
        "kite_08_08_12_08": (120.0, 1100.0),
    }
    for pattern, (stiffness, frequency) in values.items():
        for resolution, scale in ((60, 1.02), (80, 1.0)):
            rows.extend(
                [
                    {
                        "pattern": pattern,
                        "resolution_xy": resolution,
                        "analysis": "static_bending",
                        "status": "ok",
                        "bending_stiffness_N_per_mm": stiffness * scale,
                        "out_of_plane_frequency_Hz": None,
                    },
                    {
                        "pattern": pattern,
                        "resolution_xy": resolution,
                        "analysis": "storage_modal",
                        "status": "ok",
                        "bending_stiffness_N_per_mm": None,
                        "out_of_plane_frequency_Hz": frequency * scale,
                    },
                ]
            )
    gate = evaluate_toolpath_gate(rows)
    assert gate["toolpath_candidate_gate_passed"] is True
    assert gate["fine_ratios"]["kite_08_08_08_08"][
        "bending_to_triangular"
    ] == 1.3
