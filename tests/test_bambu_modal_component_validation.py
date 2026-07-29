from pathlib import Path

import pytest

from src.bambu_modal_component_validation import (
    FROZEN_PROFILE,
    MODE_LABELS,
    SOURCE_DECKS,
    classify_common_plate_modes,
    extract_test2_dimensions,
    extract_test2_modal_targets,
    render_calculix_modal_deck,
)


def test_extracts_three_physical_components_and_five_modes_each():
    targets, author = extract_test2_modal_targets()
    assert len(targets) == 15
    assert len(author) == 15
    assert {row["mode"] for row in targets} == set(MODE_LABELS)
    assert {row["replicate"] for row in targets} == {1, 2, 3}
    assert {row["mesh_size_mm"] for row in author} == {0.5, 1.0, 2.0}


def test_extracts_measured_dimensions_and_print_placements():
    rows = extract_test2_dimensions()
    assert len(rows) == 3
    assert {row["source_print_orientation"] for row in rows} == {
        "Vertical Print",
        "Horizontal Print",
    }
    assert all(row["mean_length_mm"] > 140.0 for row in rows)
    assert all(3.8 < row["mean_thickness_mm"] < 4.2 for row in rows)


def test_abaqus_mesh_is_flattened_to_a_calculix_modal_deck():
    profile = {
        "E_MPa": 2107.0,
        "nu": 0.3,
        "density_tonnes_mm3": 1.18e-9,
    }
    deck, metadata = render_calculix_modal_deck(
        SOURCE_DECKS[2.0], profile
    )
    assert "*Part" not in deck
    assert "*Assembly" not in deck
    assert "*ELEMENT,TYPE=C3D8R,ELSET=EALL" in deck
    assert "*FREQUENCY\n10" in deck
    assert "*NSET,NSET=FIXED" in deck
    assert metadata["node_count"] == 4992
    assert metadata["element_count"] == 3150
    assert metadata["fixed_node_count"] == 78


def test_frozen_profile_preexists_component_validation():
    assert Path(FROZEN_PROFILE).exists()


def test_renderer_rejects_too_few_modes():
    with pytest.raises(ValueError, match="at least five"):
        render_calculix_modal_deck(
            SOURCE_DECKS[2.0],
            {
                "E_MPa": 2107.0,
                "nu": 0.3,
                "density_tonnes_mm3": 1.18e-9,
            },
            mode_count=4,
        )


def test_mode_classifier_excludes_chordwise_bending():
    def mode(number, frequency, mx, my, rz):
        return {
            "mode": number,
            "frequency_Hz": frequency,
            "effective_mass_x": mx,
            "effective_mass_y": my,
            "effective_rotational_mass_z": rz,
        }

    modes = [
        mode(1, 55.0, 1e-5, 0.0, 1e-2),
        mode(2, 285.0, 0.0, 0.0, 4e-3),
        mode(3, 344.0, 5e-6, 0.0, 3e-3),
        mode(4, 622.0, 0.0, 2e-5, 1e-4),
        mode(5, 903.0, 0.0, 0.0, 5e-4),
        mode(6, 958.0, 2e-6, 0.0, 1e-3),
    ]
    selected = classify_common_plate_modes(modes)
    assert {label: row["mode"] for label, row in selected.items()} == {
        "1-B": 1,
        "1-T": 2,
        "2-B": 3,
        "2-T": 5,
        "3-B": 6,
    }
