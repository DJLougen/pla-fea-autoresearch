from pathlib import Path

from src.bambu_cryogenic_reference import (
    CONDITIONING_CRYOGENIC,
    conditioning_effect_rows,
    figure_mean_rows,
    lookup_effect,
    metric_effect_summary,
    verify_source,
    write_reference,
)


def test_source_and_all_figure_labels_are_quality_gated() -> None:
    source = verify_source()
    assert source["pages"] == 22
    rows = figure_mean_rows()
    assert len(rows) == 160
    assert {row["conditioning"] for row in rows} == {
        "as_printed",
        CONDITIONING_CRYOGENIC,
    }
    assert all(int(row["specimens_per_configuration"]) == 5 for row in rows)
    assert all(row["numeric_uncertainty_available"] is False for row in rows)


def test_paired_effects_retain_exact_labels_and_cell_counts() -> None:
    effects = conditioning_effect_rows()
    summaries = {
        row["metric"]: row for row in metric_effect_summary(effects)
    }
    assert len(effects) == 80
    tensile = lookup_effect("tensile_strength", 60, 80)
    flexural = lookup_effect("flexural_strength", 90, 80)
    modulus = lookup_effect("tensile_elastic_modulus", 90, 20)
    assert tensile["as_printed_mean"] == 39.27
    assert tensile["cryogenic_recovered_mean"] == 36.04
    assert flexural["as_printed_mean"] == 58.89
    assert flexural["cryogenic_recovered_mean"] == 51.39
    assert modulus["as_printed_mean"] == 971.54
    assert summaries["tensile_strength"]["configurations_decreased"] == 16
    assert summaries["flexural_modulus"]["configurations_decreased"] == 16
    assert summaries["yield_strength"]["configurations_increased"] == 1
    assert (
        summaries["tensile_elastic_modulus"][
            "configurations_increased"
        ]
        == 2
    )


def test_reference_writer_exposes_limits_and_discrepancies(
    tmp_path: Path,
) -> None:
    payload = write_reference(output_dir=tmp_path)
    assert payload["quality_gate_passed"] is True
    assert payload["testing"]["total_specimens"] == 320
    assert payload["extraction"]["figure_labeled_means"] == 160
    assert payload["extraction"]["raw_specimen_curves_available"] is False
    assert len(payload["known_internal_inconsistencies"]) == 3
    assert (tmp_path / "summary_means.csv").exists()
    assert (tmp_path / "conditioning_effects.csv").exists()
    assert (tmp_path / "metric_effect_summary.csv").exists()
    assert (tmp_path / "quality_report.md").exists()
