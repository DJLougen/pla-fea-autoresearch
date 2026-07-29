from src.bambu_flexural_process_reference import (
    DEFAULT_SOURCE,
    digitize_process_crossplots,
    verify_sources,
)
from src.bambu_flexural_digitization import parse_table3


def test_flexural_process_sources_are_checksum_pinned():
    checksums = verify_sources(DEFAULT_SOURCE)
    assert set(checksums) == {
        "article.xml",
        "figure-5-flexural-process.png",
        "figure-6-flexural-crossplot.png",
    }


def test_all_flexural_process_rows_crosscheck_exact_table():
    table = parse_table3(DEFAULT_SOURCE / "article.xml")
    rows = digitize_process_crossplots(
        DEFAULT_SOURCE / "figure-5-flexural-process.png",
        DEFAULT_SOURCE / "figure-6-flexural-crossplot.png",
        table,
    )
    assert len(rows) == 12
    assert all(row["passes_crossplot_and_table_gate"] for row in rows)
    assert max(
        row["figure5_figure6_modulus_difference_GPa"] for row in rows
    ) < 0.007
    assert max(
        row["figure5_figure6_strength_difference_MPa"] for row in rows
    ) < 0.12
