from src.bambu_dma_reference import (
    DEFAULT_SOURCE,
    digitize_220c_dma,
    digitize_dma_matrix,
    parse_table2,
    verify_sources,
)


def test_dma_primary_sources_are_checksum_pinned():
    checksums = verify_sources(DEFAULT_SOURCE)
    assert set(checksums) == {
        "article.xml",
        "figure-7-dma.png",
        "figure-9-dma-crossplot.png",
    }


def test_exact_dsc_table2_is_parsed():
    rows = parse_table2(DEFAULT_SOURCE / "article.xml")
    assert len(rows) == 4
    row_220 = next(
        row for row in rows if row["nozzle_temperature_C"] == 220
    )
    assert row_220["DSC_Tg_C"] == 55.30
    assert row_220["degree_crystallinity_percent"] == 13.32


def test_dma_figures_independently_crosscheck_220c_points():
    rows = digitize_220c_dma(
        DEFAULT_SOURCE / "figure-7-dma.png",
        DEFAULT_SOURCE / "figure-9-dma-crossplot.png",
    )
    assert len(rows) == 3
    assert all(row["passes_independent_crossplot"] for row in rows)
    assert max(
        row["storage_crossplot_absolute_difference_GPa"]
        for row in rows
    ) < 0.004
    assert [row["storage_modulus_GPa"] for row in rows] == sorted(
        [row["storage_modulus_GPa"] for row in rows], reverse=True
    )


def test_dma_matrix_preserves_source_crossplot_disagreements():
    rows = digitize_dma_matrix(
        DEFAULT_SOURCE / "figure-7-dma.png",
        DEFAULT_SOURCE / "figure-9-dma-crossplot.png",
    )
    assert len(rows) == 12
    assert {
        row["nozzle_temperature_C"] for row in rows
    } == {210, 220, 230, 240}
    failed = [
        row for row in rows if not row["passes_independent_crossplot"]
    ]
    assert failed
    assert all(
        row["raster_orientation_deg"] == 90 for row in failed
    )
    assert all(
        row["passes_independent_crossplot"]
        for row in rows
        if row["nozzle_temperature_C"] in (220, 240)
    )
