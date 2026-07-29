from src.bambu_nozzle_temperature_profiles import (
    write_nozzle_temperature_profiles,
)


def test_nozzle_temperature_profiles_are_written_with_quality_flags(tmp_path):
    manifest = write_nozzle_temperature_profiles(output_dir=tmp_path)
    assert len(manifest) == 4
    assert all(
        (tmp_path / f"print_{temperature}C.json").exists()
        for temperature in (210, 220, 230, 240)
    )
    quality = {
        row["nozzle_temperature_C"]: row["crossplot_quality"]
        for row in manifest
    }
    assert quality == {
        210: "source_disagreement",
        220: "strict_pass",
        230: "source_disagreement",
        240: "strict_pass",
    }
