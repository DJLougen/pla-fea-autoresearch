from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PDF = (
    PROJECT_ROOT / "data/public/springer-smlp-2026/paper.pdf"
)
SOURCE_SHA256 = (
    "bc5fda6d4d19c786a62b7e09f14e0bd3ca655bca62d01659db8461a3e165a1e8"
)
DOI_URL = "https://doi.org/10.1007/s00170-026-18180-9"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "data/reported/bambu_pla_interlayer_strength_2026"
)


def source_checksum(path: Path = SOURCE_PDF) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reported_rows() -> List[Dict[str, object]]:
    """Exact values from peer-reviewed Tables 3 and 4."""
    return [
        {
            "printing_strategy": "traditional_multi_head_seam",
            "response": "vertical_interlayer_tensile_strength",
            "mean_MPa": 35.07,
            "standard_deviation_MPa": 2.55,
            "standard_error_MPa": 0.74,
            "table": 3,
        },
        {
            "printing_strategy": "smlp_continuous_layers",
            "response": "vertical_interlayer_tensile_strength",
            "mean_MPa": 47.49,
            "standard_deviation_MPa": 1.16,
            "standard_error_MPa": 0.33,
            "table": 3,
        },
        {
            "printing_strategy": "traditional_multi_head_seam",
            "response": "apparent_tear_strength",
            "mean_MPa": 13.88,
            "standard_deviation_MPa": 2.98,
            "standard_error_MPa": 0.86,
            "table": 4,
        },
        {
            "printing_strategy": "smlp_continuous_layers",
            "response": "apparent_tear_strength",
            "mean_MPa": 14.15,
            "standard_deviation_MPa": 3.17,
            "standard_error_MPa": 0.91,
            "table": 4,
        },
    ]


def build_profiles() -> Dict[str, object]:
    rows = reported_rows()
    profiles: Dict[str, object] = {}
    for strategy in (
        "traditional_multi_head_seam",
        "smlp_continuous_layers",
    ):
        normal = next(
            row for row in rows
            if row["printing_strategy"] == strategy
            and row["response"] == "vertical_interlayer_tensile_strength"
        )
        tear = next(
            row for row in rows
            if row["printing_strategy"] == strategy
            and row["response"] == "apparent_tear_strength"
        )
        profiles[strategy] = {
            "normal_tensile_mean_MPa": normal["mean_MPa"],
            "normal_tensile_standard_deviation_MPa": normal[
                "standard_deviation_MPa"
            ],
            "normal_tensile_lower_1sd_MPa": (
                float(normal["mean_MPa"])
                - float(normal["standard_deviation_MPa"])
            ),
            "apparent_tear_mean_MPa": tear["mean_MPa"],
            "apparent_tear_standard_deviation_MPa": tear[
                "standard_deviation_MPa"
            ],
            "apparent_tear_lower_1sd_MPa": (
                float(tear["mean_MPa"])
                - float(tear["standard_deviation_MPa"])
            ),
        }
    return {
        "material": "Bambu PLA Basic",
        "intended_use": (
            "process-conditioned local-3 opening/apparent-tear initiation "
            "sensitivity for thin printed walls"
        ),
        "profiles": profiles,
        "interaction": (
            "sqrt((max(sigma3,0)/N)^2 + "
            "(sqrt(tau23^2+tau31^2)/T)^2)"
        ),
        "limitations": [
            "tear test is not a pure-mode shear coupon",
            "no traction-separation displacement or fracture energy",
            "no damage evolution, crack growth, or post-peak response",
            "modified dual-extruder printer, not a Bambu printer",
            "0.4 mm single-bead thin wall at 208 C and 0.3 mm layers",
            "strength changes materially with speed and deposition history",
        ],
    }


def write_reference(output_dir: Path = DEFAULT_OUTPUT) -> Dict[str, object]:
    if source_checksum() != SOURCE_SHA256:
        raise RuntimeError("peer-reviewed interlayer source checksum mismatch")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = reported_rows()
    with (output_dir / "table3_table4_strengths.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    profile = build_profiles()
    (output_dir / "interface_initiation_profiles.json").write_text(
        json.dumps(profile, indent=2) + "\n", encoding="utf-8"
    )
    provenance = {
        "title": (
            "Decomposition-free synchronous multi-layer cooperative 3D "
            "printing for enhanced strength and structural integrity"
        ),
        "journal": (
            "The International Journal of Advanced Manufacturing Technology"
        ),
        "year": 2026,
        "volume": 144,
        "pages": "7055-7079",
        "doi": "10.1007/s00170-026-18180-9",
        "url": DOI_URL,
        "local_pdf": str(SOURCE_PDF.relative_to(PROJECT_ROOT)),
        "sha256": SOURCE_SHA256,
        "license": "CC BY 4.0",
        "material": "Bambu PLA Basic",
        "printer": "modified IDEX based on Tronxy Gemini S",
        "fixed_process": {
            "nozzle_temperature_C": 208,
            "bed_temperature_C": 60,
            "layer_height_mm": 0.3,
            "wall_thickness_mm": 0.4,
            "wall_height_mm": 18,
            "travel_speed_mm_s": 60,
        },
        "varied_process": {
            "print_speed_mm_s": "5 to 60 in 5 mm/s increments",
            "layer_length_mm": [50, 100, 150],
            "replicates_per_speed_length": 2,
        },
        "reported_statistics": (
            "Tables 3 and 4 give mean, standard deviation, and standard error "
            "across the comparative strategy series"
        ),
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    report = """# Bambu PLA Basic interface-strength reference

## Usable evidence

Peer-reviewed Tables 3 and 4 provide process-conditioned strength summaries.
The traditional decomposition-seam strategy reports 35.07 +/- 2.55 MPa
vertical interlayer tension and 13.88 +/- 2.98 MPa apparent tear strength.
Continuous SMLP reports 47.49 +/- 1.16 MPa and 14.15 +/- 3.17 MPa,
respectively.

These values support an interface-initiation sensitivity with measured
dispersion. They do not support a cohesive damage law: the paper does not
report a traction-separation curve, critical separation, or fracture energy.
The tear experiment is retained as an apparent interface tear proxy, not
relabeled as a pure shear measurement.

## Transfer boundary

The specimens are 0.4 mm single-bead thin walls printed at 208 C with 0.3 mm
layers on a modified dual-extruder machine. Strength varies with print speed,
layer length, deposition timing, and seam strategy. The profiles are therefore
process bookends for screening, not universal constants for every Bambu PLA
Basic print.
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    return profile


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the peer-reviewed Bambu interlayer strength reference"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    profile = write_reference(args.output)
    print(
        f"wrote {len(profile['profiles'])} interface profiles: "
        f"{args.output.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
