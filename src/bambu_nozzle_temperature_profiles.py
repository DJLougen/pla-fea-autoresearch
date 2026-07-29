from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

from .bambu_dma_reference import (
    DEFAULT_OUTPUT as DMA_REFERENCE,
    NOZZLE_TEMPERATURES_C,
)
from .bambu_dynamic_profile import build_dynamic_profile
from .bambu_tds_profile import DEFAULT_TRANSVERSE_PROFILE
from .materials import material_card


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "data/reported/bambu_pla_nozzle_temperature_sensitivity"
)


def write_nozzle_temperature_profiles(
    output_dir: Path = DEFAULT_OUTPUT,
    dma_reference: Path = DMA_REFERENCE,
    transverse_profile: Path = DEFAULT_TRANSVERSE_PROFILE,
) -> List[Dict[str, object]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: List[Dict[str, object]] = []
    for temperature in NOZZLE_TEMPERATURES_C:
        profile = build_dynamic_profile(
            dma_reference=Path(dma_reference),
            transverse_profile=Path(transverse_profile),
            nozzle_temperature_C=temperature,
        )
        profile_path = output_dir / f"print_{temperature}C.json"
        card_path = output_dir / f"print_{temperature}C.inp"
        profile_path.write_text(
            json.dumps(profile, indent=2) + "\n", encoding="utf-8"
        )
        card_path.write_text(
            material_card(
                profile["screening_engineering_constants"],
                name=f"BAMBU_PLA_DMA_PRINT_{temperature}C",
            ),
            encoding="utf-8",
        )
        constants = profile["screening_engineering_constants"]
        reference = profile["dynamic_reference"]
        passed = int(reference["independent_crossplot_rows_passed"])
        total = int(reference["independent_crossplot_rows_total"])
        manifest.append(
            {
                "nozzle_temperature_C": temperature,
                "measurement_temperature_C": reference[
                    "measurement_temperature_C"
                ],
                "frequency_Hz": reference["frequency_Hz"],
                "E1_MPa": constants["E1"],
                "E2_MPa": constants["E2"],
                "E3_MPa": constants["E3"],
                "G12_MPa": constants["G12"],
                "G13_MPa": constants["G13"],
                "G23_MPa": constants["G23"],
                "minimum_DMA_tan_delta_peak_Tg_C": reference[
                    "minimum_DMA_tan_delta_peak_Tg_C"
                ],
                "crossplot_rows_passed": passed,
                "crossplot_rows_total": total,
                "crossplot_quality": (
                    "strict_pass" if passed == total else "source_disagreement"
                ),
                "profile_path": str(profile_path.resolve()),
                "material_card_path": str(card_path.resolve()),
            }
        )
    with (output_dir / "manifest.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# Bambu PLA Basic nozzle-temperature storage profiles",
        "",
        "These profiles describe the as-printed 30 °C / 1 Hz storage response "
        "reported for four nozzle temperatures. Nozzle temperature is a "
        "manufacturing variable—not the service temperature.",
        "",
        "| Nozzle | E1 | E2 | E3 proxy | G12 | Crossplot |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for row in manifest:
        lines.append(
            f"| {row['nozzle_temperature_C']} °C | "
            f"{float(row['E1_MPa']):.1f} | "
            f"{float(row['E2_MPa']):.1f} | "
            f"{float(row['E3_MPa']):.1f} | "
            f"{float(row['G12_MPa']):.1f} | "
            f"{row['crossplot_rows_passed']}/"
            f"{row['crossplot_rows_total']} "
            f"({row['crossplot_quality']}) |"
        )
    lines.extend(
        [
            "",
            "## Use boundary",
            "",
            "- E1/E2/E45 come from plotted summary markers; G12 is inferred "
            "from E45 with the separate exact-brand measured ν12.",
            "- E3 and out-of-plane shear use the existing proportional-scaling "
            "hypothesis because the paper reports in-plane orientations only.",
            "- The 210 °C and 230 °C 90° markers disagree between the two "
            "published cross-plots. Their profiles remain exploratory.",
            "- These cards do not represent print-flow quality, voids, "
            "interlayer defects, service-temperature softening, loss modulus, "
            "damping, strength changes, or fracture.",
            "",
        ]
    )
    (output_dir / "report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Bambu PLA nozzle-temperature storage profiles"
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--dma-reference", default=str(DMA_REFERENCE))
    parser.add_argument(
        "--transverse-profile", default=str(DEFAULT_TRANSVERSE_PROFILE)
    )
    args = parser.parse_args()
    manifest = write_nozzle_temperature_profiles(
        Path(args.output),
        Path(args.dma_reference),
        Path(args.transverse_profile),
    )
    passed = sum(row["crossplot_quality"] == "strict_pass" for row in manifest)
    print(
        f"nozzle-temperature profiles: {len(manifest)} admissible, "
        f"{passed} strict crossplot passes; "
        f"report: {(Path(args.output) / 'report.md').resolve()}"
    )


if __name__ == "__main__":
    main()
