from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean, median
from typing import Dict, List, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    PROJECT_ROOT / "data/public/bambu-pla-cryogenic-jmmp-2025"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "data/reported/bambu_pla_cryogenic_preconditioning_2025"
)
PAPER_FILENAME = "paper.pdf"
PAPER_SHA256 = (
    "556b9382b385d5e1e5b7a4a768d1e3c246f50e5e6f9544b1cc5268c85dd74e2e"
)
ARTICLE_DOI = "10.3390/jmmp9110365"
ARTICLE_URL = "https://doi.org/10.3390/jmmp9110365"
PDF_URL = (
    "https://mdpi-res.com/d_attachment/jmmp/jmmp-09-00365/"
    "article_deploy/jmmp-09-00365-v2.pdf"
)
ANGLES_DEG = (0, 30, 60, 90)
INFILL_PERCENT = (20, 40, 60, 80)
CONDITIONING_AS_PRINTED = "as_printed"
CONDITIONING_CRYOGENIC = (
    "minus80C_24h_then_room_temperature_24h"
)


# Values are the printed numeric labels in Figures 6-15. Each row is a
# printing angle (0/30/60/90 degrees) and each column is an infill level
# (20/40/60/80 percent). The article provides means and plotted error bars,
# but does not define the error-bar statistic or publish its numeric values.
TABLES: Mapping[str, Mapping[str, Mapping[str, object]]] = {
    CONDITIONING_AS_PRINTED: {
        "yield_strength": {
            "test_mode": "tensile",
            "figure": 6,
            "page": 7,
            "unit": "MPa",
            "values": (
                (26.04, 27.87, 30.34, 33.74),
                (32.72, 34.17, 37.03, 38.88),
                (34.41, 36.19, 37.42, 40.15),
                (25.32, 27.25, 31.26, 35.09),
            ),
        },
        "tensile_strength": {
            "test_mode": "tensile",
            "figure": 7,
            "page": 8,
            "unit": "MPa",
            "values": (
                (19.58, 27.87, 30.44, 33.78),
                (32.72, 34.17, 37.03, 38.88),
                (34.17, 36.19, 37.43, 39.27),
                (25.32, 27.95, 31.26, 35.09),
            ),
        },
        "tensile_elastic_modulus": {
            "test_mode": "tensile",
            "figure": 8,
            "page": 9,
            "unit": "MPa",
            "values": (
                (998.97, 1069.32, 1164.25, 1294.63),
                (1255.49, 1311.15, 1420.74, 1491.78),
                (1320.29, 1388.73, 1435.88, 1540.45),
                (971.54, 1045.50, 1199.35, 1346.40),
            ),
        },
        "flexural_modulus": {
            "test_mode": "flexural",
            "figure": 12,
            "page": 13,
            "unit": "MPa",
            "values": (
                (2739.24, 2374.22, 2521.01, 2659.13),
                (2527.93, 2627.15, 2602.79, 2605.23),
                (2505.61, 2618.12, 2621.96, 2622.94),
                (2698.43, 2562.03, 2589.35, 2698.38),
            ),
        },
        "flexural_strength": {
            "test_mode": "flexural",
            "figure": 13,
            "page": 14,
            "unit": "MPa",
            "values": (
                (50.01, 49.86, 50.99, 54.34),
                (53.93, 54.78, 55.97, 55.84),
                (54.66, 56.54, 56.34, 57.33),
                (58.81, 58.41, 58.81, 58.89),
            ),
        },
    },
    CONDITIONING_CRYOGENIC: {
        "yield_strength": {
            "test_mode": "tensile",
            "figure": 9,
            "page": 10,
            "unit": "MPa",
            "values": (
                (22.60, 25.39, 27.97, 31.98),
                (30.72, 31.24, 32.57, 35.09),
                (36.04, 33.77, 33.77, 31.36),
                (23.10, 25.54, 28.35, 31.91),
            ),
        },
        "tensile_strength": {
            "test_mode": "tensile",
            "figure": 10,
            "page": 10,
            "unit": "MPa",
            "values": (
                (17.10, 25.46, 27.97, 31.98),
                (30.72, 31.24, 32.57, 35.09),
                (31.36, 31.87, 33.77, 36.04),
                (22.90, 25.37, 28.62, 32.13),
            ),
        },
        "tensile_elastic_modulus": {
            "test_mode": "tensile",
            "figure": 11,
            "page": 11,
            "unit": "MPa",
            "values": (
                (792.24, 887.84, 1204.54, 1322.45),
                (973.09, 1037.01, 1058.84, 1125.01),
                (1102.03, 964.51, 932.62, 1004.79),
                (870.11, 957.68, 1082.83, 1165.51),
            ),
        },
        "flexural_modulus": {
            "test_mode": "flexural",
            "figure": 14,
            "page": 15,
            "unit": "MPa",
            "values": (
                (2276.05, 2307.49, 2402.80, 2580.12),
                (2448.17, 2348.30, 2431.32, 2482.56),
                (2428.07, 2447.93, 2388.08, 2438.32),
                (2577.00, 2474.26, 2472.83, 2519.63),
            ),
        },
        "flexural_strength": {
            "test_mode": "flexural",
            "figure": 15,
            "page": 16,
            "unit": "MPa",
            "values": (
                (41.11, 43.22, 46.21, 47.97),
                (47.19, 49.33, 49.66, 50.01),
                (45.45, 47.22, 46.71, 47.62),
                (51.16, 51.16, 51.16, 51.39),
            ),
        },
    },
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def verify_source(source_dir: Path = DEFAULT_SOURCE) -> Dict[str, object]:
    path = Path(source_dir) / PAPER_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"missing published source file: {path}")
    observed = _sha256(path)
    if observed != PAPER_SHA256:
        raise ValueError(
            f"{PAPER_FILENAME} checksum changed: expected {PAPER_SHA256}, "
            f"got {observed}"
        )
    return {
        "filename": PAPER_FILENAME,
        "sha256": observed,
        "pages": 22,
        "doi": ARTICLE_DOI,
        "article_url": ARTICLE_URL,
        "pdf_url": PDF_URL,
    }


def figure_mean_rows() -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for conditioning, metrics in TABLES.items():
        for metric, definition in metrics.items():
            values = definition["values"]
            for angle_index, angle in enumerate(ANGLES_DEG):
                for infill_index, infill in enumerate(INFILL_PERCENT):
                    rows.append(
                        {
                            "conditioning": conditioning,
                            "test_mode": definition["test_mode"],
                            "metric": metric,
                            "figure": definition["figure"],
                            "source_page": definition["page"],
                            "raster_angle_deg": angle,
                            "infill_percent": infill,
                            "mean_value": values[angle_index][infill_index],
                            "unit": definition["unit"],
                            "specimens_per_configuration": 5,
                            "evidence_class": "figure_label_mean",
                            "numeric_uncertainty_available": False,
                        }
                    )
    if len(rows) != 160:
        raise AssertionError(f"expected 160 labeled means, got {len(rows)}")
    return rows


def conditioning_effect_rows(
    rows: Sequence[Mapping[str, object]] | None = None,
) -> List[Dict[str, object]]:
    source_rows = list(rows or figure_mean_rows())
    lookup = {
        (
            str(row["conditioning"]),
            str(row["test_mode"]),
            str(row["metric"]),
            int(row["raster_angle_deg"]),
            int(row["infill_percent"]),
        ): float(row["mean_value"])
        for row in source_rows
    }
    paired: List[Dict[str, object]] = []
    for test_mode, metric in (
        ("tensile", "yield_strength"),
        ("tensile", "tensile_strength"),
        ("tensile", "tensile_elastic_modulus"),
        ("flexural", "flexural_modulus"),
        ("flexural", "flexural_strength"),
    ):
        for angle in ANGLES_DEG:
            for infill in INFILL_PERCENT:
                key = (test_mode, metric, angle, infill)
                as_printed = lookup[(CONDITIONING_AS_PRINTED, *key)]
                cryogenic = lookup[(CONDITIONING_CRYOGENIC, *key)]
                paired.append(
                    {
                        "test_mode": test_mode,
                        "metric": metric,
                        "raster_angle_deg": angle,
                        "infill_percent": infill,
                        "as_printed_mean": as_printed,
                        "cryogenic_recovered_mean": cryogenic,
                        "unit": "MPa",
                        "delta": cryogenic - as_printed,
                        "ratio": cryogenic / as_printed,
                        "percent_change": 100.0
                        * (cryogenic / as_printed - 1.0),
                        "paired_summary_only": True,
                        "replicates_each_condition": 5,
                    }
                )
    if len(paired) != 80:
        raise AssertionError(f"expected 80 paired effects, got {len(paired)}")
    return paired


def metric_effect_summary(
    effects: Sequence[Mapping[str, object]] | None = None,
) -> List[Dict[str, object]]:
    effect_rows = list(effects or conditioning_effect_rows())
    summaries: List[Dict[str, object]] = []
    metrics = (
        "yield_strength",
        "tensile_strength",
        "tensile_elastic_modulus",
        "flexural_modulus",
        "flexural_strength",
    )
    for metric in metrics:
        selected = [
            row for row in effect_rows if row["metric"] == metric
        ]
        ratios = [float(row["ratio"]) for row in selected]
        deltas = [float(row["percent_change"]) for row in selected]
        summaries.append(
            {
                "metric": metric,
                "configuration_count": len(selected),
                "ratio_min": min(ratios),
                "ratio_median": median(ratios),
                "ratio_mean": mean(ratios),
                "ratio_max": max(ratios),
                "percent_change_min": min(deltas),
                "percent_change_median": median(deltas),
                "percent_change_max": max(deltas),
                "configurations_decreased": sum(
                    float(row["delta"]) < 0.0 for row in selected
                ),
                "configurations_unchanged": sum(
                    float(row["delta"]) == 0.0 for row in selected
                ),
                "configurations_increased": sum(
                    float(row["delta"]) > 0.0 for row in selected
                ),
            }
        )
    return summaries


def lookup_effect(
    metric: str, raster_angle_deg: int, infill_percent: int
) -> Dict[str, object]:
    matches = [
        row
        for row in conditioning_effect_rows()
        if row["metric"] == metric
        and int(row["raster_angle_deg"]) == int(raster_angle_deg)
        and int(row["infill_percent"]) == int(infill_percent)
    ]
    if len(matches) != 1:
        raise ValueError(
            "lookup requires one published metric/angle/infill combination"
        )
    return matches[0]


def write_reference(
    source_dir: Path = DEFAULT_SOURCE,
    output_dir: Path = DEFAULT_OUTPUT,
) -> Dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source = verify_source(source_dir)
    means = figure_mean_rows()
    effects = conditioning_effect_rows(means)
    summaries = metric_effect_summary(effects)
    _write_csv(output_dir / "summary_means.csv", means)
    _write_csv(output_dir / "conditioning_effects.csv", effects)
    _write_csv(output_dir / "metric_effect_summary.csv", summaries)

    summary_lookup = {row["metric"]: row for row in summaries}
    quality_gate = (
        len(means) == 160
        and len(effects) == 80
        and len(summaries) == 5
        and {
            (
                row["conditioning"],
                row["metric"],
                int(row["raster_angle_deg"]),
                int(row["infill_percent"]),
            )
            for row in means
        }
        == {
            (conditioning, metric, angle, infill)
            for conditioning, metrics in TABLES.items()
            for metric in metrics
            for angle in ANGLES_DEG
            for infill in INFILL_PERCENT
        }
        and lookup_effect("tensile_strength", 60, 80)[
            "as_printed_mean"
        ]
        == 39.27
        and lookup_effect("flexural_strength", 90, 80)[
            "cryogenic_recovered_mean"
        ]
        == 51.39
    )
    payload: Dict[str, object] = {
        "reference_id": "bambu_pla_basic_cryo_recovery_summary_v1",
        "material": "Bambu Lab PLA Basic",
        "source": source,
        "publication": {
            "title": (
                "Effects of Printing Angle, Infill Density and Cryogenic "
                "Pre-Treatment on the Tensile and Flexural Properties of "
                "FFF-Printed PLA"
            ),
            "journal": "Journal of Manufacturing and Materials Processing",
            "year": 2025,
            "volume": 9,
            "issue": 11,
            "article": 365,
        },
        "printing": {
            "printer": "Bambu Lab P1S",
            "slicer": "Bambu Studio 2.2.2.56",
            "nozzle_temperature_C": 240,
            "bed_temperature_C": 55,
            "nozzle_diameter_mm": 0.4,
            "layer_height_mm": 0.2,
            "infill_pattern": "cubic",
            "infill_percent": list(INFILL_PERCENT),
            "raster_angle_deg": list(ANGLES_DEG),
            "perimeter_contours": 2,
            "filament_drying": "20 C for 24 h before printing",
        },
        "conditioning": {
            "as_printed": "tested without additional treatment",
            "cryogenic_pre_treatment": (
                "sealed after printing; -80 C for 24 h; warmed at laboratory "
                "temperature for 24 h before mechanical testing"
            ),
            "mechanical_test_environment": "23 +/- 2 C, 50 +/- 5% RH",
            "critical_interpretation": (
                "room-temperature response after freeze/recovery; not "
                "mechanical response while at -80 C"
            ),
        },
        "testing": {
            "tension": "ASTM D638 Type I, 5 mm/min",
            "flexure": "ASTM D790, 64 mm span, 2 mm/min",
            "specimens_per_configuration": 5,
            "total_specimens": 320,
            "as_printed_specimens": 160,
            "cryogenic_pre_treated_specimens": 160,
        },
        "extraction": {
            "figure_labeled_means": len(means),
            "paired_conditioning_effects": len(effects),
            "raw_specimen_curves_available": False,
            "numeric_dispersion_available": False,
            "error_bar_statistic_defined": False,
            "data_availability_statement": (
                "original contributions are included in the article; "
                "further inquiries may be directed to the corresponding author"
            ),
        },
        "known_internal_inconsistencies": [
            {
                "location": "Section 3.1.3 versus Figure 8",
                "issue": (
                    "prose reports the as-printed tensile-modulus minimum as "
                    "998.97 MPa, while Figure 8 labels 971.54 MPa at 90/20"
                ),
                "resolution": "retain the printed figure label and flag it",
            },
            {
                "location": "Section 3.6.2 versus Figures 13 and 15",
                "issue": (
                    "the sentence assigns 58.89 MPa to the cryogenic series "
                    "and 51.39 MPa to an unspecified/as-printed series, "
                    "opposite the charts and the stated decrease"
                ),
                "resolution": (
                    "retain Figure 13 as-printed 58.89 MPa and Figure 15 "
                    "cryogenic 51.39 MPa"
                ),
            },
            {
                "location": "Section 3.3.3 blanket comparison",
                "issue": (
                    "prose says all monitored cryogenic values are lower, "
                    "but two tensile-modulus cells and one yield cell increase"
                ),
                "resolution": (
                    "retain every chart label and report increased/decreased "
                    "cell counts instead of the blanket statement"
                ),
            },
        ],
        "effect_summary": summaries,
        "quality_gate_passed": quality_gate,
        "allowed_use": [
            "exact published cell lookup for cubic-infill summary means",
            (
                "bounded freeze/recovery sensitivity over the published "
                "20-80% infill and 0-90 degree matrix"
            ),
            (
                "adversarial ranking screen using the observed correction "
                "factor range, with explicit topology-transfer caveat"
            ),
        ],
        "prohibited_use": [
            "raw-curve or specimen-variance calibration",
            "service behavior while the part is at -80 C",
            "general temperature-dependent constitutive interpolation",
            "creep, fatigue, moisture, aging, fracture, or damage calibration",
            "replacement of missing shear moduli or Poisson ratios",
            "unqualified transfer from cubic infill to a novel topology",
        ],
        "flexural_modulus_ratio_range": {
            key: summary_lookup["flexural_modulus"][key]
            for key in (
                "ratio_min",
                "ratio_median",
                "ratio_mean",
                "ratio_max",
            )
        },
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    report = f"""# Bambu PLA Basic freeze/recovery reference

This checksum-pinned reference captures **160 printed chart means** from
**320 Bambu PLA Basic specimens**: four raster angles, four cubic-infill
levels, tensile and flexural tests, and both as-printed and freeze/recovered
conditions.

The treated specimens were held at **-80 C for 24 h**, then returned to
laboratory temperature for **24 h** before testing at 23 +/- 2 C. This is a
freeze/recovery conditioning study, not an in-service -80 C material law.

Across the 16 published configurations, the recovered/as-printed median ratios
are:

| Response | Median ratio | Range | Cells decreased / increased |
|---|---:|---:|---:|
| Yield strength | {summary_lookup['yield_strength']['ratio_median']:.3f} | {summary_lookup['yield_strength']['ratio_min']:.3f}-{summary_lookup['yield_strength']['ratio_max']:.3f} | {summary_lookup['yield_strength']['configurations_decreased']} / {summary_lookup['yield_strength']['configurations_increased']} |
| Tensile strength | {summary_lookup['tensile_strength']['ratio_median']:.3f} | {summary_lookup['tensile_strength']['ratio_min']:.3f}-{summary_lookup['tensile_strength']['ratio_max']:.3f} | {summary_lookup['tensile_strength']['configurations_decreased']} / {summary_lookup['tensile_strength']['configurations_increased']} |
| Tensile modulus | {summary_lookup['tensile_elastic_modulus']['ratio_median']:.3f} | {summary_lookup['tensile_elastic_modulus']['ratio_min']:.3f}-{summary_lookup['tensile_elastic_modulus']['ratio_max']:.3f} | {summary_lookup['tensile_elastic_modulus']['configurations_decreased']} / {summary_lookup['tensile_elastic_modulus']['configurations_increased']} |
| Flexural modulus | {summary_lookup['flexural_modulus']['ratio_median']:.3f} | {summary_lookup['flexural_modulus']['ratio_min']:.3f}-{summary_lookup['flexural_modulus']['ratio_max']:.3f} | {summary_lookup['flexural_modulus']['configurations_decreased']} / {summary_lookup['flexural_modulus']['configurations_increased']} |
| Flexural strength | {summary_lookup['flexural_strength']['ratio_median']:.3f} | {summary_lookup['flexural_strength']['ratio_min']:.3f}-{summary_lookup['flexural_strength']['ratio_max']:.3f} | {summary_lookup['flexural_strength']['configurations_decreased']} / {summary_lookup['flexural_strength']['configurations_increased']} |

Quality gate: **{'passed' if quality_gate else 'failed'}**.

The source publishes no raw machine-exported curves, numerical dispersion, or
definition for the plotted error bars. Three internal prose/chart
inconsistencies are retained in `provenance.json`; the transcribed chart labels
are never silently reconciled to the contradictory prose.
"""
    (output_dir / "quality_report.md").write_text(
        report, encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the Bambu PLA Basic freeze/recovery reference"
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = write_reference(args.source, args.output)
    print(
        json.dumps(
            {
                "reference_id": payload["reference_id"],
                "figure_labeled_means": payload["extraction"][
                    "figure_labeled_means"
                ],
                "paired_conditioning_effects": payload["extraction"][
                    "paired_conditioning_effects"
                ],
                "quality_gate_passed": payload["quality_gate_passed"],
                "output": str(args.output.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
