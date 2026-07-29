from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    PROJECT_ROOT / "data/public/bambu-pla-uav-arm-machines-2026"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "data/reported/bambu_pla_uav_arm_2026"
)
ARTICLE_DOI = "10.3390/machines14050474"
ARTICLE_URL = "https://doi.org/10.3390/machines14050474"
EXPECTED_SHA256 = {
    "article.xml": (
        "f3a6b6eb3234c04aff17c3c698cf39ef729ddd3e11a3ef1c436954a9bd1dcdea"
    ),
    "paper.pdf": (
        "34faeb65b101758110edee720d8bb75d2b773a793dfe1894da9689aeb4c1de3b"
    ),
    "figure-002.png": (
        "199c3e61690e95a64eac606fe27d390bf0c0de0f0a2571adf6643e1f87fb0009"
    ),
    "figure-004.png": (
        "7c87fafbfb8c203965e6d2b4ce32318aaef7cb6bfd07f68b968b1fb9b1011778"
    ),
    "figure-007.png": (
        "6bb31cd3a590c82ea75b2ebc4a4e0f33c5ba8e63ae4ff988c517415b200727a9"
    ),
}

PRINT_PROCESS = {
    "printer_reported": "Bambu P2S",
    "material": "Bambu PLA Basic",
    "filament_diameter_mm": 1.75,
    "reported_density_g_cm3": 1.25,
    "nozzle_diameter_mm": 0.4,
    "nozzle_temperature_C": 220,
    "bed_temperature_C": 70,
    "printing_speed_mm_s": 250,
    "layer_height_mm": 0.2,
    "wall_thickness_mm": 4.0,
    "infill_percent": 100,
    "top_bottom_thickness_mm": 0.6,
    "post_print_drying": "8 h at 25 C",
}
TEST_PROCESS = {
    "test": "quasi-static axial loading through two mounting holes",
    "crosshead_speed_mm_min": 1.0,
    "maximum_crosshead_displacement_mm": 10.0,
    "measurement": "machine load plus laser-extensometer displacement",
    "plotted_specimen_traces_per_geometry": 3,
}
AUTHOR_FEA = {
    "solver": "Abaqus/Standard 2022",
    "density_g_cm3": 1.25,
    "elastic_modulus_MPa": 3000.0,
    "poisson_ratio": 0.35,
    "load_N": 500.0,
    "mesh_sizes_mm": [3.0, 2.0, 1.5, 0.75],
    "reported_final_refinement_displacement_change_below_percent": 5.0,
}
TABLE3 = {
    "topology_optimized": {
        "elastic_modulus_mean_MPa": 2234.254,
        "elastic_modulus_sd_MPa": 76.179,
        "tensile_strength_mean_MPa": 27.913,
        "tensile_strength_sd_MPa": 0.387,
        "SEA_mean_J_m3": 742.42,
        "SEA_sd_J_m3": 29.016,
    },
    "honeycomb": {
        "elastic_modulus_mean_MPa": 886.007,
        "elastic_modulus_sd_MPa": 16.639,
        "tensile_strength_mean_MPa": 11.0844,
        "tensile_strength_sd_MPa": 0.02,
        "SEA_mean_J_m3": 431.725,
        "SEA_sd_J_m3": 4.640,
    },
    "central_lightening_hole": {
        "elastic_modulus_mean_MPa": 1726.861,
        "elastic_modulus_sd_MPa": 178.221,
        "tensile_strength_mean_MPa": 19.489,
        "tensile_strength_sd_MPa": 0.19,
        "SEA_mean_J_m3": 620.051,
        "SEA_sd_J_m3": 15.261,
    },
    "solid_baseline": {
        "elastic_modulus_mean_MPa": 1399.922,
        "elastic_modulus_sd_MPa": 47.371,
        "tensile_strength_mean_MPa": 15.777,
        "tensile_strength_sd_MPa": 1.31,
        "SEA_mean_J_m3": 530.178,
        "SEA_sd_J_m3": 89.352,
    },
}
AUTHOR_FEA_PEAK_STRESS_MPA = {
    "topology_optimized": 5.539,
    "honeycomb": 2.611,
    "central_lightening_hole": 1.997,
    "solid_baseline": 1.865,
}
FIGURE7_PANELS = {
    "topology_optimized": {
        "x_left": 117,
        "x_right": 1186,
        "y_top": 24,
        "y_bottom": 886,
    },
    "honeycomb": {
        "x_left": 1684,
        "x_right": 2754,
        "y_top": 24,
        "y_bottom": 886,
    },
    "central_lightening_hole": {
        "x_left": 117,
        "x_right": 1186,
        "y_top": 1290,
        "y_bottom": 2152,
    },
    "solid_baseline": {
        "x_left": 1684,
        "x_right": 2754,
        "y_top": 1290,
        "y_bottom": 2152,
    },
}
SAMPLE_IDS = ("sample_1", "sample_2", "sample_3")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_sources(source_dir: Path = DEFAULT_SOURCE) -> Dict[str, str]:
    observed = {}
    for filename, expected in EXPECTED_SHA256.items():
        path = Path(source_dir) / filename
        if not path.exists():
            raise FileNotFoundError(f"missing published source file: {path}")
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(
                f"{filename} checksum changed: expected {expected}, got {actual}"
            )
        observed[filename] = actual
    return observed


def _color_masks(hsv: np.ndarray) -> Dict[str, np.ndarray]:
    hue = hsv[:, :, 0].astype(float) / 255.0
    saturation = hsv[:, :, 1].astype(float) / 255.0
    value = hsv[:, :, 2].astype(float) / 255.0
    colored = (saturation > 0.20) & (value > 0.20)
    return {
        "sample_1": colored & (hue > 0.035) & (hue < 0.19),
        "sample_2": colored & ((hue > 0.86) | (hue < 0.025)),
        "sample_3": colored & (hue > 0.66) & (hue < 0.86),
    }


def _pixel_to_curve(
    x_pixel: float,
    y_pixel: float,
    definition: Mapping[str, int],
) -> tuple[float, float]:
    strain = (
        (x_pixel - definition["x_left"])
        / (definition["x_right"] - definition["x_left"])
        * 0.10
    )
    stress = (
        (definition["y_bottom"] - y_pixel)
        / (definition["y_bottom"] - definition["y_top"])
        * 30.0
    )
    return float(strain), float(stress)


def digitize_figure7(
    source_dir: Path = DEFAULT_SOURCE,
) -> tuple[
    Dict[str, Dict[str, List[Dict[str, float]]]],
    List[Dict[str, object]],
]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "UAV-arm curve digitization requires Pillow"
        ) from exc

    image = Image.open(Path(source_dir) / "figure-007.png")
    rgb = np.asarray(image.convert("RGB"))
    hsv = np.asarray(image.convert("HSV"))
    if rgb.shape != (2459, 2790, 3):
        raise ValueError(f"Figure 7 dimensions changed: {rgb.shape}")
    masks = _color_masks(hsv)
    curves: Dict[str, Dict[str, List[Dict[str, float]]]] = {}
    validation: List[Dict[str, object]] = []

    for geometry, definition in FIGURE7_PANELS.items():
        geometry_curves = {}
        geometry_peaks = []
        for sample_id in SAMPLE_IDS:
            mask = masks[sample_id].copy()
            # Figure legends reuse the curve colors inside every panel.
            mask[
                definition["y_top"] + 35:definition["y_top"] + 220,
                definition["x_left"] + 630:definition["x_right"] - 20,
            ] = False
            pixel_points = []
            observed_columns = []
            for x_pixel in range(
                definition["x_left"] + 1,
                definition["x_right"],
            ):
                candidates = (
                    np.flatnonzero(
                        mask[
                            definition["y_top"] + 1:
                            definition["y_bottom"],
                            x_pixel,
                        ]
                    )
                    + definition["y_top"]
                    + 1
                )
                if len(candidates) < 2:
                    continue
                groups = np.split(
                    candidates,
                    np.where(np.diff(candidates) > 2)[0] + 1,
                )
                retained = [group for group in groups if len(group) >= 2]
                if not retained:
                    continue
                observed_columns.append(x_pixel)
                for group in retained:
                    pixel_points.append(
                        (float(x_pixel), float(np.median(group)))
                    )
            pixel_points.sort()
            points = []
            for x_pixel, y_pixel in pixel_points:
                strain, stress = _pixel_to_curve(
                    x_pixel, y_pixel, definition
                )
                points.append(
                    {
                        "engineering_strain": strain,
                        "engineering_stress_MPa": stress,
                        "source_pixel_x": x_pixel,
                        "source_pixel_y": y_pixel,
                    }
                )
            if len(points) < 600:
                raise ValueError(
                    f"too few visible Figure 7 points for "
                    f"{geometry}/{sample_id}"
                )
            peak = max(
                float(point["engineering_stress_MPa"])
                for point in points
            )
            geometry_peaks.append(peak)
            gaps = np.diff(np.asarray(observed_columns, dtype=float))
            maximum_gap_strain = (
                max(0.0, float(np.max(gaps)) - 1.0)
                / (
                    definition["x_right"]
                    - definition["x_left"]
                )
                * 0.10
            )
            visible_span = observed_columns[-1] - observed_columns[0] + 1
            validation.append(
                {
                    "geometry": geometry,
                    "sample_id": sample_id,
                    "digitized_peak_stress_MPa": peak,
                    "digitized_visible_start_strain": points[0][
                        "engineering_strain"
                    ],
                    "digitized_visible_end_strain": points[-1][
                        "engineering_strain"
                    ],
                    "digitized_points": len(points),
                    "source_columns_observed": len(
                        set(observed_columns)
                    ),
                    "observed_column_fraction_of_visible_span": (
                        len(set(observed_columns)) / visible_span
                    ),
                    "maximum_unobserved_gap_strain": maximum_gap_strain,
                    "passes_visible_trace_gate": (
                        maximum_gap_strain <= 0.01
                    ),
                    "evidence_class": (
                        "published physical specimen trace digitized from "
                        "Figure 7; overlapping colored segments remain "
                        "unobserved rather than reconstructed"
                    ),
                }
            )
            geometry_curves[sample_id] = points
        average_peak = float(np.mean(geometry_peaks))
        table_peak = TABLE3[geometry][
            "tensile_strength_mean_MPa"
        ]
        relative_error = average_peak / table_peak - 1.0
        for row in validation:
            if row["geometry"] == geometry:
                row["three_trace_average_peak_MPa"] = average_peak
                row["reported_group_tensile_strength_MPa"] = table_peak
                row["average_peak_relative_error"] = relative_error
                row["passes_table_crosscheck"] = (
                    abs(relative_error) <= 0.02
                )
        curves[geometry] = geometry_curves
    return curves, validation


def _write_csv(
    path: Path, rows: Sequence[Mapping[str, object]]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_reference(
    source_dir: Path = DEFAULT_SOURCE,
    output_dir: Path = DEFAULT_OUTPUT,
) -> Dict[str, object]:
    checksums = verify_sources(source_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    curves_dir = output_dir / "curves"
    curves_dir.mkdir(exist_ok=True)
    curves, validation = digitize_figure7(source_dir)

    for geometry, specimen_curves in curves.items():
        for sample_id, points in specimen_curves.items():
            rows = [
                {
                    "material": "Bambu PLA Basic",
                    "component": "land-air UAV rotary arm",
                    "geometry": geometry,
                    "sample_id": sample_id,
                    "partition": "external_heldout_only",
                    **point,
                }
                for point in points
            ]
            _write_csv(
                curves_dir / f"{geometry}_{sample_id}.csv",
                rows,
            )
    _write_csv(output_dir / "curve_validation.csv", validation)
    table_rows = [
        {"geometry": geometry, **values}
        for geometry, values in TABLE3.items()
    ]
    _write_csv(output_dir / "table3_summary.csv", table_rows)
    _write_csv(
        output_dir / "author_fea_summary.csv",
        [
            {
                "geometry": geometry,
                "author_peak_von_mises_at_500N_MPa": stress,
                **AUTHOR_FEA,
            }
            for geometry, stress in AUTHOR_FEA_PEAK_STRESS_MPA.items()
        ],
    )

    peak_crosschecks = {}
    for geometry in FIGURE7_PANELS:
        rows = [
            row for row in validation if row["geometry"] == geometry
        ]
        peak_crosschecks[geometry] = {
            "trace_count": len(rows),
            "digitized_average_peak_MPa": rows[0][
                "three_trace_average_peak_MPa"
            ],
            "reported_group_strength_MPa": rows[0][
                "reported_group_tensile_strength_MPa"
            ],
            "relative_error": rows[0]["average_peak_relative_error"],
        }
    provenance = {
        "article": (
            "Topology Optimization Design and Performance Comparison of "
            "a Land-Air UAV Rotary Arm Integrating Additive "
            "Manufacturing Constraints"
        ),
        "journal": "Machines",
        "year": 2026,
        "doi": ARTICLE_DOI,
        "url": ARTICLE_URL,
        "license": "CC BY 4.0",
        "checksums": checksums,
        "material_identity": "Bambu PLA Basic",
        "print_process": PRINT_PROCESS,
        "test_process": TEST_PROCESS,
        "author_fea": AUTHOR_FEA,
        "figure7_pixel_dimensions": [2790, 2459],
        "figure7_panel_axis_bounds_pixels": FIGURE7_PANELS,
        "physical_specimen_curve_count": 12,
        "peak_crosschecks": peak_crosschecks,
        "curve_quality_gate_passed": all(
            bool(row["passes_visible_trace_gate"])
            and bool(row["passes_table_crosscheck"])
            for row in validation
        ),
        "allowed_use": (
            "independent external held-out component response and topology "
            "ranking evidence; prohibited from material calibration and "
            "model selection"
        ),
        "limitations": [
            "Machine-exported load/displacement curves are available only "
            "on request and are not public.",
            "The reconstructed CAD/STEP and exact test geometry are not "
            "published, so this repository does not claim an exact FEA "
            "replay.",
            "Overlapped colored curve segments are retained as visibility "
            "gaps and are not reconstructed.",
            "The paper's section text omits the numeric specimen count, "
            "while Figure 7 visibly labels three specimen traces per "
            "geometry.",
            "The study uses a structural stress-strain reduction; it does "
            "not identify local-1/local-2/local-3 material constants.",
            "The author FEA uses an isotropic E=3000 MPa and nu=0.35 card "
            "rather than a measured orthotropic law.",
        ],
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    max_error = max(
        abs(float(value["relative_error"]))
        for value in peak_crosschecks.values()
    )
    report = f"""# Exact-brand UAV-arm component reference

The peer-reviewed source prints Bambu PLA Basic UAV rotary-arm specimens in
four geometries: topology optimized, honeycomb, central lightening hole, and a
solid baseline. Figure 7 contains three physical stress-strain traces for each
geometry. All 12 visible traces are digitized without reconstructing
overplotted segments. Their geometry-level mean peak stresses cross-check
Table 3 within {max_error:.2%}.

This is valuable independent component-level evidence because the geometries
were not used to fit this repository's material cards. It is not an exact
simulation validation: the article says underlying data are available on
request, and neither machine-exported curves nor reconstructed CAD/STEP files
are public. The curves therefore remain `external_heldout_only` and cannot
enter material calibration or model selection.

The paper's own Abaqus comparison uses an isotropic E=3000 MPa, nu=0.35 card,
a 500 N hole-to-hole tensile load, and a 3.0/2.0/1.5/0.75 mm mesh schedule.
Those author settings are preserved as provenance, not promoted as the
canonical Bambu PLA constitutive law.
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    return provenance


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build exact-brand UAV-arm component curve reference"
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = write_reference(args.source, args.output)
    print(
        "wrote 12 UAV-arm specimen curves; "
        f"curve gate={result['curve_quality_gate_passed']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
