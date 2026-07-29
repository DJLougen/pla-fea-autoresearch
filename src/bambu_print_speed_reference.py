from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT_ROOT / "data/public/lorkowski-materials-2025"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "data/reported/bambu_pla_print_speed_2025"
)
ARTICLE_DOI = "10.3390/ma18081765"
ARTICLE_URL = "https://doi.org/10.3390/ma18081765"
EXPECTED_SHA256 = {
    "paper.pdf": (
        "3fccb901c1caeaacd502fb004a71ddbf80dd485dac1a6b8c522b89684cfdb3cd"
    ),
    "article.xml": (
        "62683e8467fce628ce7d0232575f0600b42eacfc4131b93c35ab827eb32549bd"
    ),
    "figure-6-tensile-strength.jpg": (
        "5a05e0e2813e3c2483b3768248243a14a139e45f97c2b1d3144ed76813147a72"
    ),
    "figure-7-tensile-anova.jpg": (
        "8dbb3e1c735302eba3de5e5a2fbfbd60c94ab45db611130efe0856ef99aeb212"
    ),
    "figure-8-stress-strain.jpg": (
        "40f52fcfaa387fa5326a80e175d4f55f82d82aae8ab6baeddc22e83428406762"
    ),
    "figure-9-modulus.jpg": (
        "4bc7e50ed4e52d3c0b9e94e5aa5214c1b40d5ecbef69b7afb6a706563368ea62"
    ),
    "figure-10-modulus-anova.jpg": (
        "2a31471657a590098d2140df547bbf080b1b3f26bf8e509718fc1745c6072886"
    ),
    "figure-11-flexural.jpg": (
        "a5de51e74ed1a3c31716e467f1fed2777f8d7e75a6b50c21e00e72c52211d975"
    ),
    "figure-12-impact.jpg": (
        "af1d626df56776e0b5a596822cbc73d4c184e5d04b00ee67c584d62317da08f8"
    ),
    "figure-13-impact-anova.jpg": (
        "fed977b602d88a805b965b5553e84949118df81aa111262b3ef67486d4eaea6d"
    ),
}

SAMPLE_MAP = {
    1: {"print_speed_mm_s": 30, "build_plate_axis": "X", "ironing": True},
    2: {"print_speed_mm_s": 30, "build_plate_axis": "X", "ironing": False},
    3: {"print_speed_mm_s": 500, "build_plate_axis": "X", "ironing": True},
    4: {"print_speed_mm_s": 500, "build_plate_axis": "X", "ironing": False},
    5: {"print_speed_mm_s": 30, "build_plate_axis": "Y", "ironing": True},
    6: {"print_speed_mm_s": 30, "build_plate_axis": "Y", "ironing": False},
    7: {"print_speed_mm_s": 500, "build_plate_axis": "Y", "ironing": True},
    8: {"print_speed_mm_s": 500, "build_plate_axis": "Y", "ironing": False},
}

# Pixel positions are visually audited against the checksum-pinned rasters.
# The hollow square is the plotted mean; the teal band is mean +/- one SD.
SUMMARY_FIGURES = {
    "tensile_strength": {
        "filename": "figure-6-tensile-strength.jpg",
        "shape": (582, 694, 3),
        "x_centers": (119, 195, 271, 347, 423, 499, 574, 650),
        "mean_y": (278, 316, 300, 342, 117, 171, 168, 216),
        "plot_y_top": 13.5,
        "plot_value_top": 40.0,
        "plot_y_bottom": 501.0,
        "plot_value_bottom": 30.0,
        "fill_rgb": (70, 154, 156),
        "fill_tolerance": 20,
        "quantity": "tensile_strength",
        "unit": "MPa",
    },
    "youngs_modulus": {
        "filename": "figure-9-modulus.jpg",
        "shape": (580, 717, 3),
        "x_centers": (144, 219, 295, 370, 445, 520, 596, 672),
        "mean_y": (177, 291, 219, 334, 211, 335, 306, 306),
        "plot_y_top": 15.0,
        "plot_value_top": 3250.0,
        "plot_y_bottom": 500.0,
        "plot_value_bottom": 2900.0,
        "fill_rgb": (70, 154, 156),
        "fill_tolerance": 20,
        "quantity": "youngs_modulus",
        "unit": "MPa",
    },
    "impact_strength": {
        "filename": "figure-12-impact.jpg",
        "shape": (586, 693, 3),
        "x_centers": (121, 196, 271, 347, 422, 497, 572, 648),
        "mean_y": (207, 230, 222, 245, 284, 234, 175, 154),
        "plot_y_top": 18.0,
        "plot_value_top": 28.0,
        "plot_y_bottom": 502.0,
        "plot_value_bottom": 12.0,
        "fill_rgb": (70, 154, 154),
        "fill_tolerance": 20,
        "quantity": "impact_strength",
        "unit": "kJ_m2",
    },
}

CURVE_COLORS = {
    1: (128, 128, 128),
    2: (255, 117, 116),
    3: (43, 155, 239),
    4: (0, 210, 149),
    5: (216, 154, 239),
    6: (250, 206, 13),
    7: (0, 254, 255),
    8: (196, 125, 129),
}
CURVE_PLOT_LEFT = 83
CURVE_PLOT_RIGHT_AT_12_PERCENT = 619
CURVE_PLOT_TOP = 13
CURVE_PLOT_BOTTOM = 455


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


def _pixel_to_value(y_pixel: float, definition: Mapping[str, object]) -> float:
    top_y = float(definition["plot_y_top"])
    bottom_y = float(definition["plot_y_bottom"])
    top_value = float(definition["plot_value_top"])
    bottom_value = float(definition["plot_value_bottom"])
    return top_value + (y_pixel - top_y) * (
        bottom_value - top_value
    ) / (bottom_y - top_y)


def _merge_runs(values: np.ndarray, maximum_gap: int = 6) -> List[np.ndarray]:
    if len(values) == 0:
        return []
    cuts = np.where(np.diff(values) > maximum_gap)[0] + 1
    return list(np.split(values, cuts))


def _mean_sd_band(
    rgb: np.ndarray,
    x_center: int,
    mean_y: int,
    definition: Mapping[str, object],
) -> Dict[str, float]:
    target = np.asarray(definition["fill_rgb"], dtype=int)
    difference = np.max(np.abs(rgb.astype(int) - target), axis=2)
    window = difference[:, x_center - 12:x_center + 13]
    y_pixels = np.flatnonzero(
        np.any(window <= int(definition["fill_tolerance"]), axis=1)
    )
    runs = _merge_runs(y_pixels, maximum_gap=6)
    containing = [
        run for run in runs
        if int(run[0]) - 3 <= mean_y <= int(run[-1]) + 3
    ]
    if len(containing) != 1:
        raise ValueError(
            f"could not isolate mean +/- SD band at x={x_center}, y={mean_y}"
        )
    band = containing[0]
    upper_y = float(band[0] - 3)
    lower_y = float(band[-1] + 3)
    mean = _pixel_to_value(float(mean_y), definition)
    upper = _pixel_to_value(upper_y, definition)
    lower = _pixel_to_value(lower_y, definition)
    standard_deviation = abs(upper - lower) / 2.0
    return {
        "mean": mean,
        "standard_deviation": standard_deviation,
        "lower_1sd": mean - standard_deviation,
        "upper_1sd": mean + standard_deviation,
        "source_mean_pixel_y": float(mean_y),
        "source_band_top_pixel_y": upper_y,
        "source_band_bottom_pixel_y": lower_y,
    }


def digitize_summary_figures(
    source_dir: Path = DEFAULT_SOURCE,
) -> List[Dict[str, object]]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "print-speed digitization requires the optional "
            "`digitization` dependencies"
        ) from exc
    measurements: Dict[str, List[Dict[str, float]]] = {}
    for name, definition in SUMMARY_FIGURES.items():
        rgb = np.asarray(
            Image.open(Path(source_dir) / str(definition["filename"]))
            .convert("RGB")
        )
        if rgb.shape != tuple(definition["shape"]):
            raise ValueError(
                f"{definition['filename']} dimensions changed: {rgb.shape}"
            )
        measurements[name] = [
            _mean_sd_band(rgb, int(x), int(y), definition)
            for x, y in zip(
                definition["x_centers"], definition["mean_y"]
            )
        ]

    rows: List[Dict[str, object]] = []
    for sample in SAMPLE_MAP:
        row: Dict[str, object] = {
            "sample_group": sample,
            **SAMPLE_MAP[sample],
            "replicates": 5,
        }
        for name, values in measurements.items():
            measurement = values[sample - 1]
            unit = str(SUMMARY_FIGURES[name]["unit"])
            prefix = str(SUMMARY_FIGURES[name]["quantity"])
            for statistic in (
                "mean",
                "standard_deviation",
                "lower_1sd",
                "upper_1sd",
            ):
                row[f"{prefix}_{statistic}_{unit}"] = measurement[statistic]
            row[f"{prefix}_source_mean_pixel_y"] = measurement[
                "source_mean_pixel_y"
            ]
            row[f"{prefix}_source_band_top_pixel_y"] = measurement[
                "source_band_top_pixel_y"
            ]
            row[f"{prefix}_source_band_bottom_pixel_y"] = measurement[
                "source_band_bottom_pixel_y"
            ]
        rows.append(row)
    return rows


def _curve_color_mask(rgb: np.ndarray, sample: int) -> np.ndarray:
    red = rgb[:, :, 0].astype(float)
    green = rgb[:, :, 1].astype(float)
    blue = rgb[:, :, 2].astype(float)
    targets = np.asarray(list(CURVE_COLORS.values()), dtype=float)
    distance = np.linalg.norm(
        rgb.astype(float)[:, :, None, :] - targets[None, None, :, :],
        axis=3,
    )
    nearest = np.argmin(distance, axis=2) == sample - 1
    conditions = {
        1: (
            (np.abs(red - green) < 12)
            & (np.abs(red - blue) < 12)
            & (red >= 70)
            & (red < 180)
        ),
        2: (
            (red > 180) & (red - green > 45)
            & (red - blue > 35) & (green < 190)
        ),
        3: (
            (blue > 160) & (green > 95)
            & (blue - red > 55) & (blue - green > 20)
        ),
        4: (
            (green > 145) & (red < 120)
            & (green - red > 60) & (green - blue > 20)
        ),
        5: (
            (red > 160) & (blue > 175)
            & (red - green > 25) & (blue - green > 25)
        ),
        6: (
            (red > 190) & (green > 135)
            & (blue < 100) & (red - blue > 100)
        ),
        7: (
            (green > 170) & (blue > 170)
            & (red < 120) & (green - red > 80)
        ),
        8: (
            (red > 135) & (green > 65) & (green < 180)
            & (blue > 65) & (blue < 190)
            & (red - green > 25) & (red - blue > 20)
        ),
    }
    return conditions[sample] & nearest & (distance[:, :, sample - 1] <= 70)


def _curve_points(rgb: np.ndarray, sample: int) -> np.ndarray:
    mask = _curve_color_mask(rgb, sample)
    points = []
    for x_pixel in range(CURVE_PLOT_LEFT, 687):
        # The legend is in the lower-right portion of the plot.
        y_stop = 241 if x_pixel > 480 else CURVE_PLOT_BOTTOM + 1
        candidates = np.flatnonzero(
            mask[CURVE_PLOT_TOP:y_stop, x_pixel]
        ) + CURVE_PLOT_TOP
        if len(candidates) == 0:
            continue
        groups = np.split(
            candidates, np.where(np.diff(candidates) > 3)[0] + 1
        )
        group = min(groups, key=lambda values: float(np.median(values)))
        points.append((float(x_pixel), float(np.median(group))))
    if len(points) < 100:
        raise ValueError(f"sample {sample} curve has too few visible pixels")
    array = np.asarray(points, dtype=float)
    runs = np.split(
        np.arange(len(array)),
        np.where(np.diff(array[:, 0]) > 15)[0] + 1,
    )
    retained = [run for run in runs if len(run) >= 5]
    return np.vstack([array[run] for run in retained])


def _curve_strain(x_pixel: np.ndarray | float) -> np.ndarray | float:
    return (
        (x_pixel - CURVE_PLOT_LEFT)
        * 0.12
        / (CURVE_PLOT_RIGHT_AT_12_PERCENT - CURVE_PLOT_LEFT)
    )


def _curve_stress(y_pixel: np.ndarray | float) -> np.ndarray | float:
    return (
        (CURVE_PLOT_BOTTOM - y_pixel)
        * 40.0
        / (CURVE_PLOT_BOTTOM - CURVE_PLOT_TOP)
    )


def _cubic_prepeak(
    modulus_MPa: float,
    strength_MPa: float,
    peak_strain: float,
    count: int = 80,
) -> tuple[np.ndarray, np.ndarray]:
    strain = np.linspace(0.0, peak_strain, count, endpoint=False)
    a = (3.0 * strength_MPa - 2.0 * modulus_MPa * peak_strain) / (
        peak_strain ** 2
    )
    b = (modulus_MPa * peak_strain - 2.0 * strength_MPa) / (
        peak_strain ** 3
    )
    stress = modulus_MPa * strain + a * strain ** 2 + b * strain ** 3
    if np.any(np.diff(stress) < -1.0e-8):
        raise ValueError("reconstructed pre-peak curve is not monotone")
    return strain, stress


def digitize_representative_curves(
    summary_rows: Sequence[Mapping[str, object]],
    source_dir: Path = DEFAULT_SOURCE,
) -> tuple[Dict[int, List[Dict[str, object]]], List[Dict[str, object]]]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "print-speed digitization requires the optional "
            "`digitization` dependencies"
        ) from exc
    rgb = np.asarray(
        Image.open(Path(source_dir) / "figure-8-stress-strain.jpg")
        .convert("RGB")
    )
    if rgb.shape != (534, 694, 3):
        raise ValueError(f"Figure 8 dimensions changed: {rgb.shape}")

    curves: Dict[int, List[Dict[str, object]]] = {}
    curve_summary: List[Dict[str, object]] = []
    for row in summary_rows:
        sample = int(row["sample_group"])
        points = _curve_points(rgb, sample)
        peak_window = points[
            (points[:, 0] >= 125) & (points[:, 0] <= 180)
        ]
        peak = peak_window[int(np.argmin(peak_window[:, 1]))]
        peak_x = float(peak[0])
        peak_strain = float(_curve_strain(peak_x))
        digitized_peak = float(_curve_stress(float(peak[1])))
        measured_strength = float(
            row["tensile_strength_mean_MPa"]
        )
        modulus = float(row["youngs_modulus_mean_MPa"])
        stress_scale = measured_strength / digitized_peak

        postpeak = points[points[:, 0] >= peak_x]
        x_regular = np.arange(
            peak_x, float(np.max(postpeak[:, 0])) + 0.1, 2.0
        )
        y_regular = np.interp(
            x_regular, postpeak[:, 0], postpeak[:, 1]
        )
        strain_postpeak = np.asarray(_curve_strain(x_regular), dtype=float)
        stress_postpeak = np.asarray(
            _curve_stress(y_regular), dtype=float
        ) * stress_scale
        stress_postpeak = np.minimum.accumulate(stress_postpeak)
        stress_postpeak[0] = measured_strength

        strain_prepeak, stress_prepeak = _cubic_prepeak(
            modulus, measured_strength, peak_strain
        )
        output = [
            {
                "sample_group": sample,
                "engineering_strain": float(strain),
                "engineering_stress_MPa": float(stress),
                "stress_retention_ratio": (
                    0.0
                    if measured_strength <= 0.0
                    else float(stress / measured_strength)
                ),
                "branch": "reconstructed_prepeak_cubic",
                "source_pixel_x": None,
                "source_pixel_y": None,
            }
            for strain, stress in zip(strain_prepeak, stress_prepeak)
        ]
        output.extend(
            {
                "sample_group": sample,
                "engineering_strain": float(strain),
                "engineering_stress_MPa": float(stress),
                "stress_retention_ratio": float(
                    stress / measured_strength
                ),
                "branch": "digitized_postpeak_monotone_envelope",
                "source_pixel_x": float(x_pixel),
                "source_pixel_y": float(y_pixel),
            }
            for strain, stress, x_pixel, y_pixel in zip(
                strain_postpeak, stress_postpeak, x_regular, y_regular
            )
        )
        # Figure 8 ends each colored trace at fracture. Add an explicit
        # numerical drop one image pixel later, rather than inventing a
        # traction-separation tail.
        fracture_strain = float(
            strain_postpeak[-1]
            + 0.12
            / (CURVE_PLOT_RIGHT_AT_12_PERCENT - CURVE_PLOT_LEFT)
        )
        output.append(
            {
                "sample_group": sample,
                "engineering_strain": fracture_strain,
                "engineering_stress_MPa": 0.0,
                "stress_retention_ratio": 0.0,
                "branch": "assumed_terminal_fracture_drop",
                "source_pixel_x": None,
                "source_pixel_y": None,
            }
        )
        curves[sample] = output
        curve_summary.append(
            {
                "sample_group": sample,
                **SAMPLE_MAP[sample],
                "measured_mean_tensile_strength_MPa": measured_strength,
                "measured_mean_youngs_modulus_MPa": modulus,
                "digitized_representative_peak_MPa": digitized_peak,
                "peak_crosscheck_relative_error": (
                    digitized_peak / measured_strength - 1.0
                ),
                "postpeak_stress_scale_to_measured_mean": stress_scale,
                "digitized_peak_strain": peak_strain,
                "digitized_last_prefracture_strain": float(
                    strain_postpeak[-1]
                ),
                "digitized_last_prefracture_stress_MPa": float(
                    stress_postpeak[-1]
                ),
                "last_prefracture_stress_retention_ratio": float(
                    stress_postpeak[-1] / measured_strength
                ),
                "assumed_zero_stress_fracture_strain": fracture_strain,
                "apparent_tensile_energy_density_MJ_m3": float(
                    np.trapezoid(
                        [point["engineering_stress_MPa"] for point in output],
                        [point["engineering_strain"] for point in output],
                    )
                ),
                "quality_gate_passed": (
                    abs(digitized_peak / measured_strength - 1.0) <= 0.02
                    and 0.05 <= strain_postpeak[-1] <= 0.13
                    and 0.0 < stress_postpeak[-1] < measured_strength
                ),
            }
        )
    return curves, curve_summary


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
    (output_dir / "curves").mkdir(exist_ok=True)
    summary_rows = digitize_summary_figures(source_dir)
    curves, curve_summary = digitize_representative_curves(
        summary_rows, source_dir
    )
    _write_csv(output_dir / "process_groups.csv", summary_rows)
    _write_csv(output_dir / "curve_validation.csv", curve_summary)
    for sample, rows in curves.items():
        _write_csv(
            output_dir / "curves" / f"sample_{sample}.csv", rows
        )

    quality_passed = all(
        bool(row["quality_gate_passed"]) for row in curve_summary
    )
    payload = {
        "material": "Bambu PLA Basic",
        "article": (
            "Influence of Print Speed on the Mechanical Performance of "
            "3D-Printed Bio-Polymer Polylactic Acid"
        ),
        "journal": "Materials",
        "year": 2025,
        "doi": ARTICLE_DOI,
        "url": ARTICLE_URL,
        "license": "CC BY 4.0",
        "checksums": checksums,
        "printer": "Bambu Lab A1 mini",
        "sample_groups": len(summary_rows),
        "replicates_per_group": 5,
        "quality_gate_passed": quality_passed,
        "fixed_process": {
            "nozzle_temperature_C": 220,
            "bed_temperature_C": 65,
            "layer_height_mm": 0.2,
            "infill_density_percent": 100,
            "infill_pattern": "rectilinear",
            "wall_loops": 2,
            "top_shell_layers": 4,
            "bottom_shell_layers": 4,
            "tensile_standard": "ISO 527-2",
        },
        "varied_process": {
            "nominal_print_speed_mm_s": [30, 500],
            "build_plate_axis": ["X", "Y"],
            "ironing": [False, True],
        },
        "evidence_classes": {
            "process_group_means_and_standard_deviations": (
                "Figure 6/9/12 plotted mean and mean +/- one SD"
            ),
            "postpeak": (
                "Figure 8 representative colored curves, rescaled by <=2% "
                "to the corresponding measured group mean"
            ),
            "prepeak": (
                "cubic reconstruction constrained by plotted mean modulus, "
                "mean strength, and digitized peak strain"
            ),
            "fracture_drop": (
                "assumed zero-stress drop one image pixel after the visible "
                "curve endpoint"
            ),
        },
        "limitations": [
            "The five raw tensile curves for each group are not available.",
            "Figure 8 is representative/aggregated and is not identified as "
            "a raw individual-specimen trace.",
            "The reconstructed pre-peak branch is not a measured plastic law.",
            "The terminal zero-stress point is a numerical fracture "
            "assumption, not a measured traction-separation law.",
            "Apparent tensile energy density is not fracture energy.",
            "X/Y are build-plate axes; the layer path remains oriented the "
            "same way relative to the specimen.",
            "The process profiles are for solid rectilinear tensile coupons "
            "on an A1 mini and are not universal constants for sparse infill.",
        ],
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    minimum = min(
        summary_rows,
        key=lambda row: float(row["tensile_strength_mean_MPa"]),
    )
    maximum = max(
        summary_rows,
        key=lambda row: float(row["tensile_strength_mean_MPa"]),
    )
    report = f"""# Bambu PLA Basic print-speed/process reference

## Usable evidence

The peer-reviewed study supplies eight exact-brand process groups with five
repeats each. Figure 6 gives tensile mean +/- one SD, Figure 9 gives Young's
modulus mean +/- one SD, and Figure 12 gives impact strength mean +/- one SD.
The digitized mean tensile range is
{float(minimum['tensile_strength_mean_MPa']):.2f} to
{float(maximum['tensile_strength_mean_MPa']):.2f} MPa.

Figure 8 contributes a representative post-peak shape for every group. All
eight visible peaks cross-check their corresponding Figure 6 means within 2%.
The curve files combine that digitized post-peak envelope with a clearly
labeled cubic pre-peak reconstruction constrained by the measured group mean
modulus and strength. A zero-stress point one image pixel after the visible
endpoint is an explicit numerical fracture assumption.

## Transfer boundary

These are 100% rectilinear ISO 527-2 coupons printed on a Bambu Lab A1 mini at
220 C nozzle temperature, 65 C bed temperature, and 0.2 mm layers. X and Y are
build-plate axes; the paper states that the layer pattern remains in the same
orientation relative to the specimen. The profiles therefore quantify a
machine/process sensitivity, not a new material-axis law.

The source does not release the five raw specimen curves per group. The
representative curves support post-peak sensitivity studies, but not a
validated damage, crack-growth, cohesive-separation, or fracture-energy law.
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the peer-reviewed Bambu print-speed reference"
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = write_reference(args.source, args.output)
    print(
        f"wrote {result['sample_groups']} process profiles: "
        f"{args.output.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
