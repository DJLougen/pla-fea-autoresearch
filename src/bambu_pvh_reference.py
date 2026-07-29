from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT_ROOT / "data/public/zhu-pvh-metamaterial-2025"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "data/reported/bambu_pla_pvh_metamaterial_2025"
)
ARTICLE_TITLE = (
    "Bio-inspired perturbated hierarchical mechanical metamaterial "
    "for energy absorption"
)
ARTICLE_DOI = "10.1016/j.ijmecsci.2025.110847"
ARTICLE_URL = f"https://doi.org/{ARTICLE_DOI}"
SUPPLEMENT_URL = (
    "https://ars.els-cdn.com/content/image/"
    "1-s2.0-S0020740325009294-mmc1.pdf"
)
SOURCE_SHA256 = {
    "supplement.pdf": (
        "4f8c2bb9498d1537ada4650ff9a322ef778a0660ce66baa1ddb82ce849362a7d"
    ),
    "figure-s3-tensile.png": (
        "f49b5fb803762c0757262877196ca58aaf37ac6ef4c225eaf042e39654e4ec16"
    ),
    "figure-s5-compression.png": (
        "25880c6bf1c6e20dae11eadc458b6665aa7f11300efad97f42585eefea2e4d75"
    ),
}

TENSILE_PLOT = {
    "left": 147,
    "right": 962,
    "top": 44,
    "bottom": 614,
    "x_max": 0.04,
    "y_max": 40.0,
}
COMPRESSION_PLOT = {
    "left": 183,
    "right": 1006,
    "top": 54,
    "bottom": 667,
    "x_max": 0.9,
    "y_max": 3.0,
}

OPTIMUM_VALIDATION = (
    ("high_level_optimum", "elastic_modulus_MPa", 68.94, 76.90, 5.32),
    ("high_level_optimum", "compressive_strength_MPa", 1.52, 1.55, 0.34),
    ("medium_level_optimum", "elastic_modulus_MPa", 21.59, 23.19, 0.24),
    ("medium_level_optimum", "compressive_strength_MPa", 0.62, 0.66, 0.01),
    ("baseline_honeycomb", "elastic_modulus_MPa", 25.15, 28.28, 0.88),
    ("baseline_honeycomb", "compressive_strength_MPa", 0.75, 0.90, 0.03),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(
    path: Path, rows: Sequence[Mapping[str, object]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def verify_source(
    source_dir: Path = DEFAULT_SOURCE,
) -> Dict[str, Dict[str, object]]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "PVH figure digitization requires Pillow"
        ) from exc
    source_dir = Path(source_dir)
    evidence: Dict[str, Dict[str, object]] = {}
    for filename, expected in SOURCE_SHA256.items():
        path = source_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"missing PVH source asset: {path}")
        observed = _sha256(path)
        if observed != expected:
            raise ValueError(
                f"{filename} checksum changed: expected {expected}, "
                f"got {observed}"
            )
        item: Dict[str, object] = {"sha256": observed}
        if path.suffix.lower() == ".png":
            item["pixel_dimensions"] = list(Image.open(path).size)
        evidence[filename] = item
    return evidence


def digitize_tensile_sample_3(
    image_path: Path,
) -> List[Dict[str, float]]:
    from PIL import Image

    rgb = np.asarray(Image.open(image_path).convert("RGB")).astype(int)
    red, green, blue = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    teal = (
        (red < 190)
        & (green > 140)
        & (blue > 130)
        & ((green - red) > 15)
    )
    left = int(TENSILE_PLOT["left"])
    right = int(TENSILE_PLOT["right"])
    top = int(TENSILE_PLOT["top"])
    bottom = int(TENSILE_PLOT["bottom"])
    rows: List[Dict[str, float]] = []
    for x_pixel in range(left + 1, right):
        candidates = (
            np.flatnonzero(teal[top + 1:bottom, x_pixel]) + top + 1
        )
        # Remove the teal legend stroke; the physical curve lies much lower
        # in this x range.
        if x_pixel < 300:
            candidates = candidates[candidates > 250]
        if len(candidates) == 0:
            continue
        y_pixel = float(np.median(candidates))
        rows.append(
            {
                "engineering_strain": (
                    (x_pixel - left)
                    / (right - left)
                    * float(TENSILE_PLOT["x_max"])
                ),
                "engineering_stress_MPa": (
                    (bottom - y_pixel)
                    / (bottom - top)
                    * float(TENSILE_PLOT["y_max"])
                ),
                "source_pixel_x": float(x_pixel),
                "source_pixel_y": y_pixel,
            }
        )
    return rows


def digitize_compression(
    image_path: Path,
) -> Dict[str, List[Dict[str, float]]]:
    from PIL import Image

    rgb = np.asarray(Image.open(image_path).convert("RGB")).astype(int)
    red, green, blue = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    masks = {
        "experiment": (
            (red > 150)
            & (green < 130)
            & (blue < 130)
            & ((red - green) > 70)
        ),
        "simulation": (red < 80) & (green < 80) & (blue < 80),
    }
    left = int(COMPRESSION_PLOT["left"])
    right = int(COMPRESSION_PLOT["right"])
    top = int(COMPRESSION_PLOT["top"])
    bottom = int(COMPRESSION_PLOT["bottom"])
    # The published traces end near 0.5 strain; stopping there excludes the
    # legend and most of the inset photograph.
    trace_end = int(
        round(
            left
            + 0.5
            / float(COMPRESSION_PLOT["x_max"])
            * (right - left)
        )
    )
    curves: Dict[str, List[Dict[str, float]]] = {}
    for name, mask in masks.items():
        rows: List[Dict[str, float]] = []
        for x_pixel in range(left + 3, trace_end + 1):
            candidates = (
                np.flatnonzero(
                    mask[top + 8:bottom - 8, x_pixel]
                )
                + top
                + 8
            )
            if x_pixel >= 570:
                candidates = candidates[candidates < 285]
            if name == "simulation" and x_pixel < 210:
                candidates = candidates[candidates > 250]
            if len(candidates) == 0:
                continue
            y_pixel = float(np.median(candidates))
            rows.append(
                {
                    "compressive_strain": (
                        (x_pixel - left)
                        / (right - left)
                        * float(COMPRESSION_PLOT["x_max"])
                    ),
                    "compressive_stress_MPa": (
                        (bottom - y_pixel)
                        / (bottom - top)
                        * float(COMPRESSION_PLOT["y_max"])
                    ),
                    "source_pixel_x": float(x_pixel),
                    "source_pixel_y": y_pixel,
                }
            )
        curves[name] = rows
    return curves


def _origin_slope(
    strain: np.ndarray,
    stress: np.ndarray,
    lower: float,
    upper: float,
) -> float:
    selected = (strain >= lower) & (strain <= upper)
    return float(
        np.dot(strain[selected], stress[selected])
        / np.dot(strain[selected], strain[selected])
    )


def validate_digitization(
    tensile: Sequence[Mapping[str, float]],
    compression: Mapping[str, Sequence[Mapping[str, float]]],
) -> Dict[str, object]:
    tensile_strain = np.asarray(
        [row["engineering_strain"] for row in tensile], dtype=float
    )
    tensile_stress = np.asarray(
        [row["engineering_stress_MPa"] for row in tensile], dtype=float
    )
    tensile_slope = _origin_slope(
        tensile_strain, tensile_stress, 0.003, 0.010
    )

    common_strain = np.linspace(0.005, 0.45, 446)
    metrics: Dict[str, Dict[str, float]] = {}
    interpolated: Dict[str, np.ndarray] = {}
    for name, rows in compression.items():
        strain = np.asarray(
            [row["compressive_strain"] for row in rows], dtype=float
        )
        stress = np.asarray(
            [row["compressive_stress_MPa"] for row in rows], dtype=float
        )
        values = np.interp(common_strain, strain, stress)
        interpolated[name] = values
        first_peak = (common_strain >= 0.005) & (common_strain <= 0.08)
        plateau = (common_strain >= 0.10) & (common_strain <= 0.25)
        x_pixels = np.asarray(
            [row["source_pixel_x"] for row in rows], dtype=float
        )
        metrics[name] = {
            "digitized_points": float(len(rows)),
            "maximum_horizontal_gap_px": float(np.max(np.diff(x_pixels))),
            "maximum_digitized_strain": float(np.max(strain)),
            "initial_peak_stress_MPa": float(np.max(values[first_peak])),
            "mean_plateau_stress_0p10_to_0p25_MPa": float(
                np.mean(values[plateau])
            ),
            "energy_density_to_0p45_MJ_m3": float(
                np.trapezoid(values, common_strain)
            ),
        }
    experiment = interpolated["experiment"]
    simulation = interpolated["simulation"]
    peak_error = (
        metrics["simulation"]["initial_peak_stress_MPa"]
        / metrics["experiment"]["initial_peak_stress_MPa"]
        - 1.0
    )
    energy_error = (
        metrics["simulation"]["energy_density_to_0p45_MJ_m3"]
        / metrics["experiment"]["energy_density_to_0p45_MJ_m3"]
        - 1.0
    )
    normalized_rmse = float(
        np.sqrt(np.mean((simulation - experiment) ** 2))
        / np.max(experiment)
    )
    quality_gate = (
        len(tensile) >= 700
        and abs(tensile_slope / 2140.0 - 1.0) <= 0.05
        and all(
            current["digitized_points"] >= 400
            and current["maximum_horizontal_gap_px"] <= 6
            and current["maximum_digitized_strain"] >= 0.48
            for current in metrics.values()
        )
        and abs(peak_error) <= 0.15
        and abs(energy_error) <= 0.10
    )
    return {
        "quality_gate_passed": bool(quality_gate),
        "tensile_sample_3_points": len(tensile),
        "tensile_sample_3_origin_slope_MPa": tensile_slope,
        "tensile_slope_error_vs_reported_2140": (
            tensile_slope / 2140.0 - 1.0
        ),
        "compression": metrics,
        "simulation_to_experiment_initial_peak_error": peak_error,
        "simulation_to_experiment_energy_to_0p45_error": energy_error,
        "simulation_experiment_normalized_rmse_to_0p45": normalized_rmse,
        "simulation_experiment_pearson_to_0p45": float(
            np.corrcoef(simulation, experiment)[0, 1]
        ),
    }


def _author_model() -> Dict[str, object]:
    return {
        "material": "Bambu Lab PLA Basic",
        "printer": "Bambu Lab X1 Carbon",
        "material_card": {
            "youngs_modulus_MPa": 2140.0,
            "yield_stress_MPa": 36.0,
            "yield_definition": "0.2% offset",
            "poisson_ratio": 0.35,
            "density_g_cm3": 1.2,
            "constitutive_interpretation": (
                "elastic-plastic author card; post-yield table is not "
                "published"
            ),
        },
        "tensile_test": {
            "standard": "ASTM D638-14",
            "replicates": 3,
        },
        "printing": {
            "nozzle_diameter_mm": 0.2,
            "nozzle_temperature_C": 220,
            "flow_ratio": 0.98,
            "validation_design_wall_thickness_mm": 0.4,
            "measured_mean_wall_thickness_mm": 0.41,
        },
        "geometry": {
            "class": "perturbated vertex-based hierarchical honeycomb",
            "large_hexagon_side_length_mm": 3.0,
            "rve_cells_x": 15,
            "rve_cells_y": 15,
            "relative_density": 0.2,
            "baseline_wall_thickness_mm": 0.5,
            "vertex_hexagon_side_length_range_mm": [0.0, 1.0],
            "perturbation_magnitude_range_mm": [0.0, 0.5],
            "validation_vertex_hexagon_side_length_mm": 1.0,
            "validation_perturbation_magnitude_mm": 0.5,
        },
        "finite_element_model": {
            "software": "Abaqus Dynamic/Explicit",
            "element": "S4R shell",
            "selected_element_size_mm": 0.5,
            "contact_normal": "hard",
            "contact_friction_coefficient": 0.2,
            "loading_velocity_m_s": 0.6,
            "quasistatic_gate": "kinetic energy below 5% of internal energy",
        },
        "source_conflicts": [
            (
                "The main article reports 5 mm/min experimental compression; "
                "Supplement A7 reports 2 mm/min."
            ),
            (
                "The main author card uses 1.2 g/cm3; Supplement A7 uses "
                "1.25 g/cm3 in the steady-shock calculation."
            ),
        ],
    }


def import_reference(
    source_dir: Path = DEFAULT_SOURCE,
    output_dir: Path = DEFAULT_OUTPUT,
) -> Dict[str, object]:
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_files = verify_source(source_dir)
    tensile = digitize_tensile_sample_3(
        source_dir / "figure-s3-tensile.png"
    )
    compression = digitize_compression(
        source_dir / "figure-s5-compression.png"
    )
    validation = validate_digitization(tensile, compression)
    if not validation["quality_gate_passed"]:
        raise RuntimeError(
            "PVH reference digitization failed its quality gate"
        )

    _write_csv(output_dir / "curves/tensile_sample_3.csv", tensile)
    for name, rows in compression.items():
        _write_csv(
            output_dir / f"curves/compression_{name}.csv", rows
        )

    optimum_rows = []
    for design, metric, fea, experimental, standard_deviation in (
        OPTIMUM_VALIDATION
    ):
        optimum_rows.append(
            {
                "design": design,
                "metric": metric,
                "fea_value": fea,
                "experimental_mean": experimental,
                "experimental_standard_deviation": standard_deviation,
                "absolute_relative_error": abs(fea / experimental - 1.0),
            }
        )
    _write_csv(output_dir / "reported_optimum_validation.csv", optimum_rows)

    replay_gate = [
        {
            "requirement": "large-cell and hierarchical topology",
            "available": True,
            "evidence": "main Figure 1 and Supplement A5-A6",
        },
        {
            "requirement": "design-variable bounds and validation point",
            "available": True,
            "evidence": "Supplement A5-A7",
        },
        {
            "requirement": "random perturbation direction vector or seed",
            "available": False,
            "evidence": "not published",
        },
        {
            "requirement": "exact CAD, mesh, or Abaqus input file",
            "available": False,
            "evidence": "data available only on author request",
        },
        {
            "requirement": "post-yield PLA hardening curve",
            "available": False,
            "evidence": "only E and 0.2%-offset yield are stated",
        },
    ]
    _write_csv(output_dir / "exact_replay_gate.csv", replay_gate)
    exact_replay_allowed = all(
        bool(row["available"]) for row in replay_gate
    )
    author_model = _author_model()
    (output_dir / "author_model.json").write_text(
        json.dumps(author_model, indent=2) + "\n", encoding="utf-8"
    )
    provenance = {
        "article": {
            "title": ARTICLE_TITLE,
            "authors": [
                "Yinan Zhu",
                "Zijue Chen",
                "Yuman Zhu",
                "Matthieu Gresil",
                "Yunlong Tang",
            ],
            "journal": "International Journal of Mechanical Sciences",
            "volume": 307,
            "article_number": "110847",
            "year": 2025,
            "doi": ARTICLE_DOI,
            "url": ARTICLE_URL,
            "license": "CC BY 4.0",
        },
        "supplement_url": SUPPLEMENT_URL,
        "source_files": source_files,
        "quality_gate_passed": True,
        "exact_specimen_replay_allowed": exact_replay_allowed,
        "exact_specimen_replay_exclusion_reason": (
            "The random perturbation realization, model files, and "
            "post-yield material table are not published."
        ),
        "digitization_validation": validation,
        "limitations": [
            (
                "Only the visually separable teal tensile trace is "
                "digitized; the other two overlap it for much of the plot."
            ),
            (
                "Figure S5 is an author simulation-versus-experiment "
                "validation, not a new CalculiX reproduction."
            ),
            (
                "The independent material card is appropriate as a "
                "sensitivity case, not as proof that every Bambu PLA Basic "
                "spool or print orientation is identical."
            ),
        ],
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    report = f"""# Exact-brand PVH metamaterial reference

This import preserves an independent peer-reviewed Bambu PLA Basic study:
[{ARTICLE_TITLE}]({ARTICLE_URL}), including its checksum-pinned supplement.

## What is independently supported

- Three ASTM D638-14 tensile tests give **E = 2140 MPa** and **0.2%-offset
  yield = 36 MPa**. The visible teal trace digitizes to
  **{validation['tensile_sample_3_origin_slope_MPa']:.1f} MPa**.
- The published validation specimen uses a 15 x 15 PVH honeycomb, 20%
  relative density, 1.0 mm vertex hexagons, 0.5 mm perturbation magnitude,
  and a nominal 0.4 mm wall printed with a 0.2 mm nozzle.
- The digitized author FEA and physical compression curves differ by
  **{validation['simulation_to_experiment_initial_peak_error']:+.1%}** at
  the first peak and **{validation['simulation_to_experiment_energy_to_0p45_error']:+.1%}**
  in energy through 0.45 strain.
- The six reported optimum/baseline modulus and strength comparisons have
  FEA-to-test errors from **{min(row['absolute_relative_error'] for row in optimum_rows):.1%}**
  to **{max(row['absolute_relative_error'] for row in optimum_rows):.1%}**.

## Exact-replay boundary

An exact specimen replay is deliberately excluded. The publication does not
release the random perturbation directions/seed, CAD or Abaqus files, or the
post-yield PLA hardening table. Those omissions change the collapse sequence,
so inventing them would make a visually similar structure rather than the
reported specimen.

The author material card is exported for a reproducible transfer sensitivity
on this project's explicit infill geometries.
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import the exact-brand PVH metamaterial reference"
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = import_reference(args.source, args.output)
    print(
        json.dumps(
            {
                "quality_gate_passed": result["quality_gate_passed"],
                "exact_specimen_replay_allowed": result[
                    "exact_specimen_replay_allowed"
                ],
                "output": str(args.output.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
