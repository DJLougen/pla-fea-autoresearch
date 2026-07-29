from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from .bambu_curve_calibration import (
    DEFAULT_DATASET,
    DEFAULT_PROFILE,
    fit_elastic_slope,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "results/bambu_plastic_calibration"
DEFAULT_PLASTIC_PROFILE = (
    PROJECT_ROOT
    / "data/reported/bambu_pla_raw_curve_prepeak_plastic.json"
)


def _prepeak_curve(path: Path) -> Dict[str, object]:
    frame = (
        pd.read_csv(path)
        .drop_duplicates(
            subset=["engineering_strain", "engineering_stress_MPa"]
        )
        .dropna(
            subset=["engineering_strain", "engineering_stress_MPa"]
        )
        .sort_values("engineering_strain", kind="mergesort")
    )
    modulus, intercept, _, _ = fit_elastic_slope(
        frame["engineering_strain"],
        frame["engineering_stress_MPa"],
    )
    strain = frame["engineering_strain"].to_numpy(float)
    # Shift the extensometer zero so the fitted elastic line passes through
    # the origin without altering the measured stress history.
    strain = strain + intercept / modulus
    stress = frame["engineering_stress_MPa"].to_numpy(float)
    peak_index = int(np.argmax(stress))
    mask = (strain >= 0.0) & (
        np.arange(len(strain)) <= peak_index
    )
    return {
        "label": str(frame.iloc[0]["source_specimen_label"]),
        "axis": str(frame.iloc[0]["source_platform_axis"]),
        "placement": str(frame.iloc[0]["source_placement"]),
        "strain": strain[mask],
        "stress": stress[mask],
        "peak_strain": float(strain[peak_index]),
        "peak_stress": float(stress[peak_index]),
        "zero_shift": float(intercept / modulus),
    }


def aggregate_in_plane_prepeak(
    dataset_dir: Path,
    elastic_modulus_MPa: float,
    minimum_coverage_fraction: float = 0.8,
    offset_strain: float = 0.002,
    grid_step: float = 0.000025,
    plastic_points: int = 10,
) -> tuple[List[Dict[str, float]], List[Dict[str, float]], Dict[str, float]]:
    curves = [
        _prepeak_curve(path)
        for path in sorted((Path(dataset_dir) / "curves").glob("*.csv"))
    ]
    # Z-upright is a separate, brittle build-direction population. The present
    # J2 card is only for flat-printed, in-layer infill sensitivity.
    curves = [curve for curve in curves if curve["axis"] != "Z"]
    required = int(math.ceil(minimum_coverage_fraction * len(curves)))
    maximum_peak = max(float(curve["peak_strain"]) for curve in curves)
    grid = np.arange(0.0, maximum_peak + grid_step / 2.0, grid_step)
    aggregate: List[Dict[str, float]] = []
    for strain in grid:
        values = [
            float(
                np.interp(
                    strain,
                    curve["strain"],
                    curve["stress"],
                )
            )
            for curve in curves
            if float(curve["strain"][0]) <= strain
            <= float(curve["peak_strain"])
        ]
        if len(values) < required:
            continue
        aggregate.append(
            {
                "engineering_strain": float(strain),
                "median_engineering_stress_MPa": float(
                    np.median(values)
                ),
                "q10_engineering_stress_MPa": float(
                    np.quantile(values, 0.1)
                ),
                "q90_engineering_stress_MPa": float(
                    np.quantile(values, 0.9)
                ),
                "specimen_coverage": len(values),
            }
        )
    if not aggregate:
        raise ValueError("no strain range met the requested curve coverage")
    baseline = aggregate[0]["median_engineering_stress_MPa"]
    for field in (
        "median_engineering_stress_MPa",
        "q10_engineering_stress_MPa",
        "q90_engineering_stress_MPa",
    ):
        values = np.maximum(
            0.0,
            np.asarray([row[field] for row in aggregate]) - baseline,
        )
        values = np.maximum.accumulate(values)
        for row, value in zip(aggregate, values):
            row[field] = float(value)
    strain = np.asarray(
        [row["engineering_strain"] for row in aggregate]
    )
    stress = np.asarray(
        [row["median_engineering_stress_MPa"] for row in aggregate]
    )
    offset_line = elastic_modulus_MPa * (strain - offset_strain)
    difference = stress - offset_line
    crossings = np.flatnonzero(
        (strain >= offset_strain) & (difference <= 0.0)
    )
    if crossings.size == 0:
        raise ValueError(
            "aggregate pre-peak curve does not cross the offset-yield line"
        )
    upper = int(crossings[0])
    lower = upper - 1
    fraction = difference[lower] / (
        difference[lower] - difference[upper]
    )
    yield_strain = float(
        strain[lower] + fraction * (strain[upper] - strain[lower])
    )
    yield_stress = float(
        stress[lower] + fraction * (stress[upper] - stress[lower])
    )
    post_yield = strain >= yield_strain
    engineering_strain = np.concatenate(
        [[yield_strain], strain[post_yield]]
    )
    engineering_stress = np.concatenate(
        [[yield_stress], stress[post_yield]]
    )
    true_strain = np.log1p(engineering_strain)
    true_stress = engineering_stress * (1.0 + engineering_strain)
    true_stress = np.maximum.accumulate(true_stress)
    equivalent_plastic_strain = (
        true_strain - true_stress / elastic_modulus_MPa
    )
    equivalent_plastic_strain -= equivalent_plastic_strain[0]
    if equivalent_plastic_strain[-1] <= 0.0:
        raise ValueError("derived plastic-strain range is not positive")
    target_plastic_strain = np.linspace(
        0.0, equivalent_plastic_strain[-1], plastic_points
    )
    target_stress = np.interp(
        target_plastic_strain,
        equivalent_plastic_strain,
        true_stress,
    )
    plastic = [
        {
            "true_stress_MPa": float(current_stress),
            "equivalent_plastic_strain": float(current_strain),
        }
        for current_strain, current_stress in zip(
            target_plastic_strain, target_stress
        )
    ]
    metadata = {
        "specimens": len(curves),
        "minimum_curve_coverage": required,
        "minimum_coverage_fraction": minimum_coverage_fraction,
        "offset_strain": offset_strain,
        "yield_engineering_strain": yield_strain,
        "yield_engineering_stress_MPa": yield_stress,
        "yield_true_stress_MPa": float(target_stress[0]),
        "maximum_supported_engineering_strain": float(strain[-1]),
        "maximum_supported_true_stress_MPa": float(target_stress[-1]),
        "maximum_supported_equivalent_plastic_strain": float(
            target_plastic_strain[-1]
        ),
    }
    return aggregate, plastic, metadata


def calibrate_prepeak_plasticity(
    dataset_dir: Path = DEFAULT_DATASET,
    elastic_profile_path: Path = DEFAULT_PROFILE,
    output_dir: Path = DEFAULT_OUTPUT,
    profile_path: Path = DEFAULT_PLASTIC_PROFILE,
) -> Dict[str, object]:
    elastic_profile = json.loads(
        Path(elastic_profile_path).read_text(encoding="utf-8")
    )
    modulus = float(
        elastic_profile["screening_isotropic"]["E_MPa"]
    )
    aggregate, plastic, metadata = aggregate_in_plane_prepeak(
        dataset_dir, modulus
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "aggregate_prepeak_curve.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(aggregate[0])
        )
        writer.writeheader()
        writer.writerows(aggregate)
    with (output_dir / "calculix_plastic_curve.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(plastic[0])
        )
        writer.writeheader()
        writer.writerows(plastic)
    profile = dict(elastic_profile)
    profile["profile_id"] = "bambu_pla_raw_curve_prepeak_plastic_v1"
    profile["intended_use"] = (
        "Exploratory flat-printed in-layer pre-peak tensile sensitivity; "
        "not a fracture, compression, or build-Z model"
    )
    profile["plastic_isotropic"] = {
        "model": "J2_von_Mises_piecewise_isotropic_hardening",
        "source_population": (
            "19 non-Z-upright literal raw tensile curves"
        ),
        "derivation": (
            "zero-shifted pre-peak engineering curves; conditional median "
            "with >=80% specimen coverage; 0.2% offset yield; true stress "
            "and logarithmic strain conversion"
        ),
        **metadata,
        "points": plastic,
    }
    profile["limitations"] = list(profile.get("limitations", [])) + [
        "The J2 flow assumption is not identified by uniaxial tensile data.",
        "Plastic incompressibility and equal tension/compression yield are unverified for printed PLA.",
        "Z-upright curves are excluded from the in-layer hardening aggregate.",
        "The card stops at the last strain with at least 80% non-Z pre-peak coverage.",
        "No post-peak softening, necking, damage, fracture, rate, temperature, or cyclic behavior is represented.",
    ]
    profile_path = Path(profile_path)
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(
        json.dumps(profile, indent=2) + "\n", encoding="utf-8"
    )
    plastic_lines = "\n".join(
        f"{point['true_stress_MPa']:.12g},"
        f"{point['equivalent_plastic_strain']:.12g}"
        for point in plastic
    )
    (output_dir / "prepeak_plastic_material.inp").write_text(
        (
            "*MATERIAL,NAME=BAMBU_PLA_PREPEAK_PLASTIC\n"
            "*DENSITY\n"
            f"{profile['density_tonnes_mm3']:.12g}\n"
            "*ELASTIC\n"
            f"{modulus:.12g},"
            f"{profile['screening_isotropic']['nu']:.12g}\n"
            "*PLASTIC\n"
            f"{plastic_lines}\n"
        ),
        encoding="utf-8",
    )
    lines = [
        "# Bambu PLA Basic pre-peak tensile plasticity",
        "",
        (
            "This exploratory card aggregates the 19 non-Z-upright raw "
            "tensile curves, matching the flat-printed infill use case."
        ),
        "",
        f"- Elastic modulus: {modulus:.1f} MPa",
        (
            f"- 0.2% offset yield: {metadata['yield_engineering_stress_MPa']:.2f} "
            f"MPa at {metadata['yield_engineering_strain']:.3%} engineering strain"
        ),
        (
            f"- Supported pre-peak range: through "
            f"{metadata['maximum_supported_engineering_strain']:.3%} "
            "engineering strain"
        ),
        (
            f"- Final hardening point: "
            f"{metadata['maximum_supported_true_stress_MPa']:.2f} MPa at "
            f"{metadata['maximum_supported_equivalent_plastic_strain']:.5f} "
            "equivalent plastic strain"
        ),
        "",
        "## CalculiX hardening points",
        "",
        "| True stress (MPa) | Equivalent plastic strain |",
        "|---:|---:|",
    ]
    for point in plastic:
        lines.append(
            f"| {point['true_stress_MPa']:.3f} | "
            f"{point['equivalent_plastic_strain']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Validity boundary",
            "",
            "- Use only for monotonic pre-peak tensile sensitivity of flat-printed material.",
            "- Do not interpret the last hardening point as fracture.",
            "- Do not use this card to claim build-Z, compression, impact, fatigue, creep, or temperature predictions.",
            "",
        ]
    )
    (output_dir / "report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return profile


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Derive an exploratory pre-peak plastic card from Bambu curves"
    )
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument(
        "--elastic-profile", default=str(DEFAULT_PROFILE)
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--profile", default=str(DEFAULT_PLASTIC_PROFILE)
    )
    args = parser.parse_args()
    profile = calibrate_prepeak_plasticity(
        Path(args.dataset),
        Path(args.elastic_profile),
        Path(args.output),
        Path(args.profile),
    )
    print(
        "calibrated pre-peak hardening card with "
        f"{len(profile['plastic_isotropic']['points'])} points; "
        f"profile: {Path(args.profile).resolve()}"
    )


if __name__ == "__main__":
    main()
