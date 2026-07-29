from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd

from .infill_benchmark import DEFAULT_PROPERTIES


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = (
    PROJECT_ROOT / "data/public/mendeley-8jxw5533py-v2"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "results/bambu_curve_calibration"
DEFAULT_PROFILE = (
    PROJECT_ROOT / "data/reported/bambu_pla_raw_curve_elastic.json"
)


@dataclass
class CurveModulus:
    file: str
    test_id: str
    source_specimen_label: str
    source_platform_axis: str
    source_placement: str
    points_total: int
    points_in_window: int
    strain_min: float
    strain_max: float
    modulus_MPa: float
    intercept_MPa: float
    r_squared: float
    peak_stress_MPa: float
    summary_modulus_MPa: float | None
    relative_error_vs_same_label_summary: float | None
    quality_flags: str


def fit_elastic_slope(
    strain: Iterable[float],
    stress_MPa: Iterable[float],
    strain_min: float = 0.002,
    strain_max: float = 0.006,
) -> tuple[float, float, float, int]:
    """Affine least-squares slope in a fixed, auditable elastic window."""
    frame = pd.DataFrame(
        {
            "strain": np.asarray(list(strain), dtype=float),
            "stress": np.asarray(list(stress_MPa), dtype=float),
        }
    )
    frame = (
        frame.replace([np.inf, -np.inf], np.nan)
        .dropna()
        .drop_duplicates()
        .sort_values("strain", kind="mergesort")
    )
    window = frame[
        (frame["strain"] >= strain_min)
        & (frame["strain"] <= strain_max)
    ]
    if len(window) < 20:
        raise ValueError("elastic window contains fewer than 20 unique points")
    x = window["strain"].to_numpy(float)
    y = window["stress"].to_numpy(float)
    design = np.column_stack([x, np.ones_like(x)])
    slope, intercept = np.linalg.lstsq(design, y, rcond=None)[0]
    prediction = slope * x + intercept
    residual = float(np.sum((y - prediction) ** 2))
    total = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - residual / total if total > 0.0 else 1.0
    if not math.isfinite(slope) or slope <= 0.0:
        raise ValueError("elastic fit did not produce a positive finite slope")
    return float(slope), float(intercept), float(r_squared), len(window)


def huber_location(
    values: Iterable[float],
    tuning_constant: float = 1.345,
    maximum_iterations: int = 100,
) -> float:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("Huber location requires finite observations")
    location = float(np.median(array))
    for _ in range(maximum_iterations):
        scale = 1.4826 * float(np.median(np.abs(array - location)))
        if scale <= np.finfo(float).eps:
            return location
        cutoff = tuning_constant * scale
        residual = np.abs(array - location)
        weights = np.ones_like(array)
        outside = residual > cutoff
        weights[outside] = cutoff / residual[outside]
        update = float(np.sum(weights * array) / np.sum(weights))
        if abs(update - location) <= 1.0e-10:
            return update
        location = update
    return location


def _bootstrap_huber_ci(
    values: Sequence[float],
    seed: int = 20260728,
    samples: int = 10000,
) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    locations = np.empty(samples, dtype=float)
    for index in range(samples):
        draw = rng.choice(array, size=len(array), replace=True)
        locations[index] = huber_location(draw)
    lower, upper = np.percentile(locations, [2.5, 97.5])
    return float(lower), float(upper)


def _clean_summary_label(value: object) -> str:
    label = str(value).strip()
    label = re.sub(r"^45-", "45Deg-", label)
    return label


def load_summary_moduli(path: Path) -> Dict[str, float]:
    frame = pd.read_excel(path, sheet_name=0, header=None)
    result: Dict[str, float] = {}
    for start in range(0, len(frame), 3):
        modulus_row = start + 2
        if modulus_row >= len(frame):
            continue
        for column in (1, 2, 3):
            label = frame.iloc[start, column]
            value = frame.iloc[modulus_row, column]
            if pd.isna(label) or pd.isna(value):
                continue
            try:
                number = float(str(value).replace("`", "").strip())
            except ValueError:
                continue
            result[_clean_summary_label(label)] = number
    return result


def _model_row(
    name: str,
    records: Sequence[CurveModulus],
    key: Callable[[CurveModulus], str],
) -> Dict[str, object]:
    groups: Dict[str, List[float]] = {}
    for record in records:
        groups.setdefault(key(record), []).append(record.modulus_MPa)
    predictions = np.asarray(
        [np.mean(groups[key(record)]) for record in records], dtype=float
    )
    observed = np.asarray(
        [record.modulus_MPa for record in records], dtype=float
    )
    residuals = observed - predictions
    sse = float(np.sum(residuals**2))
    count = len(records)
    parameters = len(groups)
    rmse = float(np.sqrt(sse / count))
    aic = float(count * np.log(sse / count) + 2 * parameters)
    bic = float(
        count * np.log(sse / count)
        + parameters * np.log(count)
    )
    leave_one_out = []
    for index, record in enumerate(records):
        same_group = [
            other.modulus_MPa
            for other_index, other in enumerate(records)
            if other_index != index and key(other) == key(record)
        ]
        if not same_group:
            same_group = [
                other.modulus_MPa
                for other_index, other in enumerate(records)
                if other_index != index
            ]
        leave_one_out.append(float(np.mean(same_group)))
    loocv_rmse = float(
        np.sqrt(np.mean((observed - np.asarray(leave_one_out)) ** 2))
    )
    return {
        "model": name,
        "parameters": parameters,
        "rmse_MPa": rmse,
        "loocv_rmse_MPa": loocv_rmse,
        "aic": aic,
        "bic": bic,
        "group_means_MPa": json.dumps(
            {
                group: float(np.mean(values))
                for group, values in sorted(groups.items())
            },
            sort_keys=True,
        ),
    }


def calibrate_bambu_curves(
    dataset_dir: Path = DEFAULT_DATASET,
    output_dir: Path = DEFAULT_OUTPUT,
    profile_path: Path = DEFAULT_PROFILE,
    strain_min: float = 0.002,
    strain_max: float = 0.006,
) -> Dict[str, object]:
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_profile = json.loads(
        Path(DEFAULT_PROPERTIES).read_text(encoding="utf-8")
    )
    summary_moduli = load_summary_moduli(
        dataset_dir / "source/3-1_Material_property.xlsx"
    )
    manifest = pd.read_csv(dataset_dir / "manifest.csv")
    manifest_by_test = manifest.set_index("test_id").to_dict("index")
    records: List[CurveModulus] = []
    for path in sorted((dataset_dir / "curves").glob("*.csv")):
        frame = pd.read_csv(path)
        first = frame.iloc[0]
        slope, intercept, r_squared, points = fit_elastic_slope(
            frame["engineering_strain"],
            frame["engineering_stress_MPa"],
            strain_min,
            strain_max,
        )
        test_id = str(first["test_id"])
        label = str(first["source_specimen_label"])
        summary = summary_moduli.get(label)
        relative_error = (
            None
            if summary is None
            else abs(slope - summary) / summary
        )
        flags = []
        source_flags = str(
            manifest_by_test.get(test_id, {}).get(
                "data_quality_flags", ""
            )
        )
        if source_flags and source_flags != "nan":
            flags.extend(source_flags.split(";"))
        if r_squared < 0.995:
            flags.append("elastic_window_r_squared_below_0.995")
        if summary is None:
            flags.append("no_same_label_summary_modulus")
        elif relative_error is not None and relative_error > 0.05:
            flags.append("raw_slope_differs_from_summary_modulus_gt_5pct")
        records.append(
            CurveModulus(
                str(path.relative_to(dataset_dir)),
                test_id,
                label,
                str(first["source_platform_axis"]),
                str(first["source_placement"]),
                len(frame),
                points,
                strain_min,
                strain_max,
                slope,
                intercept,
                r_squared,
                float(frame["engineering_stress_MPa"].max()),
                summary,
                relative_error,
                ";".join(dict.fromkeys(flags)),
            )
        )
    moduli = [record.modulus_MPa for record in records]
    robust_modulus = huber_location(moduli)
    ci_lower, ci_upper = _bootstrap_huber_ci(moduli)
    specimen_sd = float(np.std(moduli, ddof=1))
    model_rows = [
        _model_row("isotropic", records, lambda _: "all"),
        _model_row(
            "source_placement",
            records,
            lambda record: record.source_placement,
        ),
        _model_row(
            "source_axis_and_placement",
            records,
            lambda record: (
                f"{record.source_platform_axis}-"
                f"{record.source_placement}"
            ),
        ),
    ]
    model_rows.sort(key=lambda row: float(row["bic"]))
    with (output_dir / "per_specimen_modulus.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(asdict(records[0]))
        )
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)
    with (output_dir / "model_comparison.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(model_rows[0])
        )
        writer.writeheader()
        writer.writerows(model_rows)
    groups: Dict[str, List[float]] = {}
    for record in records:
        group = (
            f"{record.source_platform_axis}-"
            f"{record.source_placement}"
        )
        groups.setdefault(group, []).append(record.modulus_MPa)
    group_rows = [
        {
            "source_group": group,
            "specimens": len(values),
            "mean_modulus_MPa": float(np.mean(values)),
            "standard_deviation_MPa": (
                None
                if len(values) < 2
                else float(np.std(values, ddof=1))
            ),
            "minimum_MPa": float(np.min(values)),
            "maximum_MPa": float(np.max(values)),
        }
        for group, values in sorted(groups.items())
    ]
    with (output_dir / "source_group_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(group_rows[0])
        )
        writer.writeheader()
        writer.writerows(group_rows)
    z_values = groups.get("Z-upright", [])
    shared_summary = [
        record
        for record in records
        if record.summary_modulus_MPa is not None
    ]
    mismatched = [
        record
        for record in shared_summary
        if record.relative_error_vs_same_label_summary is not None
        and record.relative_error_vs_same_label_summary > 0.05
    ]
    profile = {
        "profile_id": "bambu_pla_raw_curve_elastic_v1",
        "intended_use": (
            "Exploratory linear-elastic screening from literal Bambu PLA "
            "Basic raw tensile curves; not a failure model"
        ),
        "material": base_profile["material"],
        "density_tonnes_mm3": base_profile["density_tonnes_mm3"],
        "screening_isotropic": {
            "E_MPa": robust_modulus,
            "E_standard_deviation_MPa": specimen_sd,
            "E_bootstrap_95pct_lower_MPa": ci_lower,
            "E_bootstrap_95pct_upper_MPa": ci_upper,
            "nu": base_profile["screening_isotropic"]["nu"],
            "nu_provenance": base_profile["screening_isotropic"][
                "nu_provenance"
            ],
            "selection_reason": (
                "Isotropic has the lowest BIC; source placement groups "
                "do not map completely to local material axes"
            ),
        },
        "raw_curve_extraction": {
            "specimens": len(records),
            "strain_window": [strain_min, strain_max],
            "fit": "affine ordinary least squares after exact-row deduplication",
            "robust_location": "Huber M-estimate, tuning constant 1.345",
            "bootstrap_seed": 20260728,
            "mean_modulus_MPa": float(np.mean(moduli)),
            "median_modulus_MPa": float(np.median(moduli)),
            "minimum_modulus_MPa": float(np.min(moduli)),
            "maximum_modulus_MPa": float(np.max(moduli)),
            "same_label_summary_comparisons": len(shared_summary),
            "same_label_modulus_mismatches_gt_5pct": len(mismatched),
        },
        "directional_observations": {
            "Z_upright_candidate_E3_mean_MPa": (
                None if not z_values else float(np.mean(z_values))
            ),
            "Z_upright_candidate_E3_standard_deviation_MPa": (
                None
                if len(z_values) < 2
                else float(np.std(z_values, ddof=1))
            ),
            "interpretation": (
                "Z-upright loading is consistent with the build direction; "
                "the other placement groups cannot be uniquely assigned to "
                "local bead and in-layer transverse axes from this source."
            ),
        },
        "reported_strength_reference": base_profile[
            "reported_strength_reference"
        ],
        "tds_orthotropic_initialization_prior": base_profile[
            "tds_orthotropic_initialization_prior"
        ],
        "model_comparison": model_rows,
        "sources": base_profile["sources"],
        "limitations": [
            "The raw workbook has 22 curves while the article claims 21.",
            "Source placement labels and summary-workbook specimen labels conflict.",
            "Only Z-upright has a defensible direct build-axis interpretation.",
            "Poisson ratios and shear moduli remain project priors.",
            "No compression, flexural, fatigue, creep, fracture, or held-out component measurements are available for this exact material source.",
        ],
    }
    profile_path = Path(profile_path)
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(
        json.dumps(profile, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "raw_curve_isotropic_material.inp").write_text(
        (
            "*MATERIAL,NAME=BAMBU_PLA_RAW_CURVE_ISOTROPIC\n"
            "*DENSITY\n"
            f"{profile['density_tonnes_mm3']:.12g}\n"
            "*ELASTIC\n"
            f"{robust_modulus:.12g},"
            f"{profile['screening_isotropic']['nu']:.12g}\n"
        ),
        encoding="utf-8",
    )
    reported = float(
        base_profile["screening_isotropic"]["E_MPa"]
    )
    summary_overall = 2113.147344
    lines = [
        "# Bambu PLA Basic raw-curve elastic calibration",
        "",
        (
            f"All {len(records)} literal tensile curves were fitted over "
            f"{strain_min:.3%}–{strain_max:.3%} engineering strain after "
            "exact-row deduplication."
        ),
        "",
        "## Recommended screening value",
        "",
        (
            f"Robust isotropic modulus: **{robust_modulus:.1f} MPa** "
            f"(specimen SD {specimen_sd:.1f} MPa; bootstrap 95% interval for "
            f"the robust location {ci_lower:.1f}–{ci_upper:.1f} MPa)."
        ),
        (
            f"This is {(robust_modulus / reported - 1.0):+.1%} relative to "
            f"the article-level {reported:.1f} MPa screening value and "
            f"{(robust_modulus / summary_overall - 1.0):+.1%} relative to "
            f"the summary workbook's {summary_overall:.1f} MPa overall mean."
        ),
        "",
        "## Model comparison",
        "",
        "| Model | Parameters | RMSE (MPa) | LOOCV RMSE (MPa) | AIC | BIC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in model_rows:
        lines.append(
            f"| {row['model']} | {row['parameters']} | "
            f"{row['rmse_MPa']:.1f} | {row['loocv_rmse_MPa']:.1f} | "
            f"{row['aic']:.1f} | {row['bic']:.1f} |"
        )
    lines.extend(
        [
            "",
            (
                "BIC selects the one-parameter isotropic description. "
                "Placement grouping reduces in-sample error modestly but is "
                "not strong enough to justify treating source platform labels "
                "as constitutive material axes."
            ),
            "",
            "## Source reconciliation",
            "",
            (
                f"{len(shared_summary)} raw curves have the same nominal label "
                f"in the summary workbook; {len(mismatched)} differ by more "
                "than 5% in modulus under the fixed raw-curve fit."
            ),
            (
                f"The three Z-upright curves give a candidate build-direction "
                f"modulus of {np.mean(z_values):.1f} ± "
                f"{np.std(z_values, ddof=1):.1f} MPa, but E1, E2, and the "
                "shear constants remain unidentifiable."
            ),
            "",
            "## Decision",
            "",
            "- Use the raw-curve robust isotropic profile for sensitivity runs.",
            "- Keep the article-level profile as the conservative provenance baseline.",
            "- Do not promote the placement groups to orthotropic local axes.",
            "- Acquire matched local-1, local-2, local-3 tensile and shear/flexural coupons before fitting a full engineering-constants card.",
            "",
        ]
    )
    (output_dir / "report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return profile


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract a defensible elastic profile from Bambu PLA raw curves"
    )
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--profile", default=str(DEFAULT_PROFILE))
    parser.add_argument("--strain-min", type=float, default=0.002)
    parser.add_argument("--strain-max", type=float, default=0.006)
    args = parser.parse_args()
    profile = calibrate_bambu_curves(
        Path(args.dataset),
        Path(args.output),
        Path(args.profile),
        args.strain_min,
        args.strain_max,
    )
    print(
        "calibrated "
        f"{profile['raw_curve_extraction']['specimens']} raw curves; "
        f"profile: {Path(args.profile).resolve()}"
    )


if __name__ == "__main__":
    main()
