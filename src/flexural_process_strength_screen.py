from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np

from .bambu_flexural_process_reference import (
    DEFAULT_OUTPUT as FLEXURAL_REFERENCE,
    ORIENTATIONS,
    TEMPERATURES,
)
from .bambu_tds_profile import DEFAULT_TRANSVERSE_PROFILE
from .parsing import parse_frd
from .strength_screening import stress_rows_to_local


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT_ROOT / "results/nozzle_temperature_sensitivity"
DEFAULT_OUTPUT = PROJECT_ROOT / "results/flexural_process_strength_screen"
PATTERNS = ("triangular", "sd_c2_o50")


@dataclass(frozen=True)
class FlexuralProcessStrengthScreen:
    maximum_mean_utilization: float
    p99_mean_utilization: float
    maximum_lower_cv_utilization: float
    maximum_in_plane_principal_tension_MPa: float
    maximum_positive_sigma3_MPa: float
    governing_mode: str
    governing_in_plane_angle_deg: float | None
    sampled_stress_points: int

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _fold_angle_deg(vector: np.ndarray) -> float:
    angle = math.degrees(math.atan2(float(vector[1]), float(vector[0])))
    angle %= 180.0
    return min(angle, 180.0 - angle)


def screen_flexural_process_strength(
    stress_rows: np.ndarray,
    in_plane_strengths_MPa: Sequence[float],
    in_plane_lower_cv_strengths_MPa: Sequence[float],
    build_direction_strength_MPa: float,
    build_direction_lower_strength_MPa: float,
) -> FlexuralProcessStrengthScreen | None:
    strengths = np.asarray(in_plane_strengths_MPa, dtype=float)
    lower = np.asarray(in_plane_lower_cv_strengths_MPa, dtype=float)
    if strengths.shape != (3,) or lower.shape != (3,):
        raise ValueError("in-plane strengths must contain 0/45/90 values")
    if (
        not np.isfinite(strengths).all()
        or not np.isfinite(lower).all()
        or np.any(strengths <= 0.0)
        or np.any(lower <= 0.0)
    ):
        raise ValueError("flexural strengths must be finite and positive")
    if build_direction_strength_MPa <= 0.0:
        raise ValueError("build-direction strength must be positive")
    if build_direction_lower_strength_MPa <= 0.0:
        raise ValueError("lower build-direction strength must be positive")
    local = stress_rows_to_local(stress_rows)
    if len(local) == 0:
        return None

    tensors = np.empty((len(local), 2, 2), dtype=float)
    tensors[:, 0, 0] = local[:, 0]
    tensors[:, 1, 1] = local[:, 1]
    tensors[:, 0, 1] = tensors[:, 1, 0] = local[:, 3]
    eigenvalues, eigenvectors = np.linalg.eigh(tensors)
    principal = np.maximum(eigenvalues[:, 1], 0.0)
    angles = np.asarray(
        [_fold_angle_deg(vector) for vector in eigenvectors[:, :, 1]],
        dtype=float,
    )
    in_plane_reference = np.interp(
        angles, np.asarray(ORIENTATIONS, dtype=float), strengths
    )
    in_plane_lower = np.interp(
        angles, np.asarray(ORIENTATIONS, dtype=float), lower
    )
    sigma3 = np.maximum(local[:, 2], 0.0)
    mean_components = np.column_stack(
        (
            principal / in_plane_reference,
            sigma3 / build_direction_strength_MPa,
        )
    )
    lower_components = np.column_stack(
        (
            principal / in_plane_lower,
            sigma3 / build_direction_lower_strength_MPa,
        )
    )
    mean_utilization = np.max(mean_components, axis=1)
    lower_utilization = np.max(lower_components, axis=1)
    peak_index = int(np.argmax(mean_utilization))
    mode_index = int(np.argmax(mean_components[peak_index]))
    return FlexuralProcessStrengthScreen(
        maximum_mean_utilization=float(mean_utilization[peak_index]),
        p99_mean_utilization=float(np.percentile(mean_utilization, 99.0)),
        maximum_lower_cv_utilization=float(np.max(lower_utilization)),
        maximum_in_plane_principal_tension_MPa=float(np.max(principal)),
        maximum_positive_sigma3_MPa=float(np.max(sigma3)),
        governing_mode=(
            "in_plane_principal_tension"
            if mode_index == 0
            else "build_direction_normal_proxy"
        ),
        governing_in_plane_angle_deg=(
            float(angles[peak_index]) if mode_index == 0 else None
        ),
        sampled_stress_points=len(local),
    )


def _strength_references(
    rows: Sequence[Mapping[str, object]],
    temperature: int,
    transverse_profile: Mapping[str, object],
) -> Dict[str, object]:
    current = sorted(
        (
            row
            for row in rows
            if int(float(row["nozzle_temperature_C"])) == temperature
        ),
        key=lambda row: float(row["raster_orientation_deg"]),
    )
    if len(current) != 3:
        raise ValueError(
            f"expected three flexural rows at {temperature} C"
        )
    strengths = np.asarray(
        [float(row["flexural_strength_MPa"]) for row in current]
    )
    cvs = np.asarray(
        [float(row["flexural_strength_CV_percent"]) for row in current]
    )
    lower = strengths * (1.0 - cvs / 100.0)
    row_90 = current[2]
    baseline_90 = next(
        row
        for row in rows
        if int(float(row["nozzle_temperature_C"])) == 220
        and int(float(row["raster_orientation_deg"])) == 90
    )
    scale = float(row_90["flexural_strength_MPa"]) / float(
        baseline_90["flexural_strength_MPa"]
    )
    tds = transverse_profile["reported_flexural_reference"]
    build_mean = float(tds["Z_strength_MPa"]) * scale
    build_lower = (
        float(tds["Z_strength_MPa"])
        - float(tds["Z_strength_standard_deviation_MPa"])
    ) * scale
    return {
        "strengths": strengths,
        "lower": lower,
        "build_mean": build_mean,
        "build_lower": build_lower,
        "build_scale": scale,
    }


def audit_bending_fields(
    source_root: Path = DEFAULT_SOURCE,
    reference_root: Path = FLEXURAL_REFERENCE,
    transverse_profile_path: Path = DEFAULT_TRANSVERSE_PROFILE,
) -> List[Dict[str, object]]:
    reference_rows = json.loads(
        (Path(reference_root) / "flexural_process_matrix.json").read_text(
            encoding="utf-8"
        )
    )
    transverse = json.loads(
        Path(transverse_profile_path).read_text(encoding="utf-8")
    )
    audited: List[Dict[str, object]] = []
    for temperature in TEMPERATURES:
        summary_path = (
            Path(source_root)
            / f"print_{temperature}C"
            / "solid/summary.json"
        )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        references = _strength_references(
            reference_rows, temperature, transverse
        )
        for source in summary:
            if source["pattern"] not in PATTERNS:
                continue
            if source["load_case"] != "bending":
                continue
            row: Dict[str, object] = {
                "nozzle_temperature_C": temperature,
                "pattern": source["pattern"],
                "load_case": source["load_case"],
                "source_status": source["status"],
                "run_directory": source["run_directory"],
                "applied_displacement_mm": source[
                    "applied_displacement_mm"
                ],
                "flexural_strength_0deg_MPa": float(
                    references["strengths"][0]
                ),
                "flexural_strength_45deg_MPa": float(
                    references["strengths"][1]
                ),
                "flexural_strength_90deg_MPa": float(
                    references["strengths"][2]
                ),
                "build_direction_strength_proxy_MPa": references[
                    "build_mean"
                ],
                "build_direction_proxy_scale_from_220C": references[
                    "build_scale"
                ],
            }
            if source["status"] != "ok":
                row["screen_status"] = "source_solver_failed"
                audited.append(row)
                continue
            curve = parse_frd(
                Path(str(source["run_directory"])) / "model.frd"
            )
            stress = curve.get("frd_stress")
            if stress is None:
                row["screen_status"] = "stress_field_missing"
                audited.append(row)
                continue
            screen = screen_flexural_process_strength(
                stress,
                references["strengths"],
                references["lower"],
                float(references["build_mean"]),
                float(references["build_lower"]),
            )
            if screen is None:
                row["screen_status"] = "stress_field_missing"
            else:
                row["screen_status"] = "ok"
                row.update(screen.to_dict())
                row["peak_crosses_mean_flexural_reference"] = (
                    screen.maximum_mean_utilization > 1.0
                )
                row["peak_crosses_lower_cv_flexural_reference"] = (
                    screen.maximum_lower_cv_utilization > 1.0
                )
            audited.append(row)
    return audited


def write_flexural_process_screen(
    output_dir: Path = DEFAULT_OUTPUT,
    source_root: Path = DEFAULT_SOURCE,
    reference_root: Path = FLEXURAL_REFERENCE,
    transverse_profile_path: Path = DEFAULT_TRANSVERSE_PROFILE,
) -> List[Dict[str, object]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = audit_bending_fields(
        source_root, reference_root, transverse_profile_path
    )
    fields = sorted({key for row in rows for key in row})
    with (output_dir / "screen.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "screen.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )
    valid = [row for row in rows if row["screen_status"] == "ok"]
    priorities = []
    for row in sorted(
        valid,
        key=lambda item: float(item["p99_mean_utilization"]),
        reverse=True,
    ):
        p99 = float(row["p99_mean_utilization"])
        scale = min(1.0, 0.6 / p99) if p99 > 0.0 else 1.0
        priorities.append(
            {
                "priority": len(priorities) + 1,
                "nozzle_temperature_C": row["nozzle_temperature_C"],
                "pattern": row["pattern"],
                "replicates": 3,
                "endpoint_displacement_mm": row[
                    "applied_displacement_mm"
                ],
                "recommended_first_displacement_mm": (
                    float(row["applied_displacement_mm"]) * scale
                ),
                "p99_mean_utilization": p99,
                "peak_mean_utilization": row["maximum_mean_utilization"],
                "governing_mode": row["governing_mode"],
                "governing_in_plane_angle_deg": row[
                    "governing_in_plane_angle_deg"
                ],
            }
        )
    with (output_dir / "physical_bending_priority.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(priorities[0])
        )
        writer.writeheader()
        writer.writerows(priorities)

    peak_crossings = sum(
        bool(row["peak_crosses_mean_flexural_reference"]) for row in valid
    )
    lower_crossings = sum(
        bool(row["peak_crosses_lower_cv_flexural_reference"])
        for row in valid
    )
    lines = [
        "# Bambu PLA Basic process-adjusted flexural strength screen",
        "",
        "## Result",
        "",
        (
            f"Screened {len(valid)}/8 saved bending stress fields against "
            "the exact nozzle-temperature/orientation strength matrix. "
            f"{peak_crossings} endpoints cross a mean flexural reference and "
            f"{lower_crossings} cross the one-CV-lower reference."
        ),
        "",
        "| Nozzle | Pattern | Peak | p99 | Lower-CV peak | "
        "Governing mode | Angle | First test displacement |",
        "|---:|---|---:|---:|---:|---|---:|---:|",
    ]
    priority_lookup = {
        (row["nozzle_temperature_C"], row["pattern"]): row
        for row in priorities
    }
    for row in valid:
        priority = priority_lookup[
            (row["nozzle_temperature_C"], row["pattern"])
        ]
        angle = row["governing_in_plane_angle_deg"]
        lines.append(
            f"| {row['nozzle_temperature_C']} °C | {row['pattern']} | "
            f"{float(row['maximum_mean_utilization']):.3f} | "
            f"{float(row['p99_mean_utilization']):.3f} | "
            f"{float(row['maximum_lower_cv_utilization']):.3f} | "
            f"{row['governing_mode']} | "
            f"{'n/a' if angle is None else f'{float(angle):.1f}°'} | "
            f"{float(priority['recommended_first_displacement_mm']):.3f} mm |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "- The in-plane maximum principal tensile stress is compared with "
            "a piecewise-linear 0°/45°/90° flexural-strength reference.",
            "- The local-3 value is an explicit proxy: the TDS Z flexural "
            "strength is scaled by the paper's 90° process ratio. It is not a "
            "measured build-Z process matrix.",
            "- Coefficients of variation describe the published specimen set; "
            "mean×(1−CV) is a sensitivity value, not a statistical lower "
            "confidence bound.",
            "- This checks bending initiation references only. It does not "
            "predict crack growth, delamination, final load, fatigue, impact, "
            "or another load mode.",
            "",
        ]
    )
    (output_dir / "report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Screen infill bending fields against flexural process data"
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--reference", default=str(FLEXURAL_REFERENCE))
    parser.add_argument(
        "--transverse-profile", default=str(DEFAULT_TRANSVERSE_PROFILE)
    )
    args = parser.parse_args()
    rows = write_flexural_process_screen(
        Path(args.output),
        Path(args.source),
        Path(args.reference),
        Path(args.transverse_profile),
    )
    valid = sum(row["screen_status"] == "ok" for row in rows)
    print(
        f"flexural process strength screen: {valid}/8 fields; "
        f"report: {(Path(args.output) / 'report.md').resolve()}"
    )


if __name__ == "__main__":
    main()
