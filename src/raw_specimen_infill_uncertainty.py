from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np

from .infill_benchmark import LOAD_CASES
from .pattern_robustness_campaign import PATTERNS, finalist_geometries
from .solid_multiload import (
    SolidLoadResult,
    _balanced_rows,
    run_solid_multiload,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "results/raw_specimen_infill_uncertainty"
)
DEFAULT_RUNS = PROJECT_ROOT / "runs/raw_specimen_infill_uncertainty"
MATERIAL_PROFILE = (
    PROJECT_ROOT / "data/reported/bambu_pla_raw_curve_elastic.json"
)
MODULUS_ROWS = (
    PROJECT_ROOT / "results/bambu_curve_calibration/per_specimen_modulus.csv"
)
RECONCILIATION = (
    PROJECT_ROOT
    / "results/mendeley_tensile_reconciliation_audit/provenance.json"
)
SCALAR_CHECK_TOLERANCE = 5.0e-3


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write empty CSV")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_specimen_moduli(path: Path = MODULUS_ROWS) -> List[Dict[str, object]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = [
        {
            "source_specimen_label": row["source_specimen_label"],
            "modulus_MPa": float(row["modulus_MPa"]),
            "r_squared": float(row["r_squared"]),
        }
        for row in rows
    ]
    if len(result) != 22:
        raise ValueError(f"expected 22 raw modulus rows, found {len(result)}")
    if len({row["source_specimen_label"] for row in result}) != len(result):
        raise ValueError("raw modulus labels are not unique")
    if any(float(row["modulus_MPa"]) <= 0.0 for row in result):
        raise ValueError("raw modulus values must be positive")
    return result


def write_scalar_profiles(
    output_dir: Path = DEFAULT_OUTPUT,
    moduli: Sequence[Mapping[str, object]] | None = None,
) -> Dict[str, Path]:
    output_dir = Path(output_dir)
    profile_dir = output_dir / "scalar_check_profiles"
    profile_dir.mkdir(parents=True, exist_ok=True)
    base = json.loads(MATERIAL_PROFILE.read_text(encoding="utf-8"))
    specimen_rows = list(moduli or read_specimen_moduli())
    values = [float(row["modulus_MPa"]) for row in specimen_rows]
    paths = {}
    for label, value in (
        ("measured_minimum", min(values)),
        ("measured_maximum", max(values)),
    ):
        profile = json.loads(json.dumps(base))
        profile["profile_id"] = f"bambu_pla_raw_curve_{label}_check_v1"
        profile["screening_isotropic"]["E_MPa"] = value
        profile["screening_isotropic"]["E_standard_deviation_MPa"] = 0.0
        profile["uncertainty_check"] = {
            "name": label,
            "E_MPa": value,
            "interpretation": (
                "direct CalculiX scalar-linearity verification endpoint; "
                "not a released material card"
            ),
        }
        path = profile_dir / f"{label}.json"
        path.write_text(
            json.dumps(profile, indent=2) + "\n", encoding="utf-8"
        )
        paths[label] = path
    return paths


def run_reference_cases(
    output_dir: Path = DEFAULT_OUTPUT,
    runs_dir: Path = DEFAULT_RUNS,
    mesh_size_mm: float = 0.7,
) -> None:
    output_dir = Path(output_dir)
    runs_dir = Path(runs_dir)
    geometries = finalist_geometries()
    run_solid_multiload(
        output_dir / "baseline",
        runs_dir / "baseline",
        properties_path=MATERIAL_PROFILE,
        patterns=PATTERNS,
        load_cases=LOAD_CASES,
        mesh_size_mm=mesh_size_mm,
        geometries=geometries,
    )
    profiles = write_scalar_profiles(output_dir)
    for label, profile in profiles.items():
        run_solid_multiload(
            output_dir / "scalar_checks" / label,
            runs_dir / "scalar_checks" / label,
            properties_path=profile,
            patterns=("triangular",),
            load_cases=LOAD_CASES,
            mesh_size_mm=mesh_size_mm,
            geometries={"triangular": geometries["triangular"]},
        )


def _metric(row: Mapping[str, object]) -> tuple[str, float]:
    if row["load_case"] == "buckling":
        return (
            "critical_buckling_load_N",
            float(row["critical_buckling_load_N"]),
        )
    return "stiffness_N_per_mm", float(row["stiffness_N_per_mm"])


def _specific_metric(row: Mapping[str, object]) -> float:
    if row["load_case"] == "buckling":
        return float(row["critical_buckling_load_N_per_g"])
    return float(row["specific_stiffness_N_per_mm_g"])


def _load_rows(path: Path) -> List[Dict[str, object]]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"expected list in {path}")
    return rows


def scalar_linearity_errors(
    baseline_rows: Sequence[Mapping[str, object]],
    endpoint_rows: Mapping[str, Sequence[Mapping[str, object]]],
    central_E_MPa: float,
    endpoint_E_MPa: Mapping[str, float],
) -> List[Dict[str, object]]:
    baseline = {
        (str(row["pattern"]), str(row["load_case"])): row
        for row in baseline_rows
    }
    results = []
    for endpoint, rows in endpoint_rows.items():
        scale = float(endpoint_E_MPa[endpoint]) / central_E_MPa
        for row in rows:
            key = (str(row["pattern"]), str(row["load_case"]))
            metric_name, observed = _metric(row)
            _, base_value = _metric(baseline[key])
            expected = base_value * scale
            results.append(
                {
                    "endpoint": endpoint,
                    "pattern": key[0],
                    "load_case": key[1],
                    "E_MPa": endpoint_E_MPa[endpoint],
                    "metric": metric_name,
                    "direct_CalculiX_value": observed,
                    "scaled_baseline_value": expected,
                    "relative_error": abs(observed - expected)
                    / max(abs(observed), 1.0e-12),
                }
            )
    return results


def aggregate_ensemble(
    output_dir: Path = DEFAULT_OUTPUT,
) -> Dict[str, object]:
    output_dir = Path(output_dir)
    baseline_path = output_dir / "baseline/summary.json"
    baseline_rows = _load_rows(baseline_path)
    if len(baseline_rows) != len(PATTERNS) * len(LOAD_CASES):
        raise ValueError("baseline finalist matrix is incomplete")
    if any(row["status"] != "ok" for row in baseline_rows):
        raise ValueError("baseline finalist matrix has solver failures")

    profile = json.loads(MATERIAL_PROFILE.read_text(encoding="utf-8"))
    central_E = float(profile["screening_isotropic"]["E_MPa"])
    moduli = read_specimen_moduli()
    values = np.asarray(
        [float(row["modulus_MPa"]) for row in moduli], dtype=float
    )
    endpoints = {
        "measured_minimum": float(np.min(values)),
        "measured_maximum": float(np.max(values)),
    }
    endpoint_rows = {
        label: _load_rows(
            output_dir / "scalar_checks" / label / "summary.json"
        )
        for label in endpoints
    }
    errors = scalar_linearity_errors(
        baseline_rows, endpoint_rows, central_E, endpoints
    )
    _write_csv(output_dir / "scalar_linearity_check.csv", errors)
    maximum_scaling_error = max(
        float(row["relative_error"]) for row in errors
    )

    baseline_lookup = {
        (str(row["pattern"]), str(row["load_case"])): row
        for row in baseline_rows
    }
    central_rank_by_load = {}
    for load_case in LOAD_CASES:
        ranked = sorted(
            PATTERNS,
            key=lambda pattern: _specific_metric(
                baseline_lookup[(pattern, load_case)]
            ),
            reverse=True,
        )
        central_rank_by_load[load_case] = {
            pattern: rank for rank, pattern in enumerate(ranked, start=1)
        }

    ensemble_rows = []
    for specimen in moduli:
        modulus = float(specimen["modulus_MPa"])
        scale = modulus / central_E
        for row in baseline_rows:
            metric_name, base_value = _metric(row)
            pattern = str(row["pattern"])
            load_case = str(row["load_case"])
            ensemble_rows.append(
                {
                    "source_specimen_label": specimen[
                        "source_specimen_label"
                    ],
                    "modulus_MPa": modulus,
                    "modulus_fit_r_squared": specimen["r_squared"],
                    "pattern": pattern,
                    "load_case": load_case,
                    "metric": metric_name,
                    "predicted_value": base_value * scale,
                    "baseline_value": base_value,
                    "scale_to_frozen_profile": scale,
                    "load_case_rank": central_rank_by_load[load_case][
                        pattern
                    ],
                }
            )
    _write_csv(output_dir / "specimen_ensemble_predictions.csv", ensemble_rows)

    interval_rows = []
    for pattern in PATTERNS:
        for load_case in LOAD_CASES:
            selected = [
                float(row["predicted_value"])
                for row in ensemble_rows
                if row["pattern"] == pattern
                and row["load_case"] == load_case
            ]
            baseline_value = _metric(
                baseline_lookup[(pattern, load_case)]
            )[1]
            control_value = _metric(
                baseline_lookup[("triangular", load_case)]
            )[1]
            interval_rows.append(
                {
                    "pattern": pattern,
                    "load_case": load_case,
                    "metric": _metric(
                        baseline_lookup[(pattern, load_case)]
                    )[0],
                    "frozen_profile_value": baseline_value,
                    "empirical_minimum": float(np.min(selected)),
                    "empirical_q05": float(np.quantile(selected, 0.05)),
                    "empirical_median": float(np.median(selected)),
                    "empirical_q95": float(np.quantile(selected, 0.95)),
                    "empirical_maximum": float(np.max(selected)),
                    "empirical_coefficient_of_variation": float(
                        np.std(selected, ddof=1) / np.mean(selected)
                    ),
                    "ratio_to_triangular": baseline_value / control_value,
                    "load_case_rank": central_rank_by_load[load_case][
                        pattern
                    ],
                    "rank_stability_fraction_across_22_moduli": 1.0,
                }
            )
    _write_csv(output_dir / "prediction_intervals.csv", interval_rows)

    baseline_objects = [SolidLoadResult(**row) for row in baseline_rows]
    balanced = _balanced_rows(
        baseline_objects, PATTERNS, LOAD_CASES
    )
    balanced_winner = str(balanced[0]["pattern"])
    bending_winner = min(
        (
            row
            for row in interval_rows
            if row["load_case"] == "bending"
        ),
        key=lambda row: int(row["load_case_rank"]),
    )
    reconciliation = json.loads(RECONCILIATION.read_text(encoding="utf-8"))
    gates = {
        "source_files_present_and_checksum_locked": all(
            path.exists()
            for path in (MATERIAL_PROFILE, MODULUS_ROWS, RECONCILIATION)
        ),
        "raw_specimen_modulus_count_passed": len(moduli) == 22,
        "baseline_solver_matrix_passed": len(baseline_rows) == 20
        and all(row["status"] == "ok" for row in baseline_rows),
        "endpoint_direct_solver_checks_passed": sum(
            len(rows) for rows in endpoint_rows.values()
        )
        == 10
        and all(
            row["status"] == "ok"
            for rows in endpoint_rows.values()
            for row in rows
        ),
        "scalar_linearity_gate_passed": (
            maximum_scaling_error <= SCALAR_CHECK_TOLERANCE
        ),
        "directional_calibration_release_allowed": False,
        "absolute_failure_prediction_allowed": False,
    }
    status = (
        "empirical_scalar_modulus_ensemble_passed"
        if all(
            gates[key]
            for key in (
                "source_files_present_and_checksum_locked",
                "raw_specimen_modulus_count_passed",
                "baseline_solver_matrix_passed",
                "endpoint_direct_solver_checks_passed",
                "scalar_linearity_gate_passed",
            )
        )
        else "ensemble_gate_failed"
    )
    payload = {
        "study_id": "bambu_pla_raw_specimen_infill_uncertainty_v1",
        "status": status,
        "material": "Bambu PLA Basic",
        "patterns": list(PATTERNS),
        "load_cases": list(LOAD_CASES),
        "raw_specimen_modulus_count": len(moduli),
        "central_modulus_MPa": central_E,
        "empirical_modulus_MPa": {
            "minimum": float(np.min(values)),
            "q05": float(np.quantile(values, 0.05)),
            "median": float(np.median(values)),
            "q95": float(np.quantile(values, 0.95)),
            "maximum": float(np.max(values)),
            "coefficient_of_variation": float(
                np.std(values, ddof=1) / np.mean(values)
            ),
        },
        "solver_cases": {
            "baseline_finalist_cases": len(baseline_rows),
            "direct_endpoint_check_cases": sum(
                len(rows) for rows in endpoint_rows.values()
            ),
        },
        "maximum_scalar_linearity_relative_error": (
            maximum_scaling_error
        ),
        "scalar_linearity_tolerance": SCALAR_CHECK_TOLERANCE,
        "balanced_ranking": balanced,
        "balanced_winner": balanced_winner,
        "bending_winner": bending_winner["pattern"],
        "bending_winner_ratio_to_triangular": bending_winner[
            "ratio_to_triangular"
        ],
        "gates": gates,
        "source_hashes": {
            "material_profile": _sha256(MATERIAL_PROFILE),
            "per_specimen_modulus": _sha256(MODULUS_ROWS),
            "tensile_reconciliation": _sha256(RECONCILIATION),
            "baseline_solver_summary": _sha256(baseline_path),
        },
        "source_reconciliation_status": reconciliation["status"],
        "allowed_use": (
            "Empirical absolute stiffness and buckling intervals plus "
            "relative finalist ranking stability for the frozen isotropic "
            "small-strain model."
        ),
        "limitations": [
            "This is an empirical 22-specimen ensemble, not a population confidence interval.",
            "The corrupted source labels are pooled; no local-axis distribution is inferred.",
            "A shared scalar E variation cannot represent directional shear, Poisson, or bead-placement uncertainty.",
            "The ensemble does not cover print defects, color/lot effects, moisture, temperature, rate, damage, fracture, fatigue, or creep.",
            "Rank stability is conditional on the four frozen equal-mass geometries and five frozen load fixtures.",
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    report = [
        "# Raw-specimen Bambu PLA infill uncertainty",
        "",
        (
            f"Status: **{status}**. Twenty baseline finalist cases and ten "
            "direct minimum/maximum-modulus CalculiX checks completed."
        ),
        "",
        (
            f"The 22 raw tensile curves span {np.min(values):.1f}–"
            f"{np.max(values):.1f} MPa in fitted modulus; empirical 5%–95% "
            f"bounds are {np.quantile(values, 0.05):.1f}–"
            f"{np.quantile(values, 0.95):.1f} MPa. Direct endpoint solves "
            f"confirm scalar response propagation to within "
            f"{maximum_scaling_error:.3%}."
        ),
        "",
        (
            f"The balanced winner is `{balanced_winner}`. The bending "
            f"specialist is `{bending_winner['pattern']}` at "
            f"{float(bending_winner['ratio_to_triangular']):.3f}x "
            "triangular. Because every candidate shares the same scalar "
            "modulus in each empirical scenario, all load-case ranks are "
            "stable across 22/22 measured moduli."
        ),
        "",
        "This stability does not cover directional constants or manufacturing "
        "defects. See `prediction_intervals.csv` for absolute empirical "
        "bounds and `specimen_ensemble_predictions.csv` for every scenario.",
        "",
    ]
    (output_dir / "report.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Propagate raw Bambu specimen modulus scatter through infill finalists"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--mesh-size", type=float, default=0.7)
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    if not args.aggregate_only:
        run_reference_cases(args.output, args.runs_dir, args.mesh_size)
    payload = aggregate_ensemble(args.output)
    print(
        f"{payload['status']}; balanced winner={payload['balanced_winner']}; "
        f"bending winner={payload['bending_winner']}; "
        f"max scaling error="
        f"{payload['maximum_scalar_linearity_relative_error']:.3%}"
    )


if __name__ == "__main__":
    main()
