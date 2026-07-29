from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_ROOT / "data/public/adaptive-infill-rs-7965413"
SOURCE_PDF = SOURCE_DIR / "paper.pdf"
SOURCE_SHA256 = (
    "175ae8476430b3f9b7a074660095da6d717adf2fc02589a44399b4eabb28fcc0"
)
SOURCE_URL = (
    "https://assets-eu.researchsquare.com/files/rs-7965413/"
    "v1_covered_bb4415c2-c01c-4970-b519-47d879b8a37a.pdf"
)
DATA_DIR = (
    PROJECT_ROOT
    / "data/reported/bambu_pla_adaptive_infill_preprint"
)
DEFAULT_RESULTS = (
    PROJECT_ROOT / "results/adaptive_infill_external_validation"
)
PATTERNS = ("zigzag_90", "zigzag_45", "hilbert")


def _row(
    mode: str,
    density: float,
    adaptive: bool,
    sample: int,
    pattern: str,
    mass_g: float,
    peak_load_N: float,
    energy_N_mm: float | None = None,
    specific_energy_N_mm_g: float | None = None,
) -> Dict[str, object]:
    return {
        "test_mode": mode,
        "nominal_infill_fraction": density,
        "density_strategy": "adaptive" if adaptive else "uniform",
        "sample": sample,
        "pattern": pattern,
        "mass_g": mass_g,
        "peak_load_N": peak_load_N,
        "energy_absorption_N_mm": energy_N_mm,
        "specific_energy_absorption_N_mm_per_g": specific_energy_N_mm_g,
    }


def physical_results() -> List[Dict[str, object]]:
    """Transcribe the labeled bars in Figures 7, 8, 10, and 11."""
    return [
        _row("compression", .20, False, 1, "zigzag_90", 16.93, 2028, 16420, 969.88),
        _row("compression", .20, False, 2, "zigzag_45", 16.93, 1925, 14536, 858.59),
        _row("compression", .20, False, 3, "hilbert", 16.77, 2589, 18749, 1118.00),
        _row("compression", .20, True, 4, "zigzag_90", 16.89, 2199, 19051, 1127.95),
        _row("compression", .20, True, 5, "zigzag_45", 16.89, 2067, 17860, 1057.43),
        _row("compression", .20, True, 6, "hilbert", 16.60, 3195, 22249, 1340.30),
        _row("compression", .40, False, 7, "zigzag_90", 19.01, 2895, 23590, 1235.08),
        _row("compression", .40, False, 8, "zigzag_45", 19.01, 2471, 20766, 1087.23),
        _row("compression", .40, False, 9, "hilbert", 18.66, 3863, 25831, 1384.30),
        _row("compression", .40, True, 10, "zigzag_90", 16.89, 3167, 30240, 1624.06),
        _row("compression", .40, True, 11, "zigzag_45", 16.89, 3009, 24863, 1335.28),
        _row("compression", .40, True, 12, "hilbert", 16.60, 4295, 35300, 1893.78),
        _row("tension", .20, False, 1, "zigzag_90", 21.42, 827),
        _row("tension", .20, False, 2, "zigzag_45", 21.42, 769),
        _row("tension", .20, False, 3, "hilbert", 21.27, 1147),
        _row("tension", .20, True, 4, "zigzag_90", 20.43, 931),
        _row("tension", .20, True, 5, "zigzag_45", 20.43, 910),
        _row("tension", .20, True, 6, "hilbert", 21.23, 1612),
        _row("tension", .40, False, 7, "zigzag_90", 23.58, 1164),
        _row("tension", .40, False, 8, "zigzag_45", 23.58, 1110),
        _row("tension", .40, False, 9, "hilbert", 22.91, 1663),
        _row("tension", .40, True, 10, "zigzag_90", 22.46, 1530),
        _row("tension", .40, True, 11, "zigzag_45", 22.46, 1436),
        _row("tension", .40, True, 12, "hilbert", 22.02, 2348),
    ]


def source_checksum(path: Path = SOURCE_PDF) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rank_patterns(
    rows: Iterable[Mapping[str, object]],
    metric: str,
) -> List[str]:
    return [
        str(row["pattern"])
        for row in sorted(
            rows, key=lambda item: float(item[metric]), reverse=True
        )
    ]


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _physical_ranking_rows(
    rows: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    result = []
    for mode in ("compression", "tension"):
        for density in (.20, .40):
            subset = [
                row for row in rows
                if row["test_mode"] == mode
                and row["nominal_infill_fraction"] == density
                and row["density_strategy"] == "uniform"
            ]
            ranking = rank_patterns(subset, "peak_load_N")
            result.append(
                {
                    "test_mode": mode,
                    "nominal_infill_fraction": density,
                    "ranking": ranking,
                    "hilbert_is_best": ranking[0] == "hilbert",
                }
            )
    return result


def _adaptive_rows(
    rows: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    result = []
    for mode in ("compression", "tension"):
        for density in (.20, .40):
            for pattern in PATTERNS:
                control = next(
                    row for row in rows
                    if row["test_mode"] == mode
                    and row["nominal_infill_fraction"] == density
                    and row["density_strategy"] == "uniform"
                    and row["pattern"] == pattern
                )
                adaptive = next(
                    row for row in rows
                    if row["test_mode"] == mode
                    and row["nominal_infill_fraction"] == density
                    and row["density_strategy"] == "adaptive"
                    and row["pattern"] == pattern
                )
                result.append(
                    {
                        "test_mode": mode,
                        "nominal_infill_fraction": density,
                        "pattern": pattern,
                        "peak_load_change_percent": (
                            float(adaptive["peak_load_N"])
                            / float(control["peak_load_N"])
                            - 1.0
                        ) * 100.0,
                        "mass_change_percent": (
                            float(adaptive["mass_g"])
                            / float(control["mass_g"])
                            - 1.0
                        ) * 100.0,
                    }
                )
    return result


def _simulation_ranking_rows(
    results_root: Path,
) -> List[Dict[str, object]]:
    result = []
    for density_label, density in (("20", .20), ("40", .40)):
        beam_rows = _read_csv(
            results_root / f"density_{density_label}" / "summary.csv"
        )
        for mode in ("compression", "tension"):
            subset = [
                row for row in beam_rows
                if row["load_case"] == mode and row["status"] == "ok"
            ]
            result.append(
                {
                    "fidelity": "beam",
                    "test_mode": mode,
                    "nominal_infill_fraction": density,
                    "metric": "elastic_stiffness_N_per_mm",
                    "ranking": rank_patterns(subset, "stiffness_N_per_mm"),
                }
            )
        solid_rows = json.loads(
            (
                results_root
                / f"solid_density_{density_label}"
                / "summary.json"
            ).read_text(encoding="utf-8")
        )
        result.append(
            {
                "fidelity": "solid",
                "test_mode": "tension",
                "nominal_infill_fraction": density,
                "metric": "elastic_stiffness_N_per_mm",
                "ranking": rank_patterns(solid_rows, "stiffness_N_per_mm"),
            }
        )
    return result


def write_validation(
    results_root: Path = DEFAULT_RESULTS,
    data_dir: Path = DATA_DIR,
) -> Dict[str, object]:
    if source_checksum() != SOURCE_SHA256:
        raise RuntimeError("adaptive-infill source PDF checksum mismatch")
    results_root = Path(results_root)
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    rows = physical_results()
    fields = list(rows[0])
    with (data_dir / "physical_results.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    physical_rankings = _physical_ranking_rows(rows)
    adaptive = _adaptive_rows(rows)
    simulation_rankings = _simulation_ranking_rows(results_root)
    expected = ["hilbert", "zigzag_90", "zigzag_45"]
    for row in simulation_rankings:
        row["matches_physical_full_ranking"] = row["ranking"] == expected
        row["matches_zigzag_pair_order"] = (
            row["ranking"].index("zigzag_90")
            < row["ranking"].index("zigzag_45")
        )
    summary = {
        "source_sha256": SOURCE_SHA256,
        "physical_uniform_rankings": physical_rankings,
        "adaptive_matched_changes": adaptive,
        "simulation_rankings": simulation_rankings,
        "physical_hilbert_best_cases": sum(
            row["hilbert_is_best"] for row in physical_rankings
        ),
        "physical_case_count": len(physical_rankings),
        "simulation_full_ranking_matches": sum(
            row["matches_physical_full_ranking"]
            for row in simulation_rankings
        ),
        "simulation_case_count": len(simulation_rankings),
        "simulation_zigzag_pair_matches": sum(
            row["matches_zigzag_pair_order"]
            for row in simulation_rankings
        ),
        "validation_status": "qualitative_ranking_failed",
        "permitted_use": (
            "elastic stiffness and topology screening at equal mass"
        ),
        "prohibited_use": (
            "absolute failure-load, interlayer-bond, or crash-energy prediction"
        ),
    }
    (results_root / "validation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    provenance = {
        "title": (
            "Adaptive Infill Method Based on Cross-Section Features and "
            "Transition-Layer Design"
        ),
        "source_url": SOURCE_URL,
        "local_pdf": str(SOURCE_PDF.relative_to(PROJECT_ROOT)),
        "sha256": SOURCE_SHA256,
        "status": "Research Square preprint; not peer reviewed",
        "material": "Bambu Lab PLA Basic",
        "printer": "Bambu Lab P1S",
        "process": {
            "nozzle_temperature_C": 220,
            "chamber_temperature_C": 45,
            "nozzle_diameter_mm": 0.4,
            "print_speed_mm_min": 1200,
            "filament_diameter_mm": 1.75,
        },
        "quality_limits": [
            "one specimen per parameter configuration",
            "no raw numeric curves or G-code",
            "exact internal geometry and full specimen dimensions unavailable",
            "compression Figure 8 label/text inconsistency retained",
            "reported modulus and Poisson ratio are assumed, not measured",
        ],
    }
    (data_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )

    adaptive_min = min(row["peak_load_change_percent"] for row in adaptive)
    adaptive_max = max(row["peak_load_change_percent"] for row in adaptive)
    report = f"""# External adaptive-infill check

## Outcome

The exact-brand physical study ranks **Hilbert > zigzag 90° > zigzag 45°**
in all four uniform-infill comparisons (tension/compression at nominal 20%
and 40%). Its matched adaptive-density specimens improve peak load by
{adaptive_min:.1f}% to {adaptive_max:.1f}% while changing mass by -11.1% to
-0.2%.

Our equal-material elastic models do **not** reproduce the full physical
ranking: {summary['simulation_full_ranking_matches']}/{summary['simulation_case_count']} beam/solid
checks match. They do reproduce zigzag 90° > zigzag 45° in
{summary['simulation_zigzag_pair_matches']}/{summary['simulation_case_count']} checks.
Both beam and 3D solid models favor the load-aligned zigzag over Hilbert.

## Interpretation

This is a useful failed held-out validation, not a reason to tune the geometry
until the answer agrees. The external measurements are peak failure load,
interlayer bond strength, and post-collapse energy. The current models predict
pre-failure elastic response with a homogeneous material and an idealized
40 mm square wall-plus-infill coupon. The paper does not publish enough
geometry or raw data to reproduce its specimen exactly.

Use the simulator to screen equal-mass elastic stiffness, buckling modes,
stress hot spots, and robustness. Do not use it yet to claim absolute ultimate
load, interlayer failure, or crash-energy performance. A future calibrated
interlayer/cohesive damage law and matched exported toolpaths are required
before promoting those claims.

## Source quality

The source is a 2026 Research Square preprint, not peer reviewed, with one
specimen per configuration. The local PDF is checksum pinned at
`{SOURCE_SHA256}`. Values in `physical_results.csv` are transcribed from the
labeled bars in Figures 7, 8, 10, and 11; no uncertainty bars were reported.
"""
    (results_root / "report.md").write_text(report, encoding="utf-8")
    source_quality = report.split("## Source quality", 1)[1].strip()
    (data_dir / "data_quality_report.md").write_text(
        f"# Data quality\n\n{source_quality}\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit simulated infill rankings against an exact-brand preprint"
    )
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()
    summary = write_validation(args.results, args.data_dir)
    print(
        f"external ranking matches "
        f"{summary['simulation_full_ranking_matches']}/"
        f"{summary['simulation_case_count']}; "
        f"report: {(args.results / 'report.md').resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
