from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT_ROOT / "data/public/bisic-polymers-2025"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "data/reported/bambu_pla_ballistic_damage_2025"
)
ARTICLE_DOI = "10.3390/polym17131854"
ARTICLE_URL = "https://doi.org/10.3390/polym17131854"
FIGURE6_HIGHRES_URL = (
    "https://mdpi-res.com/d_attachment/polymers/"
    "polymers-17-01854/article_deploy/html/images/"
    "polymers-17-01854-g006.png"
)
FIGURE6_DIGITIZATION_FILE = "figure-006-highres.png"
EXPECTED_SHA256 = {
    "paper.pdf": (
        "b11de514b617a4bc3f55f32f20b90152d3ca6d1abcb189fb6e09ac72ec376a5e"
    ),
    "article.xml": (
        "2157484fb9dc5a65c93153b8760996dd0a4486e4ef63bb2e0fea79a2fa6c8e5a"
    ),
    "figure-006.jpg": (
        "5d83d786022c347ddef1b6e5d9284e4f10b75d8951ba5665063c6e327abc55dd"
    ),
    FIGURE6_DIGITIZATION_FILE: (
        "6e01f342c1852fa997fb880658dfd47588a2c6acf57aba9092ab02c5db85e931"
    ),
}

TABLE4_MEASURED = {
    "tensile_strength_MPa": 34.1,
    "tensile_modulus_MPa": 2600.0,
    "tensile_breaking_elongation_percent": 8.8,
    "flexural_strength_MPa": 78.0,
}
TABLE5_MODEL = {
    "density_kg_m3": 1250.0,
    "E_x_MPa": 2580.0,
    "E_y_MPa": 2060.0,
    "E_z_MPa": 2580.0,
    "nu_xy": 0.33,
    "nu_xz": 0.33,
    "nu_yz": 0.30,
    "G_xy_MPa": 970.0,
    "G_xz_MPa": 970.0,
    "G_yz_MPa": 775.0,
    "ballistic_yield_stress_MPa": 50.0,
    "plastic_strain_at_60_MPa": 0.05,
    "damage_initiation_equivalent_plastic_strain": 0.05,
    "fracture_energy_kJ_m2": 2.5,
}
PRINT_PROCESS = {
    "printer": "Bambu Lab X1C",
    "material": "Bambu PLA Basic black",
    "nozzle_diameter_mm": 0.4,
    "layer_height_mm": 0.2,
    "wall_width_inner_mm": 0.45,
    "wall_width_outer_mm": 0.42,
    "nozzle_temperature_C": 220,
    "print_speed_outer_mm_s": 200,
    "print_speed_inner_mm_s": 300,
    "bed_temperature_C": 35,
}
BALLISTIC_TEST = {
    "projectile_mass_kg": 0.008,
    "projectile_diameter_m": 0.009,
    "mean_velocity_m_s": 355.0,
    "reported_velocity_range_m_s": [354.0, 368.0],
    "target_length_mm": 200.0,
    "target_width_mm": 60.0,
    "target_height_mm": 60.0,
    "infill_density_percent": 50.0,
    "replicates_per_pattern": 3,
}
PATTERN_RESULTS = {
    "cubic": {
        "mass_g": 474.74,
        "penetration_mm": [161.64, 162.41, 160.50],
    },
    "grid": {
        "mass_g": 472.26,
        "penetration_mm": [167.95, 169.12, 170.55],
    },
    "gyroid": {
        "mass_g": 448.45,
        "penetration_lower_bound_mm": 200.0,
    },
    "honeycomb": {
        "mass_g": 545.78,
        "penetration_mm": [110.74, 110.19, 111.25],
    },
}

FIGURE6_PANELS = {
    "tensile": {
        "x_left": 178,
        "x_right": 2072,
        "y_top": 43,
        "y_bottom": 1261,
        "peak_x_window": (430, 700),
        "curve_y_window": (760, 1050),
        "table_strength_MPa": 34.1,
    },
    "flexural": {
        "x_left": 2353,
        "x_right": 4187,
        "y_top": 71,
        "y_bottom": 1243,
        "peak_x_window": (2670, 3000),
        "curve_y_window": (250, 800),
        "table_strength_MPa": 78.0,
    },
}
CURVE_COLORS = ("red", "green", "blue", "magenta")


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
    colored = (saturation > 0.12) & (value > 0.25)
    return {
        "red": colored & ((hue < 0.08) | (hue > 0.95)),
        "green": colored & (hue > 0.25) & (hue < 0.48),
        "blue": colored & (hue > 0.50) & (hue < 0.75),
        "magenta": colored & (hue > 0.75) & (hue < 0.95),
    }


def _pixel_to_curve(
    x_pixel: np.ndarray | float,
    y_pixel: np.ndarray | float,
    definition: Mapping[str, object],
) -> tuple[np.ndarray | float, np.ndarray | float]:
    strain = (
        (x_pixel - float(definition["x_left"]))
        / (float(definition["x_right"]) - float(definition["x_left"]))
        * 0.10
    )
    stress = (
        (float(definition["y_bottom"]) - y_pixel)
        / (float(definition["y_bottom"]) - float(definition["y_top"]))
        * 100.0
    )
    return strain, stress


def digitize_figure6(
    source_dir: Path = DEFAULT_SOURCE,
) -> tuple[Dict[str, Dict[str, List[Dict[str, float]]]], List[Dict[str, object]]]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "ballistic-reference digitization requires Pillow"
        ) from exc
    image = Image.open(Path(source_dir) / FIGURE6_DIGITIZATION_FILE)
    rgb = np.asarray(image.convert("RGB"))
    hsv = np.asarray(image.convert("HSV"))
    if rgb.shape != (1579, 4262, 3):
        raise ValueError(f"Figure 6 dimensions changed: {rgb.shape}")
    masks = _color_masks(hsv)
    curves: Dict[str, Dict[str, List[Dict[str, float]]]] = {}
    validation: List[Dict[str, object]] = []
    for response, definition in FIGURE6_PANELS.items():
        response_curves: Dict[str, List[Dict[str, float]]] = {}
        response_peaks = []
        for color in CURVE_COLORS:
            mask = masks[color]
            points = []
            y_min, y_max = definition["curve_y_window"]
            for x_pixel in range(
                int(definition["x_left"]) + 1,
                int(definition["x_right"]),
            ):
                candidates = (
                    np.flatnonzero(mask[int(y_min):int(y_max) + 1, x_pixel])
                    + int(y_min)
                )
                if len(candidates) == 0:
                    continue
                groups = np.split(
                    candidates,
                    np.where(np.diff(candidates) > 2)[0] + 1,
                )
                group = max(groups, key=len)
                points.append(
                    (float(x_pixel), float(np.median(group)))
                )
            array = np.asarray(points, dtype=float)
            peak_low, peak_high = definition["peak_x_window"]
            peak_window = array[
                (array[:, 0] >= float(peak_low))
                & (array[:, 0] <= float(peak_high))
            ]
            if len(peak_window) == 0:
                raise ValueError(
                    f"no {response}/{color} peak pixels in Figure 6"
                )
            peak = peak_window[int(np.argmin(peak_window[:, 1]))]
            postpeak = array[array[:, 0] >= peak[0]]
            runs = np.split(
                np.arange(len(postpeak)),
                np.where(np.diff(postpeak[:, 0]) > 15)[0] + 1,
            )
            retained = [run for run in runs if len(run) >= 4]
            postpeak = np.vstack([postpeak[run] for run in retained])
            source_column_gaps = np.diff(postpeak[:, 0])
            maximum_interpolated_gap_pixels = max(
                0.0,
                float(np.max(source_column_gaps)) - 1.0,
            )
            maximum_interpolated_gap_strain = (
                maximum_interpolated_gap_pixels
                / (
                    float(definition["x_right"])
                    - float(definition["x_left"])
                )
                * 0.10
            )
            x_regular = np.arange(
                peak[0], float(np.max(postpeak[:, 0])) + 0.1, 1.0
            )
            y_regular = np.interp(
                x_regular, postpeak[:, 0], postpeak[:, 1]
            )
            strain, stress = _pixel_to_curve(
                x_regular, y_regular, definition
            )
            response_curves[color] = [
                {
                    "engineering_strain": float(current_strain),
                    "engineering_stress_MPa": float(current_stress),
                    "source_pixel_x": float(x_pixel),
                    "source_pixel_y": float(y_pixel),
                }
                for current_strain, current_stress, x_pixel, y_pixel in zip(
                    strain, stress, x_regular, y_regular
                )
            ]
            response_peaks.append(float(np.max(stress)))
            validation.append(
                {
                    "response": response,
                    "visible_curve_color": color,
                    "digitized_peak_stress_MPa": float(np.max(stress)),
                    "digitized_peak_strain": float(strain[0]),
                    "digitized_visible_end_strain": float(strain[-1]),
                    "digitized_visible_end_stress_MPa": float(stress[-1]),
                    "digitized_points": len(stress),
                    "source_columns_observed": len(postpeak),
                    "interpolated_column_fraction": (
                        1.0 - len(postpeak) / len(stress)
                    ),
                    "maximum_interpolated_gap_pixels": (
                        maximum_interpolated_gap_pixels
                    ),
                    "maximum_interpolated_gap_strain": (
                        maximum_interpolated_gap_strain
                    ),
                    "passes_trace_continuity_gate": (
                        maximum_interpolated_gap_strain <= 0.005
                    ),
                    "evidence_class": (
                        "visible colored specimen trace digitized from the "
                        "official high-resolution primary-source PNG; only "
                        "four distinct traces are independently visible"
                    ),
                }
            )
        average_peak = float(np.mean(response_peaks))
        table_strength = float(definition["table_strength_MPa"])
        relative_error = average_peak / table_strength - 1.0
        for row in validation:
            if row["response"] == response:
                row["four_curve_average_peak_MPa"] = average_peak
                row["reported_five_specimen_average_strength_MPa"] = (
                    table_strength
                )
                row["average_peak_relative_error"] = relative_error
                row["passes_table_crosscheck"] = (
                    abs(relative_error) <= 0.05
                )
        curves[response] = response_curves
    return curves, validation


def damage_curve(
    characteristic_length_mm: float,
    points_per_branch: int = 101,
) -> List[Dict[str, float]]:
    """Crack-band realization of the paper's author-specified damage model."""
    if characteristic_length_mm <= 0.0:
        raise ValueError("characteristic length must be positive")
    elastic_modulus = TABLE5_MODEL["E_x_MPa"]
    yield_stress = TABLE5_MODEL["ballistic_yield_stress_MPa"]
    damage_stress = 60.0
    damage_plastic_strain = TABLE5_MODEL[
        "damage_initiation_equivalent_plastic_strain"
    ]
    fracture_energy_N_mm = TABLE5_MODEL["fracture_energy_kJ_m2"]
    final_crack_opening_mm = (
        2.0 * fracture_energy_N_mm / damage_stress
    )
    final_plastic_strain = (
        damage_plastic_strain
        + final_crack_opening_mm / characteristic_length_mm
    )
    rows: List[Dict[str, float]] = []
    elastic_strain = np.linspace(
        0.0, yield_stress / elastic_modulus, points_per_branch
    )
    for strain in elastic_strain:
        rows.append(
            {
                "total_strain": float(strain),
                "nominal_stress_MPa": float(elastic_modulus * strain),
                "equivalent_plastic_strain": 0.0,
                "damage": 0.0,
                "crack_opening_mm": 0.0,
            }
        )
    plastic_strain = np.linspace(
        0.0, damage_plastic_strain, points_per_branch
    )[1:]
    for current_plastic in plastic_strain:
        flow_stress = (
            yield_stress
            + (damage_stress - yield_stress)
            * current_plastic / damage_plastic_strain
        )
        rows.append(
            {
                "total_strain": float(
                    current_plastic + flow_stress / elastic_modulus
                ),
                "nominal_stress_MPa": float(flow_stress),
                "equivalent_plastic_strain": float(current_plastic),
                "damage": 0.0,
                "crack_opening_mm": 0.0,
            }
        )
    softening_plastic = np.linspace(
        damage_plastic_strain,
        final_plastic_strain,
        points_per_branch,
    )[1:]
    for current_plastic in softening_plastic:
        crack_opening = (
            (current_plastic - damage_plastic_strain)
            * characteristic_length_mm
        )
        damage = min(1.0, crack_opening / final_crack_opening_mm)
        rows.append(
            {
                "total_strain": float(
                    current_plastic + damage_stress / elastic_modulus
                ),
                "nominal_stress_MPa": float(
                    (1.0 - damage) * damage_stress
                ),
                "equivalent_plastic_strain": float(current_plastic),
                "damage": float(damage),
                "crack_opening_mm": float(crack_opening),
            }
        )
    return rows


def damage_energy_N_mm(curve: Sequence[Mapping[str, float]]) -> float:
    softening = [
        row for row in curve
        if float(row["equivalent_plastic_strain"])
        >= TABLE5_MODEL["damage_initiation_equivalent_plastic_strain"]
    ]
    return float(
        np.trapezoid(
            [float(row["nominal_stress_MPa"]) for row in softening],
            [float(row["crack_opening_mm"]) for row in softening],
        )
    )


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
    curves, validation = digitize_figure6(source_dir)
    peak_crosschecks = {}
    for response in FIGURE6_PANELS:
        response_rows = [
            row for row in validation if row["response"] == response
        ]
        peak_crosschecks[response] = {
            "separable_trace_count": len(response_rows),
            "four_trace_average_peak_MPa": response_rows[0][
                "four_curve_average_peak_MPa"
            ],
            "reported_five_specimen_average_strength_MPa": response_rows[0][
                "reported_five_specimen_average_strength_MPa"
            ],
            "relative_error": response_rows[0][
                "average_peak_relative_error"
            ],
        }
    for response, response_curves in curves.items():
        for color, points in response_curves.items():
            rows = [
                {
                    "material": "Bambu PLA Basic black",
                    "response": response,
                    "visible_curve_color": color,
                    "evidence_class": (
                        "published specimen-curve figure digitization"
                    ),
                    **point,
                }
                for point in points
            ]
            _write_csv(
                output_dir / "curves" / f"{response}_{color}.csv",
                rows,
            )
    _write_csv(output_dir / "curve_validation.csv", validation)

    damage_validation = []
    for length in (0.5, 1.0, 2.0):
        curve = damage_curve(length)
        _write_csv(
            output_dir
            / "curves"
            / f"author_damage_Lc_{str(length).replace('.', 'p')}mm.csv",
            curve,
        )
        energy = damage_energy_N_mm(curve)
        damage_validation.append(
            {
                "characteristic_length_mm": length,
                "integrated_softening_energy_N_mm": energy,
                "target_fracture_energy_N_mm": TABLE5_MODEL[
                    "fracture_energy_kJ_m2"
                ],
                "relative_error": (
                    energy
                    / TABLE5_MODEL["fracture_energy_kJ_m2"]
                    - 1.0
                ),
                "full_failure_total_strain": curve[-1][
                    "total_strain"
                ],
                "passes_energy_gate": abs(
                    energy
                    / TABLE5_MODEL["fracture_energy_kJ_m2"]
                    - 1.0
                )
                <= 0.01,
            }
        )
    _write_csv(
        output_dir / "damage_regularization_validation.csv",
        damage_validation,
    )
    model = {
        "material": "Bambu PLA Basic black",
        "table4_measured_quasistatic": TABLE4_MEASURED,
        "table5_author_ballistic_model": TABLE5_MODEL,
        "print_process": PRINT_PROCESS,
        "ballistic_test": BALLISTIC_TEST,
        "critical_interpretation": {
            "quasistatic_yield_observation_MPa": (
                "approximately 35 with minimal post-yield hardening"
            ),
            "author_ballistic_yield_MPa": 50.0,
            "strain_rate_law_in_FE_model": None,
            "rate_effect_status": (
                "50 MPa was rationalized using strain-rate sensitivity, "
                "but the paper later states that rate hardening was not "
                "included"
            ),
            "damage_status": (
                "author-calibrated/literature-grounded model; not direct "
                "fracture-coupon identification"
            ),
            "FE_validation": (
                "homogenized grid target: 184 mm simulation versus "
                "169.2 mm measured mean; internal 50% infill not explicit"
            ),
        },
    }
    (output_dir / "author_model.json").write_text(
        json.dumps(model, indent=2) + "\n", encoding="utf-8"
    )
    provenance = {
        "article": (
            "Influence of Infill Pattern on Ballistic Resistance "
            "Capabilities of 3D-Printed Polymeric Structures"
        ),
        "journal": "Polymers",
        "year": 2025,
        "doi": ARTICLE_DOI,
        "url": ARTICLE_URL,
        "license": "CC BY 4.0",
        "checksums": checksums,
        "figure6_digitization_asset": {
            "filename": FIGURE6_DIGITIZATION_FILE,
            "official_source_url": FIGURE6_HIGHRES_URL,
            "pixel_dimensions": [4262, 1579],
            "panel_axis_bounds_pixels": FIGURE6_PANELS,
        },
        "five_tensile_specimens_reported": True,
        "five_flexural_specimens_reported": True,
        "separable_colored_curves_per_response": 4,
        "peak_crosschecks": peak_crosschecks,
        "curve_quality_gate_passed": all(
            bool(row["passes_table_crosscheck"])
            and bool(row["passes_trace_continuity_gate"])
            for row in validation
        ),
        "damage_energy_gate_passed": all(
            bool(row["passes_energy_gate"]) for row in damage_validation
        ),
        "limitations": [
            "The article reports five specimens per response, but the "
            "official high-resolution Figure 6 rendering contains only "
            "four independently separable colored traces. A fifth trace "
            "may be fully overplotted or absent from the rendering and is "
            "not fabricated.",
            "Machine-exported raw tensile/flexural data are unavailable.",
            "The 50 MPa ballistic yield exceeds the measured quasi-static "
            "strength and is an author modeling choice.",
            "No explicit strain-rate constitutive law was used.",
            "Damage initiation and fracture energy were not identified from "
            "matched exact-brand fracture coupons.",
            "The paper's FE validation homogenized the grid target rather "
            "than meshing its internal infill.",
        ],
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    report = f"""# Bambu PLA Basic ballistic/damage reference

## Measured evidence

The peer-reviewed source tested five tensile and five flexural specimens made
from black Bambu PLA Basic on a Bambu X1C. Table 4 reports 34.1 MPa tensile
strength, 2600 MPa tensile modulus, 8.8% tensile breaking elongation, and
78 MPa flexural strength. The official high-resolution primary-source PNG
contains four independently separable colored specimen traces for each
response. Their average digitized peaks cross-check the reported five-specimen
averages to {peak_crosschecks["tensile"]["relative_error"]:+.2%} for tensile
and {peak_crosschecks["flexural"]["relative_error"]:+.2%} for flexure.
Although the article reports five tested specimens, a fifth distinct trace is
not independently visible in the rendering; it may be fully overplotted or
omitted and is not fabricated by the importer. The continuity gate records
every bridged source-column gap and limits the largest to 0.5 percentage
points of engineering strain.

## Author damage model

The source's Abaqus model uses 50 MPa yield, hardening to 60 MPa at 5%
equivalent plastic strain, damage initiation at 5%, linear energy-regularized
softening with 2.5 kJ/m2 fracture energy, and element deletion. The exported
0.5, 1.0, and 2.0 mm crack-band material-point curves integrate to the same
2.5 N/mm softening energy.

This is useful model evidence, but not direct material identification. The
measured quasi-static specimens yielded at about 35 MPa with little hardening;
50 MPa was selected for the impact model. The paper invokes strain-rate
effects to rationalize that choice but later states that rate hardening was
not actually included. The damage parameters are literature-grounded and
calibrated, not measured in matched exact-brand fracture coupons.

## Validation boundary

The author FE model predicts 184 mm penetration versus 169.2 mm measured for
the grid target, about 9% high, with a qualitatively similar cavity. It uses a
homogenized solid block rather than explicit 50% infill. Therefore this model
is a bounded high-rate damage sensitivity and penetration reference, not a
validated universal sparse-infill fracture law.
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    return provenance


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build exact-brand ballistic and damage reference"
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = write_reference(args.source, args.output)
    print(
        "wrote ballistic/damage reference; "
        f"curve gate={result['curve_quality_gate_passed']}, "
        f"energy gate={result['damage_energy_gate_passed']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
