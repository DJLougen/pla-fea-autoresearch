from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Mapping, Tuple

from .materials import check_admissibility


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "results/bambu_tds_profiles"
DEFAULT_ISOTROPIC_PROFILE = (
    PROJECT_ROOT / "data/reported/bambu_pla_tds_isotropic.json"
)
DEFAULT_TRANSVERSE_PROFILE = (
    PROJECT_ROOT
    / "data/reported/bambu_pla_tds_transverse_sensitivity.json"
)
DEFAULT_POISSON_REFERENCE = (
    PROJECT_ROOT
    / "data/reported/bambu_pla_viscoelastic_wave_2026/provenance.json"
)
TDS_SOURCE = (
    "https://manuals.plus/m/"
    "3e29a29bc0a01fb36a156bc22f81655976678d017876dcea051f6250cb7892e2"
)


def _measured_in_plane_poisson(
    reference_path: Path = DEFAULT_POISSON_REFERENCE,
) -> Tuple[float, Dict[str, object]]:
    reference_path = Path(reference_path)
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    if (
        reference.get("reference_id")
        != "bambu_pla_viscoelastic_wave_2026_v1"
        or reference.get("material") != "Bambu PLA Basic"
        or reference.get("quality_gate_passed") is not True
        or "in-plane flat Poisson screen"
        not in reference.get("admitted_uses", [])
    ):
        raise ValueError(
            "exact-brand in-plane Poisson reference is not admitted"
        )
    value = float(
        reference["reported_static_measurements"]["poisson_ratio"]
    )
    if not 0.0 < value < 0.5:
        raise ValueError("measured in-plane Poisson ratio is implausible")
    scope = reference["print_and_test_scope"]
    return value, {
        "reference_id": reference["reference_id"],
        "reference_path": str(reference_path.resolve()),
        "article_doi": reference["article"]["doi"],
        "value": value,
        "measurement": "direct in-plane flat Poisson screen",
        "printer": scope["printer"],
        "nozzle_temperature_C": scope["nozzle_temperature_C"],
        "ambient_temperature_C": scope["ambient_temperature_C"],
        "ambient_relative_humidity_percent": (
            scope["ambient_relative_humidity_percent"]
        ),
        "process_transfer_restriction": (
            "exact-brand process-conditioned value; used as a bounded "
            "room-temperature in-plane sensitivity, not a universal "
            "out-of-plane or environmental law"
        ),
    }


def build_tds_profiles(
    poisson_reference: Path = DEFAULT_POISSON_REFERENCE,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    """Build provenance-separated isotropic and transverse TDS sensitivities."""
    e_inplane = 2580.0
    e_build = 2060.0
    nu12, poisson_evidence = _measured_in_plane_poisson(
        poisson_reference
    )
    nu13 = 0.31
    g12 = e_inplane / (2.0 * (1.0 + nu12))
    transverse_constants = {
        "E1": e_inplane,
        "E2": e_inplane,
        "E3": e_build,
        "nu12": nu12,
        "nu13": nu13,
        "nu23": nu13,
        "G12": g12,
        "G13": 780.0,
        "G23": 780.0,
    }
    admissibility = check_admissibility(transverse_constants)
    if not admissibility.valid:
        raise ValueError(
            "TDS transverse profile is inadmissible: "
            + "; ".join(admissibility.reasons)
        )
    common: Dict[str, object] = {
        "material": "Bambu PLA Basic",
        "density_tonnes_mm3": 1.24e-9,
        "screening_isotropic": {
            "E_MPa": e_inplane,
            "E_standard_deviation_MPa": 220.0,
            "nu": nu12,
            "nu_provenance": (
                "Exact-brand Bambu-X1C in-plane flat measurement at 18 C; "
                "separate process from the TDS summary"
            ),
        },
        "reported_strength_reference": {
            "overall_tensile_UTS_MPa": 35.0,
            "overall_tensile_UTS_standard_deviation_MPa": 4.0,
            "z_upright_tensile_UTS_MPa": 31.0,
            "z_upright_tensile_UTS_standard_deviation_MPa": 3.0,
        },
        "reported_flexural_reference": {
            "XY_modulus_MPa": 2750.0,
            "XY_modulus_standard_deviation_MPa": 160.0,
            "Z_modulus_MPa": 2370.0,
            "Z_modulus_standard_deviation_MPa": 150.0,
            "XY_strength_MPa": 76.0,
            "XY_strength_standard_deviation_MPa": 5.0,
            "Z_strength_MPa": 59.0,
            "Z_strength_standard_deviation_MPa": 6.0,
        },
        "reported_thermal_reference": {
            "glass_transition_C": 60.0,
            "melting_temperature_C": 160.0,
            "vicat_softening_temperature_C": 57.0,
            "heat_deflection_temperature_C_at_0_45_MPa": 57.0,
            "heat_deflection_temperature_C_at_1_8_MPa": 54.0,
        },
        "source": {
            "title": "Bambu PLA Basic Technical Data Sheet V3.0",
            "url": TDS_SOURCE,
            "accessed": "2026-07-28",
        },
        "supplemental_in_plane_poisson_evidence": poisson_evidence,
        "limitations": [
            "TDS values are summary means and standard deviations, not raw curves.",
            "The TDS does not report Poisson ratios or shear moduli; nu12 "
            "comes from a separate exact-brand Bambu-X1C process.",
            "Out-of-plane Poisson ratios and all directly measured "
            "quasistatic shear moduli remain unavailable.",
            "No temperature-dependent constitutive curve is provided.",
            "Strength references do not define a damage or fracture law.",
        ],
    }
    isotropic = {
        **common,
        "profile_id": "bambu_pla_tds_isotropic_v2",
        "intended_use": (
            "Isotropic TDS-mean elastic sensitivity; not a calibrated "
            "failure or temperature model"
        ),
        "model_class": "isotropic",
    }
    transverse = {
        **common,
        "profile_id": "bambu_pla_tds_transverse_sensitivity_v2",
        "intended_use": (
            "Transverse-isotropic elastic sensitivity using TDS XY/Z Young "
            "moduli, a separate exact-brand measured in-plane Poisson "
            "ratio, and explicitly marked remaining out-of-plane priors"
        ),
        "model_class": "transverse_isotropic",
        "screening_engineering_constants": transverse_constants,
        "engineering_constants_evidence": {
            "E1": "TDS XY Young modulus; local 1 is bead direction",
            "E2": (
                "TDS XY Young modulus tied to E1; local 2 is in-layer "
                "transverse direction"
            ),
            "E3": "TDS Z Young modulus; local 3 is build direction",
            "nu12": (
                "direct exact-brand in-plane flat measurement at 18 C / "
                "40% RH from the checksum-pinned Bambu-X1C reference"
            ),
            "nu13": "project initialization prior",
            "nu23": "tied to nu13 by transverse-isotropic hypothesis",
            "G12": "derived from E1 and nu12 by in-plane isotropy",
            "G13": "project initialization prior",
            "G23": "tied to G13 by transverse-isotropic hypothesis",
        },
        "admissibility": {
            "valid": True,
            "minimum_compliance_eigenvalue": (
                admissibility.minimum_eigenvalue
            ),
        },
    }
    return isotropic, transverse


def write_tds_profiles(
    output_dir: Path = DEFAULT_OUTPUT,
    isotropic_path: Path = DEFAULT_ISOTROPIC_PROFILE,
    transverse_path: Path = DEFAULT_TRANSVERSE_PROFILE,
) -> Tuple[Mapping[str, object], Mapping[str, object]]:
    isotropic, transverse = build_tds_profiles()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for path, profile in (
        (Path(isotropic_path), isotropic),
        (Path(transverse_path), transverse),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(profile, indent=2) + "\n", encoding="utf-8"
        )
    constants = transverse["screening_engineering_constants"]
    report = [
        "# Bambu PLA Basic TDS elastic sensitivity profiles",
        "",
        "Two provenance-separated profiles were generated from the reported "
        "TDS means plus one checksum-pinned exact-brand in-plane Poisson "
        "measurement.",
        "",
        "| Profile | E1 (MPa) | E2 (MPa) | E3 (MPa) | G12 (MPa) | "
        "G13/G23 (MPa) |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| isotropic | 2580 | 2580 | 2580 | "
            f"{2580 / (2 * (1 + constants['nu12'])):.3f} | "
            f"{2580 / (2 * (1 + constants['nu12'])):.3f} |"
        ),
        (
            f"| transverse sensitivity | {constants['E1']:.0f} | "
            f"{constants['E2']:.0f} | {constants['E3']:.0f} | "
            f"{constants['G12']:.3f} | {constants['G13']:.0f} |"
        ),
        "",
        "The transverse profile is mechanically admissible and maps local "
        "1/2/3 to bead/in-layer-transverse/build directions. E1/E2 and E3 "
        "are TDS measurements; nu12 is a separate exact-brand, "
        "process-conditioned in-plane measurement. nu13/nu23 and G13/G23 "
        "remain project priors, while G12 is derived under in-plane "
        "isotropy. This remains a sensitivity model rather than an "
        "inverse-calibrated orthotropic law.",
        "",
    ]
    (output_dir / "report.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    return isotropic, transverse


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Bambu PLA Basic TDS elastic profiles"
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--isotropic-profile", default=str(DEFAULT_ISOTROPIC_PROFILE)
    )
    parser.add_argument(
        "--transverse-profile", default=str(DEFAULT_TRANSVERSE_PROFILE)
    )
    args = parser.parse_args()
    write_tds_profiles(
        Path(args.output),
        Path(args.isotropic_profile),
        Path(args.transverse_profile),
    )
    print(
        "wrote TDS isotropic and transverse sensitivity profiles; "
        f"report: {(Path(args.output) / 'report.md').resolve()}"
    )


if __name__ == "__main__":
    main()
