from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict

from .bambu_dma_reference import DEFAULT_OUTPUT as DMA_REFERENCE
from .bambu_tds_profile import DEFAULT_TRANSVERSE_PROFILE
from .materials import check_admissibility, material_card


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data/reported/bambu_pla_dma_30C_1Hz_sensitivity.json"
)
DEFAULT_CARD = (
    PROJECT_ROOT
    / "data/reported/bambu_pla_dma_30C_1Hz_sensitivity.inp"
)


def infer_in_plane_shear_modulus(
    E1_MPa: float,
    E2_MPa: float,
    E45_MPa: float,
    nu12: float,
) -> float:
    """Infer G12 from the plane-stress off-axis compliance at 45 degrees."""
    s11 = 1.0 / E1_MPa
    s22 = 1.0 / E2_MPa
    s12 = -nu12 / E1_MPa
    inverse_g12 = 4.0 / E45_MPa - s11 - s22 - 2.0 * s12
    if inverse_g12 <= 0.0:
        raise ValueError("DMA points imply a non-positive G12")
    return 1.0 / inverse_g12


def build_dynamic_profile(
    dma_reference: Path = DMA_REFERENCE,
    transverse_profile: Path = DEFAULT_TRANSVERSE_PROFILE,
    nozzle_temperature_C: int = 220,
) -> Dict[str, object]:
    with (Path(dma_reference) / "dma_30C_1Hz.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if int(float(row["nozzle_temperature_C"]))
            == nozzle_temperature_C
        ]
    if len(rows) != 3:
        raise ValueError(
            f"expected three {nozzle_temperature_C} C DMA orientation rows"
        )
    storage = {
        int(float(row["raster_orientation_deg"])): (
            float(row["storage_modulus_GPa"]) * 1000.0
        )
        for row in rows
    }
    if set(storage) != {0, 45, 90}:
        raise ValueError("DMA orientations must be 0, 45, and 90 degrees")
    static = json.loads(Path(transverse_profile).read_text(encoding="utf-8"))
    static_constants = {
        key: float(value)
        for key, value in static[
            "screening_engineering_constants"
        ].items()
    }
    nu12 = static_constants["nu12"]
    g12 = infer_in_plane_shear_modulus(
        storage[0], storage[90], storage[45], nu12
    )
    in_plane_scale = (
        (storage[0] + storage[90])
        / (static_constants["E1"] + static_constants["E2"])
    )
    constants = {
        "E1": storage[0],
        "E2": storage[90],
        "E3": static_constants["E3"] * in_plane_scale,
        "nu12": nu12,
        "nu13": static_constants["nu13"],
        "nu23": static_constants["nu23"],
        "G12": g12,
        "G13": static_constants["G13"] * in_plane_scale,
        "G23": static_constants["G23"] * in_plane_scale,
    }
    admissibility = check_admissibility(constants)
    if not admissibility.valid:
        raise ValueError(
            "dynamic storage profile is inadmissible: "
            + "; ".join(admissibility.reasons)
        )
    minimum_dma_tg = min(
        float(row["DMA_tan_delta_peak_Tg_C"]) for row in rows
    )
    crossplot_rows_passed = sum(
        str(row["passes_independent_crossplot"]).lower() == "true"
        for row in rows
    )
    return {
        "profile_id": (
            "bambu_pla_dma_30C_1Hz_"
            f"print{nozzle_temperature_C}C_sensitivity_v1"
        ),
        "material": "Bambu PLA Basic",
        "model_class": "orthotropic_storage_sensitivity",
        "density_tonnes_mm3": float(static["density_tonnes_mm3"]),
        "screening_isotropic": {
            "E_MPa": (storage[0] + storage[90]) / 2.0,
            "E_standard_deviation_MPa": 0.0,
            "nu": nu12,
        },
        "screening_engineering_constants": constants,
        "reported_strength_reference": static[
            "reported_strength_reference"
        ],
        "dynamic_reference": {
            "measurement_temperature_C": 30.0,
            "frequency_Hz": 1.0,
            "nozzle_temperature_C": float(nozzle_temperature_C),
            "Eprime_0deg_MPa": storage[0],
            "Eprime_45deg_MPa": storage[45],
            "Eprime_90deg_MPa": storage[90],
            "minimum_DMA_tan_delta_peak_Tg_C": minimum_dma_tg,
            "G12_inference": (
                "plane-stress 45-degree compliance using the separate "
                "exact-brand measured in-plane nu12"
            ),
            "out_of_plane_scaling": in_plane_scale,
            "independent_crossplot_rows_passed": crossplot_rows_passed,
            "independent_crossplot_rows_total": len(rows),
        },
        "admissibility": {
            "valid": admissibility.valid,
            "minimum_compliance_eigenvalue": (
                admissibility.minimum_eigenvalue
            ),
        },
        "intended_use": (
            "in-phase 30 C, 1 Hz storage-stiffness sensitivity only"
        ),
        "limitations": [
            "E1/E2/E45 are digitized plotted summary points, not raw sweeps.",
            "nu12 is a separate exact-brand process measurement; nu13, "
            "nu23 and out-of-plane scaling remain priors.",
            "G12 is inferred from E45 and measured nu12; it is not a direct "
            "shear measurement.",
            "No loss modulus or damping is available.",
            "Do not use as a service-temperature, creep, Prony, fatigue, "
            "resonance, or failure law.",
            "The minimum DMA Tg is an exclusion boundary, not an allowable "
            "service temperature.",
        ],
        "source_reference": str(Path(dma_reference).resolve()),
    }


def write_dynamic_profile(
    output_path: Path = DEFAULT_OUTPUT,
    card_path: Path = DEFAULT_CARD,
    dma_reference: Path = DMA_REFERENCE,
    transverse_profile: Path = DEFAULT_TRANSVERSE_PROFILE,
    nozzle_temperature_C: int = 220,
) -> Dict[str, object]:
    profile = build_dynamic_profile(
        dma_reference,
        transverse_profile,
        nozzle_temperature_C,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(profile, indent=2) + "\n", encoding="utf-8"
    )
    Path(card_path).write_text(
        material_card(
            profile["screening_engineering_constants"],
            name="BAMBU_PLA_DMA_30C_1HZ_SENSITIVITY",
        ),
        encoding="utf-8",
    )
    return profile


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the Bambu PLA 30 C, 1 Hz storage sensitivity"
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--card", default=str(DEFAULT_CARD))
    parser.add_argument("--dma-reference", default=str(DMA_REFERENCE))
    parser.add_argument(
        "--transverse-profile", default=str(DEFAULT_TRANSVERSE_PROFILE)
    )
    parser.add_argument("--nozzle-temperature", type=int, default=220)
    args = parser.parse_args()
    profile = write_dynamic_profile(
        Path(args.output),
        Path(args.card),
        Path(args.dma_reference),
        Path(args.transverse_profile),
        args.nozzle_temperature,
    )
    constants = profile["screening_engineering_constants"]
    print(
        f"DMA storage profile: E1/E2/G12="
        f"{constants['E1']:.1f}/{constants['E2']:.1f}/"
        f"{constants['G12']:.1f} MPa; output: {Path(args.output).resolve()}"
    )


if __name__ == "__main__":
    main()
