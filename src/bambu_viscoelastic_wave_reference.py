from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    PROJECT_ROOT / "data/public/bambu-printer-pla-viscoelastic-2026"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "data/reported/bambu_pla_viscoelastic_wave_2026"
)

ARTICLE_DOI = "10.1039/D6MH00395H"
DATASET_DOI = "10.34894/ACD7BE"
MATERIAL = "Bambu PLA Basic"
REFERENCE_TEMPERATURE_C = 18.0
REFERENCE_RELATIVE_HUMIDITY_PERCENT = 40.0
MINIMUM_REDUCED_FREQUENCY_HZ = 1.0
MAXIMUM_REDUCED_FREQUENCY_HZ = 25_000.0
POISSON_RATIO = 0.3459
REPORTED_YOUNGS_MODULUS_MPA = 1_555.0
REPORTED_DENSITY_KG_M3 = 1_144.0
REPORTED_SHEAR_STORAGE_MPA = 714.7
REPORTED_SHEAR_LOSS_MPA = 26.6
WLF_ALPHA = -17.44
WLF_BETA_C = 51.6

EXPECTED_SHA1 = {
    "dma/DMA_100_25000Hz.m": (
        "6e982f1f112dea963e438dd2d111ff5ca8e0ddd7"
    ),
    "dma/DMA_Fig.m": "8fa28769ac962867af79f59673ff6f321602dd43",
    "dma/pla_bambu_1.csv": (
        "d4d3a03591168e0f7515f63a5cca13f058e1c23f"
    ),
    "dma/pla_bambu_2.csv": (
        "b32edaee71a887ffa32ac113f37584ab9d8840a4"
    ),
    "dma/pla_bambu_3.csv": (
        "5dff0b3ca4404635d776d8983258207e6751bc88"
    ),
    "poisson/PoissonsRatio_PLA.m": (
        "f8414707960530d29f4fd679a76f8eb984211d57"
    ),
    "tensile/Tensile_ISO_PLA_1.csv": (
        "cfeb152fa03245cdda6e3ba22f779f0e748ea8e5"
    ),
    "tensile/Tensile_ISO_PLA_2.csv": (
        "4b53357ad9b8693820a2755887927831a11d2c57"
    ),
    "tensile/Tensile_ISO_PLA_3.csv": (
        "e1362263ed98b41aa93720a5596ec102ff6032ec"
    ),
    "tensile/Tensile_ISO_PLA_4.csv": (
        "33fe6a5221e7a886e3a3516f8185fe4e2bdec9a9"
    ),
    "tensile/Tensile_ISO_PLA_5.csv": (
        "7d0fe567a914c982c08b9d8a0617d6736c0ff0d4"
    ),
}


def _hash(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    digest.update(Path(path).read_bytes())
    return digest.hexdigest()


def verify_source(source_dir: Path = DEFAULT_SOURCE) -> Dict[str, object]:
    source_dir = Path(source_dir)
    files = []
    for relative, expected in EXPECTED_SHA1.items():
        path = source_dir / relative
        if not path.exists():
            raise FileNotFoundError(f"missing deposited source file: {path}")
        observed = _hash(path, "sha1")
        if observed != expected:
            raise ValueError(
                f"source checksum changed for {relative}: "
                f"expected {expected}, got {observed}"
            )
        files.append(
            {
                "path": relative,
                "sha1": observed,
                "bytes": path.stat().st_size,
            }
        )
    return {
        "dataset_doi": DATASET_DOI,
        "verified_file_count": len(files),
        "files": files,
    }


def _read_dma(path: Path) -> np.ndarray:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    if len(rows) < 2 or len(rows[0]) < 43:
        raise ValueError(f"DMA file has an unexpected shape: {path}")
    numeric = []
    for row in rows[1:]:
        if len(row) < 43:
            continue
        try:
            numeric.append([float(value) for value in row[:43]])
        except ValueError:
            continue
    if not numeric:
        raise ValueError(f"DMA file contains no numeric rows: {path}")
    return np.asarray(numeric, dtype=float)


def reconstruct_reduced_master_curve(
    source_dir: Path = DEFAULT_SOURCE,
) -> list[Dict[str, object]]:
    rows: list[Dict[str, object]] = []
    for index in range(1, 4):
        path = Path(source_dir) / "dma" / f"pla_bambu_{index}.csv"
        data = _read_dma(path)
        temperature_c = data[:, 1]
        measured_frequency_hz = data[:, 3]
        storage_modulus_pa = data[:, 37]
        loss_modulus_pa = data[:, 38]
        shear_storage_pa = storage_modulus_pa / (2.0 * (1.0 + POISSON_RATIO))
        shear_loss_pa = loss_modulus_pa / (2.0 * (1.0 + POISSON_RATIO))
        positive = (
            (measured_frequency_hz > 0.0)
            & (shear_storage_pa > 0.0)
            & (shear_loss_pa > 0.0)
        )
        temperature_shift_c = (
            temperature_c[positive] - REFERENCE_TEMPERATURE_C
        )
        denominator = temperature_shift_c + WLF_BETA_C
        shift_factor = 10.0 ** (
            WLF_ALPHA * temperature_shift_c / denominator
        )
        reduced_frequency_hz = (
            measured_frequency_hz[positive] * shift_factor
        )
        storage = shear_storage_pa[positive]
        loss = shear_loss_pa[positive]
        measured = measured_frequency_hz[positive]
        temperatures = temperature_c[positive]
        selected = (
            (reduced_frequency_hz >= MINIMUM_REDUCED_FREQUENCY_HZ)
            & (reduced_frequency_hz <= MAXIMUM_REDUCED_FREQUENCY_HZ)
        )
        for source_row, values in enumerate(
            zip(
                temperatures[selected],
                measured[selected],
                reduced_frequency_hz[selected],
                storage[selected],
                loss[selected],
            ),
            start=1,
        ):
            temperature, frequency, reduced, g_storage, g_loss = values
            rows.append(
                {
                    "source_file": path.name,
                    "selected_row_number": source_row,
                    "measured_temperature_C": float(temperature),
                    "measured_frequency_Hz": float(frequency),
                    "reduced_frequency_at_18C_Hz": float(reduced),
                    "shear_storage_modulus_MPa": float(g_storage / 1.0e6),
                    "shear_loss_modulus_MPa": float(g_loss / 1.0e6),
                    "loss_factor": float(g_loss / g_storage),
                }
            )
    return sorted(rows, key=lambda row: row["reduced_frequency_at_18C_Hz"])


def _stats(rows: Sequence[Mapping[str, object]], field: str) -> Dict[str, float]:
    values = np.asarray([float(row[field]) for row in rows], dtype=float)
    return {
        "mean": float(np.mean(values)),
        "sample_standard_deviation": float(np.std(values, ddof=1)),
        "median": float(np.median(values)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }


def build_reference(
    source_dir: Path = DEFAULT_SOURCE,
) -> Dict[str, object]:
    source = verify_source(source_dir)
    curve = reconstruct_reduced_master_curve(source_dir)
    if len(curve) != 85:
        raise ValueError(
            f"expected 85 reduced-frequency rows, found {len(curve)}"
        )
    storage_stats = _stats(curve, "shear_storage_modulus_MPa")
    loss_stats = _stats(curve, "shear_loss_modulus_MPa")
    loss_factor = (
        REPORTED_SHEAR_LOSS_MPA / REPORTED_SHEAR_STORAGE_MPA
    )
    dynamic_equivalent_e = (
        2.0 * REPORTED_SHEAR_STORAGE_MPA * (1.0 + POISSON_RATIO)
    )
    static_inferred_g = (
        REPORTED_YOUNGS_MODULUS_MPA / (2.0 * (1.0 + POISSON_RATIO))
    )
    storage_relative_error = abs(
        storage_stats["mean"] / REPORTED_SHEAR_STORAGE_MPA - 1.0
    )
    loss_relative_error = abs(
        loss_stats["mean"] / REPORTED_SHEAR_LOSS_MPA - 1.0
    )
    quality_gate = (
        source["verified_file_count"] == len(EXPECTED_SHA1)
        and storage_relative_error <= 0.01
        and loss_relative_error <= 0.15
        and loss_stats["minimum"]
        <= REPORTED_SHEAR_LOSS_MPA
        <= loss_stats["maximum"]
    )
    return {
        "reference_id": "bambu_pla_viscoelastic_wave_2026_v1",
        "material": MATERIAL,
        "article": {
            "title": (
                "Predictive wave engineering in polymer phononic "
                "materials via viscoelastic-geometric coupling"
            ),
            "doi": ARTICLE_DOI,
            "publisher_url": (
                "https://pubs.rsc.org/en/content/articlelanding/2026/mh/"
                "d6mh00395h"
            ),
            "license": "CC BY 3.0",
        },
        "dataset": source,
        "identity_basis": (
            "The article's PLA_B manufacturer reference is the Bambu PLA "
            "Basic TDS; samples were printed on a Bambu Lab X1 Carbon."
        ),
        "print_and_test_scope": {
            "printer": "Bambu Lab X1 Carbon",
            "filament_diameter_mm": 1.75,
            "nozzle_diameter_mm": 0.4,
            "nozzle_temperature_C": 200.0,
            "bed_temperature_C": 60.0,
            "print_speed_mm_s": 40.0,
            "infill_percent": 100.0,
            "infill_pattern": "line",
            "print_direction_deg": 0.0,
            "ambient_temperature_C": REFERENCE_TEMPERATURE_C,
            "ambient_relative_humidity_percent": (
                REFERENCE_RELATIVE_HUMIDITY_PERCENT
            ),
        },
        "reported_static_measurements": {
            "youngs_modulus_MPa": REPORTED_YOUNGS_MODULUS_MPA,
            "poisson_ratio": POISSON_RATIO,
            "printed_density_kg_m3": REPORTED_DENSITY_KG_M3,
            "inferred_quasistatic_shear_modulus_MPa": static_inferred_g,
            "inference_note": (
                "The quasistatic shear value is inferred from measured E "
                "and nu under isotropy; it is not a direct shear test."
            ),
        },
        "reported_dynamic_measurements": {
            "reference_temperature_C": REFERENCE_TEMPERATURE_C,
            "minimum_reduced_frequency_Hz": MINIMUM_REDUCED_FREQUENCY_HZ,
            "maximum_reduced_frequency_Hz": MAXIMUM_REDUCED_FREQUENCY_HZ,
            "shear_storage_modulus_MPa": REPORTED_SHEAR_STORAGE_MPA,
            "shear_loss_modulus_MPa": REPORTED_SHEAR_LOSS_MPA,
            "loss_factor": loss_factor,
            "small_damping_ratio_approximation": loss_factor / 2.0,
            "storage_equivalent_youngs_modulus_MPa": dynamic_equivalent_e,
            "dynamic_shear_wave_speed_m_s": math.sqrt(
                REPORTED_SHEAR_STORAGE_MPA
                * 1.0e6
                / REPORTED_DENSITY_KG_M3
            ),
        },
        "raw_reconstruction": {
            "method": (
                "Reimplemented deposited DMA_Fig.m: convert E'/E'' to "
                "G'/G'' with nu=0.3459, apply WLF alpha=-17.44 and "
                "beta=51.6 C at 18 C, retain 1-25000 Hz."
            ),
            "selected_row_count": len(curve),
            "manual_outliers_removed": 0,
            "storage_modulus_MPa": storage_stats,
            "loss_modulus_MPa": loss_stats,
            "reported_storage_mean_relative_error": storage_relative_error,
            "reported_loss_mean_relative_error": loss_relative_error,
            "caveat": (
                "The deposited plotting script selects outliers interactively, "
                "so its exact cleaned mean is not reproducible without the "
                "authors' clicks. The paper's representative values are "
                "retained separately from the no-click reconstruction."
            ),
        },
        "quality_gate_passed": quality_gate,
        "admitted_uses": [
            "small-strain in-phase storage-stiffness screen at 18 C",
            "1 Hz to 25 kHz reduced-frequency wave screen",
            "loss-factor damping sensitivity",
            "in-plane flat Poisson screen",
            "printed-density wave or modal screen",
        ],
        "hard_exclusions": [
            "quasistatic G12/G13/G23 measurement claim",
            "out-of-plane Poisson-ratio claim",
            "large-strain or failure prediction",
            "creep, fatigue, moisture, aging, or arbitrary-temperature law",
            "direct use of the 18 C master curve at -80 C or near Tg",
        ],
    }


def _write_csv(
    path: Path, rows: Sequence[Mapping[str, object]]
) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_reference(
    source_dir: Path = DEFAULT_SOURCE,
    output_dir: Path = DEFAULT_OUTPUT,
) -> Dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reference = build_reference(source_dir)
    curve = reconstruct_reduced_master_curve(source_dir)
    _write_csv(output_dir / "dma_master_curve_18C.csv", curve)
    static = reference["reported_static_measurements"]
    dynamic = reference["reported_dynamic_measurements"]
    _write_csv(
        output_dir / "reported_measurements.csv",
        [
            {
                "property": "youngs_modulus",
                "value": static["youngs_modulus_MPa"],
                "units": "MPa",
                "measurement_or_derivation": "measured",
            },
            {
                "property": "poisson_ratio",
                "value": static["poisson_ratio"],
                "units": "dimensionless",
                "measurement_or_derivation": "measured",
            },
            {
                "property": "printed_density",
                "value": static["printed_density_kg_m3"],
                "units": "kg/m3",
                "measurement_or_derivation": "measured",
            },
            {
                "property": "shear_storage_modulus",
                "value": dynamic["shear_storage_modulus_MPa"],
                "units": "MPa",
                "measurement_or_derivation": "reported DMA representative",
            },
            {
                "property": "shear_loss_modulus",
                "value": dynamic["shear_loss_modulus_MPa"],
                "units": "MPa",
                "measurement_or_derivation": "reported DMA representative",
            },
            {
                "property": "loss_factor",
                "value": dynamic["loss_factor"],
                "units": "dimensionless",
                "measurement_or_derivation": "G_double_prime/G_prime",
            },
        ],
    )
    (output_dir / "provenance.json").write_text(
        json.dumps(reference, indent=2) + "\n", encoding="utf-8"
    )
    equivalent_e = float(dynamic["storage_equivalent_youngs_modulus_MPa"])
    (output_dir / "storage_proxy.inp").write_text(
        (
            "*MATERIAL,NAME=BAMBU_PLA_18C_WAVE_STORAGE_PROXY\n"
            "*DENSITY\n"
            f"{REPORTED_DENSITY_KG_M3 * 1.0e-12:.12g}\n"
            "*ELASTIC\n"
            f"{equivalent_e:.12g},{POISSON_RATIO:.12g}\n"
        ),
        encoding="utf-8",
    )
    report = f"""# Bambu PLA Basic broadband viscoelastic reference

The exact-brand source reports a printed density of
{REPORTED_DENSITY_KG_M3:.0f} kg/m3, Poisson ratio {POISSON_RATIO:.4f},
and, at {REFERENCE_TEMPERATURE_C:.0f} C over
{MINIMUM_REDUCED_FREQUENCY_HZ:.0f}-{MAXIMUM_REDUCED_FREQUENCY_HZ:.0f} Hz,
representative G'={REPORTED_SHEAR_STORAGE_MPA:.1f} MPa and
G''={REPORTED_SHEAR_LOSS_MPA:.1f} MPa.

The checksum-pinned raw DMA files produce {len(curve)} reduced-frequency
points. Their no-click mean G' is
{reference['raw_reconstruction']['storage_modulus_MPa']['mean']:.3f} MPa
and mean G'' is
{reference['raw_reconstruction']['loss_modulus_MPa']['mean']:.3f} MPa.
The storage mean is within 1% of the reported representative value. The
deposited plotting script removes outliers by interactive clicking, so the
paper's cleaned loss value cannot be reproduced exactly; both versions are
retained rather than silently choosing one.

Quality gate: **{'passed' if reference['quality_gate_passed'] else 'failed'}**.

The CalculiX card is an in-phase isotropic storage proxy derived as
E'=2G'(1+nu). It is restricted to small-strain wave/modal sensitivity at
18 C. It is not a static shear, failure, creep, fatigue, moisture, aging, or
arbitrary-temperature law.
"""
    (output_dir / "quality_report.md").write_text(report, encoding="utf-8")
    return reference


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import the exact-brand broadband viscoelastic reference"
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = write_reference(args.source, args.output)
    print(
        f"viscoelastic wave reference: "
        f"{result['raw_reconstruction']['selected_row_count']} raw points; "
        f"quality gate={'passed' if result['quality_gate_passed'] else 'failed'}; "
        f"output={args.output.resolve()}"
    )


if __name__ == "__main__":
    main()
