from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Union

import numpy as np
import pandas as pd

from .evaluate import _effective_global_xy_poisson
from .materials import off_axis_youngs_modulus


TRUTH = {
    "E1": 2580.0,
    "E2": 2320.0,
    "E3": 1800.0,
    "nu12": 0.33,
    "nu13": 0.29,
    "nu23": 0.30,
    "G12": 910.0,
    "G13": 650.0,
    "G23": 610.0,
}


def _flexural_modulus(constants: Dict[str, float]) -> float:
    # A Timoshenko-like blend makes the synthetic flexural curve sensitive to
    # both bead-direction axial stiffness and inter-layer shear stiffness.
    return 1.0 / (0.82 / constants["E1"] + 0.18 / (3.0 * constants["G13"]))


def _heldout_modulus(constants: Dict[str, float]) -> float:
    return (
        0.42 * constants["E1"]
        + 0.18 * constants["E2"]
        + 0.18 * constants["E3"]
        + 0.12 * 2.6 * constants["G13"]
        + 0.10 * 2.6 * constants["G23"]
    )


def _curve(
    test_id: str,
    partition: str,
    test_type: str,
    direction: str,
    modulus: float,
    seed: int,
    replicate: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    strain = np.linspace(0.0, 0.012, 41)
    systematic = 1.0 + (replicate - 0.5) * 0.012
    noise = rng.normal(0.0, 0.0015, size=strain.size)
    stress = modulus * strain * systematic * (1.0 + noise)
    width, thickness, gauge = 10.0, 4.0, 50.0
    displacement = strain * gauge
    force = stress * width * thickness
    data = {
        "test_id": test_id,
        "partition": partition,
        "test_type": test_type,
        "print_direction": direction,
        "time_s": np.arange(strain.size, dtype=float),
        "force_N": force,
        "displacement_mm": displacement,
        "engineering_strain": strain,
        "engineering_stress_MPa": stress,
        "specimen_width_mm": width,
        "specimen_thickness_mm": thickness,
        "gauge_length_mm": gauge,
        "raster_angle_deg": 0.0,
        "layer_height_mm": 0.2,
        "nozzle_temperature_C": 220.0,
        "print_speed_mm_s": 50.0,
        "infill_percent": 100.0,
    }
    if test_type == "tensile":
        lateral_noise = rng.normal(0.0, 0.0015, size=strain.size)
        poisson = _effective_global_xy_poisson(TRUTH, direction)
        data["transverse_strain"] = (
            -poisson * strain * (1.0 + lateral_noise)
        )
    return pd.DataFrame(data)


def generate(output: Union[str, Path]) -> Iterable[Path]:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    definitions = []
    for partition, suffix, replicate in (
        ("calibration", "cal", 0),
        ("validation", "val", 1),
    ):
        definitions.extend(
            [
                (f"tensile_1_{suffix}", partition, "tensile", "1", TRUTH["E1"], 101 + replicate, replicate),
                (f"tensile_2_{suffix}", partition, "tensile", "2", TRUTH["E2"], 201 + replicate, replicate),
                (f"tensile_3_{suffix}", partition, "tensile", "3", TRUTH["E3"], 301 + replicate, replicate),
                (
                    f"tensile_12_45_{suffix}",
                    partition,
                    "tensile",
                    "12_45",
                    off_axis_youngs_modulus(TRUTH, "12"),
                    311 + replicate,
                    replicate,
                ),
                (
                    f"tensile_13_45_{suffix}",
                    partition,
                    "tensile",
                    "13_45",
                    off_axis_youngs_modulus(TRUTH, "13"),
                    321 + replicate,
                    replicate,
                ),
                (
                    f"tensile_23_45_{suffix}",
                    partition,
                    "tensile",
                    "23_45",
                    off_axis_youngs_modulus(TRUTH, "23"),
                    331 + replicate,
                    replicate,
                ),
                (
                    f"flexural_{suffix}",
                    partition,
                    "flexural",
                    "1",
                    _flexural_modulus(TRUTH),
                    401 + replicate,
                    replicate,
                ),
            ]
        )
    definitions.append(
        (
            "heldout_component",
            "heldout",
            "component",
            "mixed",
            _heldout_modulus(TRUTH),
            501,
            0,
        )
    )
    paths = []
    for args in definitions:
        frame = _curve(*args)
        path = output / f"{args[0]}.csv"
        frame.to_csv(path, index=False, float_format="%.10g")
        paths.append(path)
    pd.DataFrame([TRUTH]).to_csv(output / "synthetic_truth.csv", index=False)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic synthetic elastic curves")
    parser.add_argument("--output", default="data/example")
    args = parser.parse_args()
    paths = list(generate(args.output))
    print(f"wrote {len(paths)} synthetic curve files to {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
