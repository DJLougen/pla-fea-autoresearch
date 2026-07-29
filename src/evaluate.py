from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .calculix import CalculixRunner, ORIENTATIONS
from .config import load_config
from .materials import (
    PRIORS,
    check_admissibility,
    complexity,
    compliance_matrix,
    initial_vector,
    material_card,
    off_axis_youngs_modulus,
    parameter_names,
    vector_to_constants,
)
from .metrics import CurveMetrics, aggregate_metrics, curve_metrics
from .validation import assert_heldout_not_calibration, load_manifest


PHYSICAL_INVALIDITY_PENALTY = 1.0e6
SOLVER_FAILURE_PENALTY = 1.0e5
TEMPLATE_NAMES = {
    ("tensile", "1"): "tensile_x.inp.template",
    ("tensile", "2"): "tensile_y.inp.template",
    ("tensile", "3"): "tensile_z.inp.template",
    ("tensile", "12_45"): "tensile_x.inp.template",
    ("tensile", "13_45"): "tensile_x.inp.template",
    ("tensile", "23_45"): "tensile_x.inp.template",
    ("flexural", "1"): "flexural.inp.template",
    ("component", "mixed"): "heldout_component.inp.template",
}


@dataclass
class EvaluationResult:
    score: float
    partition: str
    aggregate: Dict[str, float]
    per_test: Dict[str, Dict[str, float]]
    solver_failures: int
    statuses: Dict[str, str]
    constants: Dict[str, float]
    invalid_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _effective_modulus(constants: Mapping[str, float], test_type: str, direction: str) -> float:
    if test_type == "tensile":
        if direction.endswith("_45"):
            return off_axis_youngs_modulus(
                constants, direction.split("_", 1)[0], 45.0
            )
        return float(constants[f"E{direction}"])
    if test_type == "flexural":
        return float(1.0 / (0.82 / constants["E1"] + 0.18 / (3.0 * constants["G13"])))
    if test_type == "component":
        return float(
            0.42 * constants["E1"]
            + 0.18 * constants["E2"]
            + 0.18 * constants["E3"]
            + 0.12 * 2.6 * constants["G13"]
            + 0.10 * 2.6 * constants["G23"]
        )
    raise ValueError(f"unsupported test type: {test_type}")


def _curve_columns(frame: pd.DataFrame, test_type: str) -> Tuple[np.ndarray, np.ndarray, str]:
    if test_type == "tensile":
        return (
            frame["engineering_strain"].to_numpy(float),
            frame["engineering_stress_MPa"].to_numpy(float),
            "stress_strain",
        )
    if {"displacement_mm", "force_N"}.issubset(frame.columns):
        return (
            frame["displacement_mm"].to_numpy(float),
            frame["force_N"].to_numpy(float),
            "force_displacement",
        )
    return (
        frame["engineering_strain"].to_numpy(float),
        frame["engineering_stress_MPa"].to_numpy(float),
        "stress_strain",
    )


def _mock_curve(
    constants: Mapping[str, float],
    frame: pd.DataFrame,
    test_type: str,
    direction: str,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    x, _, kind = _curve_columns(frame, test_type)
    modulus = _effective_modulus(constants, test_type, direction)
    if kind == "stress_strain":
        lateral = (
            -_effective_global_xy_poisson(constants, direction) * x
            if test_type == "tensile"
            else None
        )
        return x.copy(), modulus * x, lateral
    width = float(frame["specimen_width_mm"].dropna().iloc[0])
    thickness = float(frame["specimen_thickness_mm"].dropna().iloc[0])
    gauge = float(frame["gauge_length_mm"].dropna().iloc[0])
    stiffness = modulus * width * thickness / gauge
    return x.copy(), stiffness * x, None


def _orientation_matrix(direction: str) -> np.ndarray:
    values = np.fromstring(ORIENTATIONS[direction], sep=",")
    if values.shape != (6,):
        raise ValueError(f"invalid CalculiX orientation for {direction!r}")
    axis_1 = values[:3]
    axis_1 = axis_1 / np.linalg.norm(axis_1)
    axis_2 = values[3:] - np.dot(values[3:], axis_1) * axis_1
    axis_2 = axis_2 / np.linalg.norm(axis_2)
    axis_3 = np.cross(axis_1, axis_2)
    return np.column_stack((axis_1, axis_2, axis_3))


def _effective_global_xy_poisson(
    constants: Mapping[str, float],
    direction: str,
) -> float:
    """Poisson ratio for global-X loading and global-Y lateral strain."""
    axes = _orientation_matrix(direction)
    global_stress = np.zeros((3, 3), dtype=float)
    global_stress[0, 0] = 1.0
    local_stress = axes.T @ global_stress @ axes
    stress_vector = np.asarray(
        [
            local_stress[0, 0],
            local_stress[1, 1],
            local_stress[2, 2],
            local_stress[1, 2],
            local_stress[0, 2],
            local_stress[0, 1],
        ]
    )
    engineering_strain = compliance_matrix(constants) @ stress_vector
    local_strain = np.asarray(
        [
            [
                engineering_strain[0],
                engineering_strain[5] / 2.0,
                engineering_strain[4] / 2.0,
            ],
            [
                engineering_strain[5] / 2.0,
                engineering_strain[1],
                engineering_strain[3] / 2.0,
            ],
            [
                engineering_strain[4] / 2.0,
                engineering_strain[3] / 2.0,
                engineering_strain[2],
            ],
        ]
    )
    global_strain = axes @ local_strain @ axes.T
    if global_strain[0, 0] <= 0.0:
        raise ValueError("effective axial compliance is not positive")
    return float(-global_strain[1, 1] / global_strain[0, 0])


class FixedEvaluator:
    """The immutable scoring path used by both optimization and reporting."""

    def __init__(self, config: Mapping[str, object]):
        self.config = config
        manifest_path = Path(config["data"]["manifest"])
        self.manifest = load_manifest(manifest_path)
        assert_heldout_not_calibration(self.manifest)
        self.processed_dir = Path(config["data"]["processed_dir"])
        self.parameter_ties = config["experiment"].get("parameter_ties")
        solver = config["solver"]
        self.mode = solver["mode"]
        self.runner = None
        if self.mode == "calculix":
            self.runner = CalculixRunner(
                Path(config["output"]["runs_dir"]),
                executable=solver.get("executable"),
                timeout_seconds=float(solver["timeout_seconds"]),
                keep_successful=bool(solver.get("keep_successful_runs", False)),
            )

    def _simulate(
        self,
        constants: Mapping[str, float],
        row: pd.Series,
        frame: pd.DataFrame,
        candidate_tag: str,
    ) -> Tuple[
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        str,
    ]:
        test_type, direction = str(row["test_type"]), str(row["direction"])
        if self.mode == "mock":
            x, y, lateral = _mock_curve(
                constants, frame, test_type, direction
            )
            return x, y, lateral, "mock_ok"
        template_name = TEMPLATE_NAMES.get((test_type, direction))
        if template_name is None:
            return None, None, None, "template_missing"
        measured_x, _, kind = _curve_columns(frame, test_type)
        mesh_level = str(self.config["experiment"].get("mesh_level", "coarse"))
        level_template = Path("meshes") / mesh_level / template_name
        template_path = level_template if level_template.exists() else Path("meshes") / template_name
        if kind == "stress_strain":
            gauge = float(frame["gauge_length_mm"].dropna().iloc[0])
            maximum_displacement = float(np.max(np.abs(measured_x))) * gauge
        else:
            maximum_displacement = float(np.max(np.abs(measured_x)))
        result = self.runner.run(
            template_path,
            {
                "ORIENTATION": ORIENTATIONS[direction],
                "MATERIAL_CARD": material_card(constants),
                "MAX_DISPLACEMENT": maximum_displacement,
            },
            run_id=f"{candidate_tag}-{row['test_id']}",
        )
        if not result.ok:
            return None, None, None, result.status
        displacement = np.abs(result.curve["displacement"])
        force = np.abs(result.curve["reaction_force"])
        if len(displacement) == 1:
            # CalculiX intentionally solves a purely linear *STATIC step in one
            # increment. Reconstruct the exact linear history from that physics
            # endpoint so the fixed full-curve metric still receives a curve.
            endpoint = float(displacement[-1])
            if endpoint <= 0:
                return None, None, None, "zero_displacement"
            displacement = np.linspace(0.0, endpoint, 21)
            force = float(force[-1]) * displacement / endpoint
        if kind == "force_displacement":
            return displacement, force, None, "ok"
        width = float(frame["specimen_width_mm"].dropna().iloc[0])
        thickness = float(frame["specimen_thickness_mm"].dropna().iloc[0])
        gauge = float(frame["gauge_length_mm"].dropna().iloc[0])
        axial_strain = displacement / gauge
        lateral_strain = None
        if test_type == "tensile":
            strain_field = result.curve.get("frd_tostrain")
            if strain_field is None:
                strain_field = result.curve.get("frd_strain")
            if (
                strain_field is not None
                and strain_field.ndim == 2
                and strain_field.shape[1] >= 2
            ):
                finite = strain_field[
                    np.isfinite(strain_field[:, :2]).all(axis=1)
                ]
                if len(finite):
                    lateral_endpoint = float(np.mean(finite[:, 1]))
                    lateral_strain = np.linspace(
                        0.0, lateral_endpoint, len(axial_strain)
                    )
        return (
            axial_strain,
            force / (width * thickness),
            lateral_strain,
            "ok",
        )

    def evaluate_constants(
        self,
        constants: Mapping[str, float],
        partition: str,
        candidate_tag: str = "candidate",
    ) -> EvaluationResult:
        status = check_admissibility(constants)
        if not status.valid:
            return EvaluationResult(
                PHYSICAL_INVALIDITY_PENALTY,
                partition,
                {},
                {},
                0,
                {},
                dict(constants),
                "; ".join(status.reasons),
            )
        selected = self.manifest.loc[self.manifest["partition"] == partition]
        if selected.empty:
            raise ValueError(f"no tests in partition {partition}")
        per_test_metrics: Dict[str, CurveMetrics] = {}
        statuses: Dict[str, str] = {}
        failures = 0
        for _, row in selected.iterrows():
            test_id = str(row["test_id"])
            frame = pd.read_csv(self.processed_dir / str(row["file"]))
            measured_x, measured_y, _ = _curve_columns(frame, str(row["test_type"]))
            (
                simulated_x,
                simulated_y,
                simulated_lateral,
                solver_status,
            ) = self._simulate(
                constants, row, frame, candidate_tag
            )
            statuses[test_id] = solver_status
            if simulated_x is None:
                failures += 1
                continue
            try:
                measured_lateral = None
                if (
                    str(row["test_type"]) == "tensile"
                    and "transverse_strain" in frame
                ):
                    lateral_values = pd.to_numeric(
                        frame["transverse_strain"], errors="coerce"
                    ).to_numpy(float)
                    if np.isfinite(lateral_values).sum() >= 2:
                        measured_lateral = lateral_values
                per_test_metrics[test_id] = curve_metrics(
                    measured_x,
                    measured_y,
                    simulated_x,
                    simulated_y,
                    measured_lateral_strain=measured_lateral,
                    simulated_lateral_strain=simulated_lateral,
                )
            except ValueError:
                failures += 1
                statuses[test_id] = "metric_failed"
        if per_test_metrics:
            weighting = self.config["experiment"].get("curve_weighting", "equal_test")
            weights = None
            if weighting == "equal_type":
                counts = selected["test_type"].value_counts().to_dict()
                test_types = dict(zip(selected["test_id"].astype(str), selected["test_type"]))
                weights = {
                    test_id: 1.0 / counts[test_types[test_id]]
                    for test_id in per_test_metrics
                }
            elif weighting == "custom":
                specified = self.config["experiment"].get("test_weights", {})
                weights = {test_id: float(specified.get(test_id, 1.0)) for test_id in per_test_metrics}
            elif weighting != "equal_test":
                raise ValueError(f"unknown curve_weighting: {weighting}")
            aggregate = aggregate_metrics(per_test_metrics, weights)
            base_score = aggregate["total"]
        else:
            aggregate = {}
            base_score = 0.0
        score = base_score + failures * SOLVER_FAILURE_PENALTY
        return EvaluationResult(
            float(score),
            partition,
            aggregate,
            {name: metric.to_dict() for name, metric in per_test_metrics.items()},
            failures,
            statuses,
            dict(constants),
        )

    def evaluate_vector(
        self, vector: Sequence[float], partition: str = "calibration", candidate_tag: str = "candidate"
    ) -> EvaluationResult:
        experiment = self.config["experiment"]
        constants = vector_to_constants(
            experiment["model_class"],
            vector,
            bool(experiment.get("fit_poisson", True)),
            experiment.get("parameter_ties"),
        )
        result = self.evaluate_constants(constants, partition, candidate_tag)
        if partition == "calibration" and result.score < SOLVER_FAILURE_PENALTY:
            prior_scale = np.mean(
                [
                    ((constants[key] - PRIORS[key]) / max(PRIORS[key], 1.0)) ** 2
                    for key in ("E1", "E2", "E3", "G12", "G13", "G23")
                ]
            )
            strength = float(experiment.get("regularization_strength", 0.0))
            result.score += strength * float(prior_scale)
        return result


def evaluate_config(
    config: Mapping[str, object],
    parameters: Optional[Mapping[str, float]] = None,
) -> Dict[str, object]:
    experiment = config["experiment"]
    names = parameter_names(
        experiment["model_class"],
        bool(experiment.get("fit_poisson", True)),
        experiment.get("parameter_ties"),
    )
    vector = initial_vector(
        experiment["model_class"],
        bool(experiment.get("fit_poisson", True)),
        experiment.get("parameter_ties"),
        experiment.get("initial_parameters"),
    )
    if parameters:
        vector = np.array(
            [float(parameters.get(name, vector[index])) for index, name in enumerate(names)]
        )
    evaluator = FixedEvaluator(config)
    return {
        partition: evaluator.evaluate_vector(vector, partition, "evaluate").to_dict()
        for partition in ("calibration", "validation", "heldout")
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a material candidate with fixed metrics")
    parser.add_argument("--config", required=True)
    parser.add_argument("--parameters", help="JSON mapping of model parameters")
    parser.add_argument("--output")
    args = parser.parse_args()
    config = load_config(args.config)
    parameters = json.loads(args.parameters) if args.parameters else None
    result = evaluate_config(config, parameters)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
