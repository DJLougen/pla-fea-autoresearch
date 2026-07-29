from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np


DENSITY_TONNES_MM3 = 1.24e-9
PRIORS = {
    "E1": 2580.0,
    "E2": 2580.0,
    "E3": 2060.0,
    "nu12": 0.33,
    "nu13": 0.31,
    "nu23": 0.31,
    "G12": 950.0,
    "G13": 780.0,
    "G23": 780.0,
}
DEFAULT_BOUNDS = {
    "E": (500.0, 6000.0),
    "nu": (0.05, 0.48),
    "E_inplane": (500.0, 6000.0),
    "E3": (300.0, 5000.0),
    "nu12": (0.05, 0.48),
    "nu13": (0.02, 0.47),
    "nu23": (0.02, 0.47),
    "G13": (100.0, 2500.0),
    "E1": (500.0, 6000.0),
    "E2": (400.0, 6000.0),
    "G12": (100.0, 2500.0),
    "G23": (100.0, 2500.0),
}


@dataclass(frozen=True)
class Admissibility:
    valid: bool
    reasons: Tuple[str, ...]
    minimum_eigenvalue: float


def parameter_names(
    model_class: str,
    fit_poisson: bool = True,
    parameter_ties: Mapping[str, str] = None,
) -> List[str]:
    if model_class == "isotropic":
        names = ["E", "nu"] if fit_poisson else ["E"]
    elif model_class == "transverse_isotropic":
        names = ["E_inplane", "E3", "G13"]
        names += ["nu12", "nu13"] if fit_poisson else []
    elif model_class == "orthotropic":
        names = ["E1", "E2", "E3", "G12", "G13", "G23"]
        names += ["nu12", "nu13", "nu23"] if fit_poisson else []
    else:
        raise ValueError(f"unknown model class: {model_class}")
    ties = parameter_ties or {}
    if ties and model_class != "orthotropic":
        raise ValueError("custom parameter_ties are supported for orthotropic models")
    unknown = (set(ties) | set(ties.values())) - set(names)
    if unknown:
        raise ValueError(f"parameter_ties reference unknown fitted names: {sorted(unknown)}")
    for target, source in ties.items():
        if target == source or source in ties:
            raise ValueError("parameter_ties must be direct, acyclic target-to-source mappings")
    return [name for name in names if name not in ties]


def parameter_bounds(
    model_class: str,
    fit_poisson: bool = True,
    parameter_ties: Mapping[str, str] = None,
    overrides: Mapping[str, Sequence[float]] = None,
) -> List[Tuple[float, float]]:
    overrides = overrides or {}
    result = []
    for name in parameter_names(model_class, fit_poisson, parameter_ties):
        values = overrides.get(name, DEFAULT_BOUNDS[name])
        if len(values) != 2 or float(values[0]) >= float(values[1]):
            raise ValueError(f"invalid bounds for {name}: {values}")
        result.append((float(values[0]), float(values[1])))
    return result


def initial_vector(
    model_class: str,
    fit_poisson: bool = True,
    parameter_ties: Mapping[str, str] = None,
    overrides: Mapping[str, float] = None,
) -> np.ndarray:
    seeds = {
        "E": PRIORS["E1"],
        "nu": PRIORS["nu12"],
        "E_inplane": PRIORS["E1"],
        **PRIORS,
    }
    seeds.update(overrides or {})
    return np.array(
        [
            seeds[name]
            for name in parameter_names(model_class, fit_poisson, parameter_ties)
        ],
        dtype=float,
    )


def expand_parameters(
    model_class: str,
    values: Mapping[str, float],
    fit_poisson: bool = True,
    parameter_ties: Mapping[str, str] = None,
) -> Dict[str, float]:
    if model_class == "isotropic":
        young = float(values["E"])
        nu = float(values.get("nu", PRIORS["nu12"]))
        shear = young / (2.0 * (1.0 + nu))
        return {
            "E1": young,
            "E2": young,
            "E3": young,
            "nu12": nu,
            "nu13": nu,
            "nu23": nu,
            "G12": shear,
            "G13": shear,
            "G23": shear,
        }
    if model_class == "transverse_isotropic":
        e_plane = float(values["E_inplane"])
        nu12 = float(values.get("nu12", PRIORS["nu12"]))
        nu13 = float(values.get("nu13", PRIORS["nu13"]))
        return {
            "E1": e_plane,
            "E2": e_plane,
            "E3": float(values["E3"]),
            "nu12": nu12,
            "nu13": nu13,
            "nu23": nu13,
            "G12": e_plane / (2.0 * (1.0 + nu12)),
            "G13": float(values["G13"]),
            "G23": float(values["G13"]),
        }
    if model_class == "orthotropic":
        constants = {
            key: float(values.get(key, PRIORS[key]))
            for key in (
                "E1",
                "E2",
                "E3",
                "nu12",
                "nu13",
                "nu23",
                "G12",
                "G13",
                "G23",
            )
        }
        for target, source in (parameter_ties or {}).items():
            constants[target] = constants[source]
        return constants
    raise ValueError(f"unknown model class: {model_class}")


def vector_to_constants(
    model_class: str,
    vector: Sequence[float],
    fit_poisson: bool = True,
    parameter_ties: Mapping[str, str] = None,
) -> Dict[str, float]:
    names = parameter_names(model_class, fit_poisson, parameter_ties)
    if len(vector) != len(names):
        raise ValueError(f"expected {len(names)} parameters, received {len(vector)}")
    return expand_parameters(
        model_class,
        dict(zip(names, map(float, vector))),
        fit_poisson=fit_poisson,
        parameter_ties=parameter_ties,
    )


def compliance_matrix(constants: Mapping[str, float]) -> np.ndarray:
    e1, e2, e3 = (float(constants[name]) for name in ("E1", "E2", "E3"))
    nu12, nu13, nu23 = (
        float(constants[name]) for name in ("nu12", "nu13", "nu23")
    )
    g12, g13, g23 = (
        float(constants[name]) for name in ("G12", "G13", "G23")
    )
    return np.array(
        [
            [1.0 / e1, -nu12 / e1, -nu13 / e1, 0.0, 0.0, 0.0],
            [-nu12 / e1, 1.0 / e2, -nu23 / e2, 0.0, 0.0, 0.0],
            [-nu13 / e1, -nu23 / e2, 1.0 / e3, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0 / g23, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0 / g13, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0 / g12],
        ],
        dtype=float,
    )


def reciprocal_poisson(constants: Mapping[str, float]) -> Dict[str, float]:
    return {
        "nu21": constants["nu12"] * constants["E2"] / constants["E1"],
        "nu31": constants["nu13"] * constants["E3"] / constants["E1"],
        "nu32": constants["nu23"] * constants["E3"] / constants["E2"],
    }


def off_axis_youngs_modulus(
    constants: Mapping[str, float],
    plane: str,
    angle_degrees: float = 45.0,
) -> float:
    """Plane-stress Young modulus at an angle between two material axes."""
    if plane not in {"12", "13", "23"}:
        raise ValueError("off-axis plane must be 12, 13, or 23")
    first, second = plane
    angle = np.deg2rad(float(angle_degrees))
    m = float(np.cos(angle))
    n = float(np.sin(angle))
    e_first = float(constants[f"E{first}"])
    e_second = float(constants[f"E{second}"])
    nu = float(constants[f"nu{plane}"])
    shear = float(constants[f"G{plane}"])
    inverse = (
        m**4 / e_first
        + n**4 / e_second
        + m**2 * n**2 * (1.0 / shear - 2.0 * nu / e_first)
    )
    if not np.isfinite(inverse) or inverse <= 0.0:
        raise ValueError("off-axis compliance is non-positive")
    return 1.0 / inverse


def check_admissibility(constants: Mapping[str, float]) -> Admissibility:
    required = {
        "E1",
        "E2",
        "E3",
        "nu12",
        "nu13",
        "nu23",
        "G12",
        "G13",
        "G23",
    }
    reasons = []
    if required - set(constants):
        return Admissibility(False, ("missing engineering constants",), float("-inf"))
    values = np.array([constants[name] for name in sorted(required)], dtype=float)
    if not np.isfinite(values).all():
        return Admissibility(False, ("NaN or infinite material value",), float("-inf"))
    if any(constants[name] <= 0 for name in ("E1", "E2", "E3", "G12", "G13", "G23")):
        reasons.append("Young and shear moduli must be positive")
    if any(not -0.2 < constants[name] < 0.5 for name in ("nu12", "nu13", "nu23")):
        reasons.append("Poisson ratios are outside plausible limits")
    if reasons:
        return Admissibility(False, tuple(reasons), float("-inf"))
    matrix = compliance_matrix(constants)
    if not np.allclose(matrix, matrix.T, rtol=1e-12, atol=1e-12):
        reasons.append("compliance matrix is not symmetric")
    eigenvalues = np.linalg.eigvalsh(matrix)
    minimum = float(eigenvalues.min())
    if minimum <= 1.0e-12:
        reasons.append("compliance matrix is not positive definite")
    reciprocals = reciprocal_poisson(constants)
    if any(not np.isfinite(value) or abs(value) >= 0.5 for value in reciprocals.values()):
        reasons.append("reciprocal Poisson ratio is implausible")
    return Admissibility(not reasons, tuple(reasons), minimum)


def material_card(constants: Mapping[str, float], name: str = "PLA") -> str:
    status = check_admissibility(constants)
    if not status.valid:
        raise ValueError("inadmissible material: " + "; ".join(status.reasons))
    first = (
        f"{constants['E1']:.9g}, {constants['E2']:.9g}, {constants['E3']:.9g}, "
        f"{constants['nu12']:.9g}, {constants['nu13']:.9g}, {constants['nu23']:.9g}, "
        f"{constants['G12']:.9g}, {constants['G13']:.9g}"
    )
    return (
        f"*MATERIAL, NAME={name}\n"
        f"*DENSITY\n{DENSITY_TONNES_MM3:.9g}\n"
        "*ELASTIC, TYPE=ENGINEERING CONSTANTS\n"
        f"{first}\n{constants['G23']:.9g}\n"
    )


def complexity(
    model_class: str,
    fit_poisson: bool = True,
    parameter_ties: Mapping[str, str] = None,
) -> int:
    return len(parameter_names(model_class, fit_poisson, parameter_ties))
