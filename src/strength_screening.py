from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np

from .materials import check_admissibility, compliance_matrix


@dataclass(frozen=True)
class DirectionalNormalStrengthScreen:
    """Maximum-normal-stress screen using only measured tensile directions."""

    maximum_tensile_sigma1_MPa: float
    maximum_tensile_sigma2_MPa: float
    maximum_tensile_sigma3_MPa: float
    maximum_mean_strength_utilization: float
    p99_mean_strength_utilization: float
    maximum_lower_1sd_strength_utilization: float | None
    governing_direction: str
    sampled_stress_points: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class BoundedTensionShearScreen:
    """Theory-bounded interaction screen where measured shear strength is absent."""

    maximum_von_mises_shear_bookend_utilization: float
    p99_von_mises_shear_bookend_utilization: float
    maximum_equal_tensile_shear_bookend_utilization: float
    p99_equal_tensile_shear_bookend_utilization: float
    maximum_lower_1sd_von_mises_shear_bookend_utilization: float | None
    p99_lower_1sd_von_mises_shear_bookend_utilization: float | None
    maximum_abs_tau12_MPa: float
    maximum_abs_tau23_MPa: float
    maximum_abs_tau31_MPa: float
    governing_component: str
    sampled_stress_points: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class InterfaceInitiationScreen:
    """Mixed normal/tear interface-initiation screen without damage evolution."""

    maximum_utilization: float
    p99_utilization: float
    maximum_lower_1sd_utilization: float
    p99_lower_1sd_utilization: float
    maximum_tensile_sigma3_MPa: float
    maximum_resultant_interface_tear_MPa: float
    governing_component: str
    sampled_stress_points: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _unit_orientation(
    local_axes_global: Sequence[Sequence[float]] | np.ndarray | None,
) -> np.ndarray:
    """Return a direction-cosine matrix whose columns are local axes."""
    if local_axes_global is None:
        return np.eye(3)
    axes = np.asarray(local_axes_global, dtype=float)
    if axes.shape != (3, 3) or not np.isfinite(axes).all():
        raise ValueError("local_axes_global must be a finite 3x3 matrix")
    gram = axes.T @ axes
    if not np.allclose(gram, np.eye(3), atol=1.0e-8, rtol=0.0):
        raise ValueError("local material axes must be orthonormal")
    if np.linalg.det(axes) <= 0.0:
        raise ValueError("local material axes must form a right-handed basis")
    return axes


def stress_rows_to_local(
    stress_rows: np.ndarray,
    local_axes_global: Sequence[Sequence[float]] | np.ndarray | None = None,
) -> np.ndarray:
    """Rotate [Sxx,Syy,Szz,Sxy,Syz,Szx] rows into local material axes."""
    stress = np.asarray(stress_rows, dtype=float)
    if stress.ndim != 2 or stress.shape[1] < 6:
        raise ValueError("stress rows must have at least six tensor components")
    stress = stress[:, :6]
    stress = stress[np.isfinite(stress).all(axis=1)]
    if stress.size == 0:
        return np.empty((0, 6), dtype=float)
    axes = _unit_orientation(local_axes_global)
    tensors = np.empty((len(stress), 3, 3), dtype=float)
    tensors[:, 0, 0] = stress[:, 0]
    tensors[:, 1, 1] = stress[:, 1]
    tensors[:, 2, 2] = stress[:, 2]
    tensors[:, 0, 1] = tensors[:, 1, 0] = stress[:, 3]
    tensors[:, 1, 2] = tensors[:, 2, 1] = stress[:, 4]
    tensors[:, 2, 0] = tensors[:, 0, 2] = stress[:, 5]
    local = np.einsum("ia,nij,jb->nab", axes, tensors, axes)
    return np.column_stack(
        (
            local[:, 0, 0],
            local[:, 1, 1],
            local[:, 2, 2],
            local[:, 0, 1],
            local[:, 1, 2],
            local[:, 2, 0],
        )
    )


def orthotropic_stress_from_tensor_strain(
    strain_rows: np.ndarray,
    engineering_constants: Mapping[str, float],
) -> np.ndarray:
    """Recover solid-material stress from CalculiX tensor-strain rows.

    CalculiX FRD strain uses tensor shear components. The orthotropic
    compliance matrix uses engineering shear, so the three shear strains are
    doubled before applying the stiffness. Returned stress rows follow the
    screening convention [S11,S22,S33,S12,S23,S31].
    """
    strain = np.asarray(strain_rows, dtype=float)
    if strain.ndim != 2 or strain.shape[1] < 6:
        raise ValueError("strain rows must have at least six components")
    strain = strain[:, :6]
    strain = strain[np.isfinite(strain).all(axis=1)]
    if len(strain) == 0:
        return np.empty((0, 6), dtype=float)
    constants = {
        name: float(engineering_constants[name])
        for name in (
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
    admissibility = check_admissibility(constants)
    if not admissibility.valid:
        raise ValueError(
            "inadmissible engineering constants: "
            + "; ".join(admissibility.reasons)
        )
    # FRD: 11,22,33,12,23,31. Compliance: 11,22,33,23,13,12.
    engineering_strain = np.column_stack(
        (
            strain[:, 0],
            strain[:, 1],
            strain[:, 2],
            2.0 * strain[:, 4],
            2.0 * strain[:, 5],
            2.0 * strain[:, 3],
        )
    )
    stress_compliance_order = (
        np.linalg.inv(compliance_matrix(constants))
        @ engineering_strain.T
    ).T
    return np.column_stack(
        (
            stress_compliance_order[:, 0],
            stress_compliance_order[:, 1],
            stress_compliance_order[:, 2],
            stress_compliance_order[:, 5],
            stress_compliance_order[:, 3],
            stress_compliance_order[:, 4],
        )
    )


def directional_tensile_strengths(
    properties: Mapping[str, object],
) -> tuple[np.ndarray, np.ndarray | None]:
    """Extract local 1/2/3 tensile means and optional one-SD lower values."""
    raw = properties.get("reported_strength_reference")
    if not isinstance(raw, Mapping):
        raise ValueError("material profile has no reported strength reference")
    xy = float(raw["overall_tensile_UTS_MPa"])
    z = float(raw["z_upright_tensile_UTS_MPa"])
    means = np.asarray([xy, xy, z], dtype=float)
    if not np.isfinite(means).all() or np.any(means <= 0.0):
        raise ValueError("directional tensile strengths must be finite and positive")
    xy_sd = raw.get("overall_tensile_UTS_standard_deviation_MPa")
    z_sd = raw.get("z_upright_tensile_UTS_standard_deviation_MPa")
    if xy_sd is None or z_sd is None:
        return means, None
    lower = means - np.asarray([float(xy_sd), float(xy_sd), float(z_sd)])
    if not np.isfinite(lower).all() or np.any(lower <= 0.0):
        raise ValueError("one-SD lower tensile strengths must remain positive")
    return means, lower


def screen_directional_normal_strength(
    stress_rows: np.ndarray,
    mean_strengths_MPa: Sequence[float],
    lower_1sd_strengths_MPa: Sequence[float] | None = None,
    local_axes_global: Sequence[Sequence[float]] | np.ndarray | None = None,
) -> DirectionalNormalStrengthScreen | None:
    """Screen tensile normal stress; shear, compression, and fracture are omitted."""
    strengths = np.asarray(mean_strengths_MPa, dtype=float)
    if strengths.shape != (3,) or not np.isfinite(strengths).all():
        raise ValueError("mean_strengths_MPa must contain three finite values")
    if np.any(strengths <= 0.0):
        raise ValueError("mean tensile strengths must be positive")
    lower = None
    if lower_1sd_strengths_MPa is not None:
        lower = np.asarray(lower_1sd_strengths_MPa, dtype=float)
        if lower.shape != (3,) or not np.isfinite(lower).all():
            raise ValueError(
                "lower_1sd_strengths_MPa must contain three finite values"
            )
        if np.any(lower <= 0.0):
            raise ValueError("lower one-SD tensile strengths must be positive")
    local = stress_rows_to_local(stress_rows, local_axes_global)
    if len(local) == 0:
        return None
    tensile = np.maximum(local[:, :3], 0.0)
    mean_utilization = tensile / strengths
    point_utilization = np.max(mean_utilization, axis=1)
    maximum_by_direction = np.max(tensile, axis=0)
    maximum_utilization_by_direction = maximum_by_direction / strengths
    governing_index = int(np.argmax(maximum_utilization_by_direction))
    lower_utilization = (
        None
        if lower is None
        else float(np.max(maximum_by_direction / lower))
    )
    return DirectionalNormalStrengthScreen(
        maximum_tensile_sigma1_MPa=float(maximum_by_direction[0]),
        maximum_tensile_sigma2_MPa=float(maximum_by_direction[1]),
        maximum_tensile_sigma3_MPa=float(maximum_by_direction[2]),
        maximum_mean_strength_utilization=float(
            maximum_utilization_by_direction[governing_index]
        ),
        p99_mean_strength_utilization=float(
            np.percentile(point_utilization, 99.0)
        ),
        maximum_lower_1sd_strength_utilization=lower_utilization,
        governing_direction=f"local_{governing_index + 1}",
        sampled_stress_points=len(local),
    )


def _tension_shear_utilization(
    local_stress: np.ndarray,
    tensile_strengths_MPa: np.ndarray,
    shear_strength_factor: float,
) -> tuple[np.ndarray, np.ndarray]:
    tensile = np.maximum(local_stress[:, :3], 0.0)
    shear = np.abs(local_stress[:, 3:6])
    pair_strengths = np.asarray(
        [
            min(tensile_strengths_MPa[0], tensile_strengths_MPa[1]),
            min(tensile_strengths_MPa[1], tensile_strengths_MPa[2]),
            min(tensile_strengths_MPa[2], tensile_strengths_MPa[0]),
        ],
        dtype=float,
    )
    shear_strengths = shear_strength_factor * pair_strengths
    components = np.column_stack(
        (tensile / tensile_strengths_MPa, shear / shear_strengths)
    )
    utilization = np.sqrt(np.sum(np.square(components), axis=1))
    return utilization, components


def screen_bounded_tension_shear_interaction(
    stress_rows: np.ndarray,
    mean_strengths_MPa: Sequence[float],
    lower_1sd_strengths_MPa: Sequence[float] | None = None,
    local_axes_global: Sequence[Sequence[float]] | np.ndarray | None = None,
) -> BoundedTensionShearScreen | None:
    """Screen tensile/shear interaction between two explicit shear bookends.

    The lower shear-strength bookend maps uniaxial tensile strength to pure
    shear with S=X/sqrt(3), as in a von-Mises equivalence. The upper bookend
    uses S=X. Neither is measured Bambu PLA shear strength. Compressive normal
    stress is intentionally omitted because no matched compression strength
    is available.
    """
    strengths = np.asarray(mean_strengths_MPa, dtype=float)
    if strengths.shape != (3,) or not np.isfinite(strengths).all():
        raise ValueError("mean_strengths_MPa must contain three finite values")
    if np.any(strengths <= 0.0):
        raise ValueError("mean tensile strengths must be positive")
    lower = None
    if lower_1sd_strengths_MPa is not None:
        lower = np.asarray(lower_1sd_strengths_MPa, dtype=float)
        if lower.shape != (3,) or not np.isfinite(lower).all():
            raise ValueError(
                "lower_1sd_strengths_MPa must contain three finite values"
            )
        if np.any(lower <= 0.0):
            raise ValueError("lower one-SD tensile strengths must be positive")

    local = stress_rows_to_local(stress_rows, local_axes_global)
    if len(local) == 0:
        return None
    von_mises, components = _tension_shear_utilization(
        local, strengths, 1.0 / np.sqrt(3.0)
    )
    equal_tensile, _ = _tension_shear_utilization(local, strengths, 1.0)
    lower_utilization = None
    lower_p99_utilization = None
    if lower is not None:
        lower_values, _ = _tension_shear_utilization(
            local, lower, 1.0 / np.sqrt(3.0)
        )
        lower_utilization = float(np.max(lower_values))
        lower_p99_utilization = float(
            np.percentile(lower_values, 99.0)
        )

    peak_index = int(np.argmax(von_mises))
    labels = ("sigma1+", "sigma2+", "sigma3+", "|tau12|", "|tau23|", "|tau31|")
    governing_index = int(np.argmax(np.square(components[peak_index])))
    shear_maxima = np.max(np.abs(local[:, 3:6]), axis=0)
    return BoundedTensionShearScreen(
        maximum_von_mises_shear_bookend_utilization=float(np.max(von_mises)),
        p99_von_mises_shear_bookend_utilization=float(
            np.percentile(von_mises, 99.0)
        ),
        maximum_equal_tensile_shear_bookend_utilization=float(
            np.max(equal_tensile)
        ),
        p99_equal_tensile_shear_bookend_utilization=float(
            np.percentile(equal_tensile, 99.0)
        ),
        maximum_lower_1sd_von_mises_shear_bookend_utilization=lower_utilization,
        p99_lower_1sd_von_mises_shear_bookend_utilization=(
            lower_p99_utilization
        ),
        maximum_abs_tau12_MPa=float(shear_maxima[0]),
        maximum_abs_tau23_MPa=float(shear_maxima[1]),
        maximum_abs_tau31_MPa=float(shear_maxima[2]),
        governing_component=labels[governing_index],
        sampled_stress_points=len(local),
    )


def screen_interface_initiation(
    stress_rows: np.ndarray,
    normal_strength_MPa: float,
    tear_strength_MPa: float,
    lower_1sd_normal_strength_MPa: float,
    lower_1sd_tear_strength_MPa: float,
    local_axes_global: Sequence[Sequence[float]] | np.ndarray | None = None,
) -> InterfaceInitiationScreen | None:
    """Screen local-3 opening plus apparent interface tear interaction.

    The quadratic interaction is a transparent initiation sensitivity. It is
    not a cohesive traction-separation relation and has no damage evolution or
    fracture energy.
    """
    strengths = np.asarray(
        [
            normal_strength_MPa,
            tear_strength_MPa,
            lower_1sd_normal_strength_MPa,
            lower_1sd_tear_strength_MPa,
        ],
        dtype=float,
    )
    if not np.isfinite(strengths).all() or np.any(strengths <= 0.0):
        raise ValueError("interface strengths must be finite and positive")
    local = stress_rows_to_local(stress_rows, local_axes_global)
    if len(local) == 0:
        return None
    opening = np.maximum(local[:, 2], 0.0)
    tear = np.sqrt(np.square(local[:, 4]) + np.square(local[:, 5]))
    opening_component = opening / normal_strength_MPa
    tear_component = tear / tear_strength_MPa
    utilization = np.sqrt(
        np.square(opening_component) + np.square(tear_component)
    )
    lower_utilization = np.sqrt(
        np.square(opening / lower_1sd_normal_strength_MPa)
        + np.square(tear / lower_1sd_tear_strength_MPa)
    )
    peak_index = int(np.argmax(utilization))
    governing = (
        "local_3_opening"
        if opening_component[peak_index] >= tear_component[peak_index]
        else "resultant_tau23_tau31_tear"
    )
    return InterfaceInitiationScreen(
        maximum_utilization=float(np.max(utilization)),
        p99_utilization=float(np.percentile(utilization, 99.0)),
        maximum_lower_1sd_utilization=float(np.max(lower_utilization)),
        p99_lower_1sd_utilization=float(
            np.percentile(lower_utilization, 99.0)
        ),
        maximum_tensile_sigma3_MPa=float(np.max(opening)),
        maximum_resultant_interface_tear_MPa=float(np.max(tear)),
        governing_component=governing,
        sampled_stress_points=len(local),
    )


def screen_curve_directional_normal_strength(
    curve: Mapping[str, np.ndarray],
    properties: Mapping[str, object],
    local_axes_global: Sequence[Sequence[float]] | np.ndarray | None = None,
) -> DirectionalNormalStrengthScreen | None:
    stress = curve.get("frd_stress")
    if stress is None:
        return None
    means, lower = directional_tensile_strengths(properties)
    return screen_directional_normal_strength(
        stress,
        means,
        lower,
        local_axes_global=local_axes_global,
    )


def screen_curve_bounded_tension_shear_interaction(
    curve: Mapping[str, np.ndarray],
    properties: Mapping[str, object],
    local_axes_global: Sequence[Sequence[float]] | np.ndarray | None = None,
) -> BoundedTensionShearScreen | None:
    stress = curve.get("frd_stress")
    if stress is None:
        return None
    means, lower = directional_tensile_strengths(properties)
    return screen_bounded_tension_shear_interaction(
        stress,
        means,
        lower,
        local_axes_global=local_axes_global,
    )
