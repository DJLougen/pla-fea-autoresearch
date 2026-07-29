from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, Mapping, Tuple

import numpy as np


GRID_POINTS = 101
EPSILON = 1.0e-12
LATERAL_STRAIN_WEIGHT = 0.20


@dataclass(frozen=True)
class CurveMetrics:
    curve_rmse: float
    slope_error: float
    peak_error: float
    x_at_peak_error: float
    lateral_strain_error: float
    lateral_strain_used: bool
    total: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def clean_curve(x: Iterable[float], y: Iterable[float]) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(list(x), dtype=float)
    y = np.asarray(list(y), dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 2:
        raise ValueError("curve needs at least two finite points")
    order = np.argsort(x, kind="mergesort")
    x, y = x[order], y[order]
    unique_x, inverse = np.unique(x, return_inverse=True)
    if len(unique_x) != len(x):
        sums = np.zeros(len(unique_x))
        counts = np.zeros(len(unique_x))
        np.add.at(sums, inverse, y)
        np.add.at(counts, inverse, 1)
        x, y = unique_x, sums / counts
    if x[-1] - x[0] <= EPSILON:
        raise ValueError("curve has no independent-variable range")
    return x, y


def interpolate_common(
    measured_x: Iterable[float],
    measured_y: Iterable[float],
    simulated_x: Iterable[float],
    simulated_y: Iterable[float],
    points: int = GRID_POINTS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mx, my = clean_curve(measured_x, measured_y)
    sx, sy = clean_curve(simulated_x, simulated_y)
    mxn = (mx - mx[0]) / (mx[-1] - mx[0])
    sxn = (sx - sx[0]) / (sx[-1] - sx[0])
    grid = np.linspace(0.0, min(mxn[-1], sxn[-1]), points)
    return grid, np.interp(grid, mxn, my), np.interp(grid, sxn, sy)


def initial_slope(x: np.ndarray, y: np.ndarray, fraction: float = 0.2) -> float:
    cutoff = max(3, int(np.ceil(len(x) * fraction)))
    design = np.column_stack([x[:cutoff], np.ones(cutoff)])
    return float(np.linalg.lstsq(design, y[:cutoff], rcond=None)[0][0])


def curve_metrics(
    measured_x: Iterable[float],
    measured_y: Iterable[float],
    simulated_x: Iterable[float],
    simulated_y: Iterable[float],
    measured_lateral_strain: Iterable[float] | None = None,
    simulated_lateral_strain: Iterable[float] | None = None,
) -> CurveMetrics:
    grid, measured, simulated = interpolate_common(
        measured_x, measured_y, simulated_x, simulated_y
    )
    scale = max(np.ptp(measured), np.max(np.abs(measured)), EPSILON)
    curve_rmse = float(np.sqrt(np.mean((simulated - measured) ** 2)) / scale)
    measured_slope = initial_slope(grid, measured)
    simulated_slope = initial_slope(grid, simulated)
    slope_error = abs(simulated_slope - measured_slope) / max(
        abs(measured_slope), EPSILON
    )
    measured_peak_index = int(np.argmax(np.abs(measured)))
    simulated_peak_index = int(np.argmax(np.abs(simulated)))
    measured_peak = abs(float(measured[measured_peak_index]))
    simulated_peak = abs(float(simulated[simulated_peak_index]))
    peak_error = abs(simulated_peak - measured_peak) / max(measured_peak, EPSILON)
    x_at_peak_error = abs(
        float(grid[simulated_peak_index] - grid[measured_peak_index])
    )
    axial_total = (
        0.45 * curve_rmse
        + 0.25 * slope_error
        + 0.20 * peak_error
        + 0.10 * x_at_peak_error
    )
    lateral_used = measured_lateral_strain is not None
    lateral_error = 0.0
    if lateral_used:
        if simulated_lateral_strain is None:
            raise ValueError(
                "measured lateral strain requires simulated lateral strain"
            )
        measured_axial, measured_lateral = clean_curve(
            measured_x, measured_lateral_strain
        )
        simulated_axial, simulated_lateral = clean_curve(
            simulated_x, simulated_lateral_strain
        )
        measured_lateral_slope = initial_slope(
            measured_axial, measured_lateral
        )
        simulated_lateral_slope = initial_slope(
            simulated_axial, simulated_lateral
        )
        lateral_error = float(
            abs(simulated_lateral_slope - measured_lateral_slope)
            / max(abs(measured_lateral_slope), EPSILON)
        )
    total = (
        axial_total
        if not lateral_used
        else (1.0 - LATERAL_STRAIN_WEIGHT) * axial_total
        + LATERAL_STRAIN_WEIGHT * lateral_error
    )
    return CurveMetrics(
        curve_rmse,
        slope_error,
        peak_error,
        x_at_peak_error,
        lateral_error,
        lateral_used,
        float(total),
    )


def aggregate_metrics(
    per_test: Mapping[str, CurveMetrics], weights: Mapping[str, float] = None
) -> Dict[str, float]:
    if not per_test:
        raise ValueError("cannot aggregate an empty metric collection")
    weights = weights or {}
    raw = np.array([float(weights.get(name, 1.0)) for name in per_test])
    if not np.isfinite(raw).all() or np.any(raw < 0) or raw.sum() <= 0:
        raise ValueError("metric weights must be finite, non-negative, and nonzero")
    normalized = raw / raw.sum()
    fields = (
        "curve_rmse",
        "slope_error",
        "peak_error",
        "x_at_peak_error",
        "lateral_strain_error",
        "total",
    )
    return {
        field: float(
            np.dot(
                normalized,
                [getattr(metric, field) for metric in per_test.values()],
            )
        )
        for field in fields
    }
