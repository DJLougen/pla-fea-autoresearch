from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import pandas as pd


def _json(value: object) -> Dict[str, object]:
    if value is None or (isinstance(value, float) and pd.isna(value)) or value == "":
        return {}
    return json.loads(str(value))


def _latest_summary_rows(
    frame: pd.DataFrame, summary_file: Path
) -> tuple[pd.DataFrame, str]:
    summary = json.loads(Path(summary_file).read_text(encoding="utf-8"))
    attempts = summary.get("attempts", [])
    rows = []
    for attempt in attempts:
        experiment_id = str(attempt["experiment_id"])
        matches = frame.loc[
            frame["experiment_id"].astype(str) == experiment_id
        ]
        if matches.empty:
            raise ValueError(
                f"summary experiment {experiment_id!r} is absent from results"
            )
        rows.append(matches.iloc[-1])
    if not rows:
        raise ValueError("autoresearch summary contains no attempts")
    return pd.DataFrame(rows).reset_index(drop=True), str(
        summary["best_experiment_id"]
    )


def build_report(
    results_file: Path, summary_file: Path | None = None
) -> str:
    frame = pd.read_csv(results_file, sep="\t")
    if frame.empty:
        raise ValueError("results file is empty")
    summary_best = None
    if summary_file is not None:
        frame, summary_best = _latest_summary_rows(frame, summary_file)
    frame["accepted_bool"] = frame["accepted"].astype(str).str.lower().isin(
        {"true", "1", "yes"}
    )
    eligible = frame.loc[frame["accepted_bool"]]
    if eligible.empty:
        raise ValueError("results contain no accepted experiment")
    if summary_best is None:
        best = eligible.loc[
            eligible["objective_score"].astype(float).idxmin()
        ]
    else:
        selected = eligible.loc[
            eligible["experiment_id"].astype(str) == summary_best
        ]
        if selected.empty:
            raise ValueError(
                "summary best experiment is not accepted in latest results"
            )
        best = selected.iloc[-1]
    constants = _json(best["engineering_constants_json"])
    parameters = _json(best["parameters_json"])
    bounds = _json(best["bounds_json"])
    per_test_scores = _json(best["per_test_scores_json"])
    lateral_curve_count = sum(
        bool(metrics.get("lateral_strain_used"))
        for partition in per_test_scores.values()
        if isinstance(partition, dict)
        for metrics in partition.values()
        if isinstance(metrics, dict)
    )

    lines = [
        "# PLA elastic inverse-calibration report",
        "",
        "## Selected model",
        "",
        f"- Experiment: `{best['experiment_id']}`",
        f"- Model class: **{best['model_class']}**",
        f"- Hypothesis: {best['hypothesis']}",
        f"- Complexity-adjusted validation score: {float(best['objective_score']):.6g}",
        f"- Calibration error: {float(best['calibration_score']):.6g}",
        f"- Validation error: {float(best['validation_score']):.6g}",
        f"- Held-out component error: {float(best['heldout_score']):.6g}",
        f"- Solver failures: {int(best['solver_failure_count'])}",
        f"- Objective evaluations: {int(best['objective_evaluations'])}",
        "",
        "## Best-fit optimization parameters",
        "",
        "| Parameter | Value | Bounds |",
        "|---|---:|---:|",
    ]
    for name, value in parameters.items():
        limit = bounds.get(name, ["—", "—"])
        lines.append(f"| {name} | {float(value):.6g} | [{limit[0]}, {limit[1]}] |")
    lines.extend(
        [
            "",
            "## Expanded CalculiX engineering constants",
            "",
            "| Constant | Value |",
            "|---|---:|",
        ]
    )
    for name in ("E1", "E2", "E3", "nu12", "nu13", "nu23", "G12", "G13", "G23"):
        lines.append(f"| {name} | {float(constants[name]):.6g} |")

    lines.extend(
        [
            "",
            "## Model-complexity comparison",
            "",
            "| Experiment | Model | Parameters | Calibration | Validation | Held out | Decision |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for _, row in frame.iterrows():
        parameter_count = len(_json(row["parameters_json"]))
        decision = "accepted" if str(row["accepted"]).lower() in {"true", "1"} else "rejected"
        lines.append(
            f"| {row['experiment_id']} | {row['model_class']} | {parameter_count} | "
            f"{float(row['calibration_score']):.5g} | {float(row['validation_score']):.5g} | "
            f"{float(row['heldout_score']):.5g} | {decision} |"
        )
    orthotropic = frame.loc[frame["model_class"] == "orthotropic"]
    simpler = frame.loc[frame["model_class"] != "orthotropic"]
    if not orthotropic.empty and not simpler.empty:
        ortho_best = float(orthotropic["validation_score"].astype(float).min())
        simple_best = float(simpler["validation_score"].astype(float).min())
        improvement = (simple_best - ortho_best) / max(abs(simple_best), 1.0e-12)
        material = improvement >= 0.02 and str(
            orthotropic.loc[
                orthotropic["validation_score"].astype(float).idxmin(), "accepted"
            ]
        ).lower() in {"true", "1"}
        change_text = (
            f"improved by {improvement:.1%}"
            if improvement >= 0.0
            else f"worsened by {-improvement:.1%}"
        )
        lines.extend(
            [
                "",
                "## Orthotropy conclusion",
                "",
                f"The best orthotropic validation error {change_text} "
                f"relative to the best simpler model. Added orthotropic parameters "
                f"{'materially improved prediction' if material else 'did not materially improve prediction'} "
                "under the fixed 2% reporting threshold and complexity-aware selection rule.",
            ]
        )
    poisson_text = (
        f"measured lateral strain was included in {lateral_curve_count} "
        "tensile curve scores, so Poisson-sensitive contraction contributes "
        "to this fit. "
        if lateral_curve_count
        else "Measured lateral strain is still needed for independent "
        "Poisson identification. "
    )
    lines.extend(
        [
            "",
            "## Interpretation cautions",
            "",
            "Axial curves alone do not identify shear moduli or Poisson ratios. "
            "The supported 12/13/23 off-axis curves constrain shear compliance, "
            f"and {poisson_text}"
            "Uncertainty analysis is required before treating "
            "a full orthotropic fit as unique. Tensile ultimate strength is not "
            "yield stress, and a bending modulus must not automatically replace "
            "an axial Young's modulus.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a calibration summary report")
    parser.add_argument("--results", required=True)
    parser.add_argument(
        "--summary",
        help="Scope the report to attempts in an autoresearch summary JSON",
    )
    parser.add_argument("--output", default="results/report.md")
    args = parser.parse_args()
    report = build_report(
        Path(args.results),
        None if args.summary is None else Path(args.summary),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
