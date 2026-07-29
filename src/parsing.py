from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Mapping

import numpy as np


NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?"
NUMBER_PATTERN = re.compile(NUMBER)


def _number(value: str) -> float:
    return float(value.replace("D", "E").replace("d", "e"))


def parse_dat(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        raise ValueError(f"missing CalculiX data file: {path}")
    text = path.read_text(encoding="utf-8", errors="replace")
    lower = text.lower()
    if any(
        marker in lower
        for marker in ("divergence", "no convergence", "job failed", "singular matrix")
    ):
        raise ValueError("CalculiX output reports non-convergence or singularity")

    # Also accept a compact machine-readable table. This is useful for wrapper
    # compatibility and deterministic parser tests.
    compact = re.findall(
        rf"^\s*CURVE\s+({NUMBER})\s+({NUMBER})\s+({NUMBER})(?:\s+({NUMBER})\s+({NUMBER}))?",
        text,
        flags=re.MULTILINE | re.IGNORECASE,
    )
    if compact:
        rows = [[_number(item) if item else np.nan for item in row] for row in compact]
        array = np.asarray(rows, dtype=float)
        return {
            "time": array[:, 0],
            "displacement": array[:, 1],
            "reaction_force": array[:, 2],
            "stress": array[:, 3],
            "strain": array[:, 4],
        }

    header = re.compile(
        rf"^\s*(displacements|forces)\s+\([^)]+\)\s+for set\s+(\S+)\s+and time\s+({NUMBER})",
        flags=re.MULTILINE | re.IGNORECASE,
    )
    any_history_header = re.compile(
        r"^\s*(?:displacements|forces|stresses|strains)\s+\(",
        flags=re.MULTILINE | re.IGNORECASE,
    )
    matches = list(header.finditer(text))
    observations: Dict[float, Dict[str, float]] = {}
    row_pattern = re.compile(rf"^\s*\d+\s+({NUMBER})\s+({NUMBER})\s+({NUMBER})", re.MULTILINE)
    for index, match in enumerate(matches):
        following = any_history_header.search(text, match.end())
        end = following.start() if following else len(text)
        block = text[match.end() : end]
        rows = row_pattern.findall(block)
        if not rows:
            continue
        time = _number(match.group(3))
        values = np.asarray([[_number(item) for item in row] for row in rows])
        record = observations.setdefault(time, {})
        kind = match.group(1).lower()
        set_name = match.group(2).upper()
        if kind == "displacements" and "LOAD" in set_name:
            means = np.mean(values, axis=0)
            axis = int(np.argmax(np.abs(means)))
            record["axis"] = axis
            record["displacement"] = float(means[axis])
        elif kind == "forces" and ("FIX" in set_name or "SUPPORT" in set_name):
            sums = np.sum(values, axis=0)
            axis = int(record.get("axis", np.argmax(np.abs(sums))))
            record["reaction_force"] = float(abs(sums[axis]))
    field_header = re.compile(
        rf"^\s*(stresses|strains).*?for set\s+\S+\s+and time\s+({NUMBER})",
        flags=re.MULTILINE | re.IGNORECASE,
    )
    field_matches = list(field_header.finditer(text))
    field_row = re.compile(
        rf"^\s*\d+\s+\d+\s+({NUMBER})\s+({NUMBER})\s+({NUMBER})"
        rf"\s+({NUMBER})\s+({NUMBER})\s+({NUMBER})",
        re.MULTILINE,
    )
    for index, match in enumerate(field_matches):
        end = (
            field_matches[index + 1].start()
            if index + 1 < len(field_matches)
            else len(text)
        )
        rows = field_row.findall(text[match.end() : end])
        if not rows:
            continue
        time = _number(match.group(2))
        values = np.asarray([[_number(item) for item in row] for row in rows])
        record = observations.setdefault(time, {})
        axis = int(record.get("axis", 0))
        record["stress" if match.group(1).lower() == "stresses" else "strain"] = float(
            np.mean(values[:, axis])
        )
    complete = [
        (
            time,
            row["displacement"],
            row["reaction_force"],
            row.get("stress", np.nan),
            row.get("strain", np.nan),
        )
        for time, row in sorted(observations.items())
        if "displacement" in row and "reaction_force" in row
    ]
    if not complete:
        raise ValueError("no paired displacement/reaction-force history found")
    array = np.asarray(complete, dtype=float)
    return {
        "time": array[:, 0],
        "displacement": array[:, 1],
        "reaction_force": array[:, 2],
        "stress": array[:, 3],
        "strain": array[:, 4],
    }


def parse_frd(path: Path) -> Mapping[str, np.ndarray]:
    if not path.exists():
        return {}
    records: Dict[str, List[List[float]]] = {}
    field = None
    # FRD repeats a complete field block for every saved increment. Keep the
    # latest block for each field; callers use it as the endpoint state. This
    # avoids retaining millions of obsolete rows for nonlinear histories.
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith(" -4"):
                field = line[4:16].strip().lower()
                records[field] = []
                continue
            if not line.startswith(" -1") or not field:
                continue
            values = NUMBER_PATTERN.findall(line[3:])
            if len(values) >= 2:
                records[field].append(
                    [_number(value) for value in values[1:]]
                )
    if not records:
        return {}
    result = {}
    for name, rows in records.items():
        if not rows:
            continue
        width = max(len(row) for row in rows)
        array = np.full((len(rows), width), np.nan)
        for index, row in enumerate(rows):
            array[index, : len(row)] = row
        result[f"frd_{name}"] = array
    return result


def parse_frd_scalar_max_history(
    path: Path, field_name: str
) -> Dict[float, float]:
    """Stream the maximum scalar field value for every saved FRD time."""
    if not path.exists():
        raise ValueError(f"missing CalculiX results file: {path}")
    target = field_name.strip().lower()
    current_time: float | None = None
    current_field: str | None = None
    maxima: Dict[float, float] = {}
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.lstrip().startswith("100CL"):
                tokens = line.split()
                if len(tokens) >= 3:
                    # CalculiX writes: 100CL, result-set key, analysis time.
                    # Splitting avoids treating the "100" inside 100CL as a
                    # separate numeric token.
                    current_time = _number(tokens[2])
                continue
            if line.startswith(" -4"):
                current_field = line[4:16].strip().lower()
                continue
            if (
                current_field != target
                or current_time is None
                or not line.startswith(" -1")
            ):
                continue
            values = NUMBER_PATTERN.findall(line[3:])
            if len(values) < 2:
                continue
            scalar = max(abs(_number(value)) for value in values[1:])
            maxima[current_time] = max(maxima.get(current_time, 0.0), scalar)
    return dict(sorted(maxima.items()))


def parse_solver_outputs(run_dir: Path, job_name: str) -> Dict[str, np.ndarray]:
    curve = parse_dat(run_dir / f"{job_name}.dat")
    curve.update(parse_frd(run_dir / f"{job_name}.frd"))
    return curve
