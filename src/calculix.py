from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Mapping, Optional

import numpy as np

from .parsing import parse_solver_outputs


TOKEN = re.compile(r"\{\{([A-Z0-9_]+)\}\}")


@dataclass
class SolverResult:
    ok: bool
    status: str
    run_dir: Path
    runtime_seconds: float
    stdout: str = ""
    stderr: str = ""
    curve: Dict[str, np.ndarray] = field(default_factory=dict)
    error: Optional[str] = None
    return_code: Optional[int] = None


def find_ccx(explicit: Optional[str] = None) -> Optional[Path]:
    if explicit:
        path = Path(explicit).expanduser()
        if path.is_file() and os.access(path, os.X_OK):
            return path.resolve()
        resolved = shutil.which(explicit)
        if resolved:
            return Path(resolved).resolve()
        if explicit != "ccx":
            return None
    resolved = shutil.which("ccx")
    if resolved:
        return Path(resolved).resolve()
    versioned = []
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        if not directory:
            continue
        for candidate in Path(directory).glob("ccx_[0-9]*"):
            if not candidate.is_file() or not os.access(candidate, os.X_OK):
                continue
            match = re.fullmatch(r"ccx_(\d+(?:\.\d+)*)", candidate.name)
            if match:
                version = tuple(int(part) for part in match.group(1).split("."))
                versioned.append((version, candidate.resolve()))
    return max(versioned, default=((), None), key=lambda item: item[0])[1]


def render_template(template: str, values: Mapping[str, object]) -> str:
    def replace(match: re.Match) -> str:
        key = match.group(1)
        if key not in values:
            raise ValueError(f"missing template value: {key}")
        return str(values[key])

    rendered = TOKEN.sub(replace, template)
    remaining = TOKEN.findall(rendered)
    if remaining:
        raise ValueError(f"unrendered template tokens: {remaining}")
    return rendered


ORIENTATIONS = {
    # The coupon loads in global X. These bases map that load to local 1, 2, 3.
    "1": "1., 0., 0., 0., 1., 0.",
    "2": "0., 1., 0., 1., 0., 0.",
    "3": "0., 1., 0., 0., 0., 1.",
    # Global X lies at 45 degrees between the named local-axis pair.
    "12_45": "1., 1., 0., -1., 1., 0.",
    "13_45": "1., 0., 1., 0., 1., 0.",
    "23_45": "0., 1., 0., 1., 0., 1.",
    "mixed": "1., 1., 0., 0., 0., 1.",
}


class CalculixRunner:
    def __init__(
        self,
        runs_dir: Path,
        executable: Optional[str] = None,
        timeout_seconds: float = 120.0,
        keep_successful: bool = False,
    ):
        self.runs_dir = Path(runs_dir)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.executable = find_ccx(executable)
        self.timeout_seconds = float(timeout_seconds)
        self.keep_successful = bool(keep_successful)

    @property
    def available(self) -> bool:
        return self.executable is not None

    def run(
        self,
        template_path: Path,
        values: Mapping[str, object],
        run_id: Optional[str] = None,
    ) -> SolverResult:
        identifier = run_id or uuid.uuid4().hex[:16]
        run_dir = self.runs_dir / identifier
        if run_dir.exists():
            run_dir = self.runs_dir / f"{identifier}-{uuid.uuid4().hex[:8]}"
        run_dir.mkdir(parents=True, exist_ok=False)
        if not self.available:
            return SolverResult(
                False,
                "executable_missing",
                run_dir,
                0.0,
                error="CalculiX executable was not found",
            )
        try:
            template = Path(template_path).read_text(encoding="utf-8")
            rendered = render_template(template, values)
            job_name = "model"
            (run_dir / f"{job_name}.inp").write_text(rendered, encoding="utf-8")
        except Exception as exc:
            return SolverResult(
                False, "render_failed", run_dir, 0.0, error=str(exc)
            )
        start = time.monotonic()
        try:
            process = subprocess.run(
                [str(self.executable), job_name],
                cwd=run_dir,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            runtime = time.monotonic() - start
            (run_dir / "stdout.txt").write_text(exc.stdout or "", encoding="utf-8")
            (run_dir / "stderr.txt").write_text(exc.stderr or "", encoding="utf-8")
            return SolverResult(
                False,
                "timeout",
                run_dir,
                runtime,
                stdout=exc.stdout or "",
                stderr=exc.stderr or "",
                error=f"solver exceeded {self.timeout_seconds:g} seconds",
            )
        runtime = time.monotonic() - start
        (run_dir / "stdout.txt").write_text(process.stdout, encoding="utf-8")
        (run_dir / "stderr.txt").write_text(process.stderr, encoding="utf-8")
        combined = (process.stdout + "\n" + process.stderr).lower()
        # CalculiX prints "no convergence" for an individual Newton iteration
        # before cutting/increasing an increment and retrying it.  That message
        # is only terminal when the analysis never reaches "Job finished".
        # Treating every occurrence as fatal incorrectly rejects successful
        # NLGEOM analyses.
        fatal_markers = ("job failed", "divergence", "*error", "singular matrix")
        terminal_nonconvergence = (
            "no convergence" in combined and "job finished" not in combined
        )
        if (
            process.returncode != 0
            or terminal_nonconvergence
            or any(marker in combined for marker in fatal_markers)
        ):
            return SolverResult(
                False,
                "solver_failed",
                run_dir,
                runtime,
                stdout=process.stdout,
                stderr=process.stderr,
                error=f"CalculiX returned {process.returncode}",
                return_code=process.returncode,
            )
        try:
            curve = parse_solver_outputs(run_dir, job_name)
        except Exception as exc:
            return SolverResult(
                False,
                "parse_failed",
                run_dir,
                runtime,
                stdout=process.stdout,
                stderr=process.stderr,
                error=str(exc),
                return_code=process.returncode,
            )
        return SolverResult(
            True,
            "ok",
            run_dir,
            runtime,
            stdout=process.stdout,
            stderr=process.stderr,
            curve=curve,
            return_code=process.returncode,
        )
