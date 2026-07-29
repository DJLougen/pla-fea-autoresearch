#!/usr/bin/env bash
set -euo pipefail
PLA_PYTHON="${PYTHON:-python3}"
"$PLA_PYTHON" -m src.synthetic --output data/example
"$PLA_PYTHON" -m src.prepare --input data/example --output data/processed
"$PLA_PYTHON" -m src.autoresearch --program program.md --research-space configs/research_space.yaml --max-experiments 3
"$PLA_PYTHON" -m src.report \
  --results results/results.tsv \
  --summary results/autoresearch_summary.json
