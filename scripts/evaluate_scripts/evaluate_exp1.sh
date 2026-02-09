#!/bin/bash
# Exp1 Evaluation Script: Automated data merging + Point-wise evaluation + Distribution evaluation
#
# Usage:
#   bash scripts/evaluate_scripts/evaluate_exp1.sh              # Evaluate all models
#   MODEL=Qwen3-30B bash scripts/evaluate_scripts/evaluate_exp1.sh  # Evaluate specific model

set -e

# Get project root directory (scripts/evaluate_scripts/ -> scripts/ -> project root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

cd "${PROJECT_ROOT}"

BASE_DIR=${BASE_DIR:-"/root/autodl-fs"}
EXP=${EXP:-"exp1"}

echo "=========================================="
echo "Experiment: ${EXP}"
if [ -n "${MODEL}" ]; then
    echo "Model: ${MODEL}"
else
    echo "Model: All (auto-discovered)"
fi
echo "Project Path: ${PROJECT_ROOT}"
echo "=========================================="

# Build command - pass --model argument only if MODEL is set
CMD="python analysis/evaluators/evaluate.py --mode full --base-dir ${BASE_DIR} --exp ${EXP}"
if [ -n "${MODEL}" ]; then
    CMD="${CMD} --model ${MODEL}"
fi

eval ${CMD}

echo ""
echo "=========================================="
echo "Evaluation completed"
echo "=========================================="
echo "Results directory: ${BASE_DIR}/result_analysed/${EXP}/"
