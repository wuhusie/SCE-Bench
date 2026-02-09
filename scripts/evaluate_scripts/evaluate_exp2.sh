#!/bin/bash
# ============================================================
# Exp2 Ablation Study Evaluation Script
#
# Automated data merging + Point-wise evaluation (N1) + Distribution evaluation (N50)
# Reuses the complete workflow of evaluators/evaluate.py
#
# Usage:
#   bash scripts/evaluate_scripts/evaluate_exp2.sh                    # Evaluate all ablation configurations
#   MODEL=NOage bash scripts/evaluate_scripts/evaluate_exp2.sh        # Evaluate specific ablation configuration
#
# Prerequisites:
#   1. Reorganize directory structure using reorganize_exp2.sh
#   2. Directory structure: result/exp2/N1/{ablation_config}/*.csv
# ============================================================

set -e

# Get project root directory (scripts/evaluate_scripts/ -> scripts/ -> project root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

cd "${PROJECT_ROOT}"

BASE_DIR=${BASE_DIR:-"/root/autodl-fs"}
EXP=${EXP:-"exp2"}

echo "=========================================="
echo "Exp2 Ablation Study Evaluation"
echo "=========================================="
echo "Experiment: ${EXP}"
if [ -n "${MODEL}" ]; then
    echo "Ablation Config: ${MODEL}"
else
    echo "Ablation Config: All (auto-discovered)"
fi
echo "Project Path: ${PROJECT_ROOT}"
echo "=========================================="

# Check directory structure
N1_DIR="${BASE_DIR}/result/${EXP}/N1"
if [ ! -d "$N1_DIR" ]; then
    echo ""
    echo "Error: N1 directory does not exist: ${N1_DIR}"
    echo ""
    echo "Please run the reorganization script first:"
    echo "  bash scripts/exp2/reorganize_exp2.sh"
    exit 1
fi

# Check for subdirectories (ablation configuration directories)
SUBDIR_COUNT=$(find "$N1_DIR" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
if [ "$SUBDIR_COUNT" -eq 0 ]; then
    echo ""
    echo "Error: No ablation configuration subdirectories found in N1"
    echo ""
    echo "Please run the reorganization script first:"
    echo "  bash scripts/exp2/reorganize_exp2.sh"
    exit 1
fi

echo ""
echo "Found ${SUBDIR_COUNT} ablation configurations"

# Build command
CMD="python analysis/evaluators/evaluate.py --mode full --base-dir ${BASE_DIR} --exp ${EXP}"
if [ -n "${MODEL}" ]; then
    CMD="${CMD} --model ${MODEL}"
fi

echo ""
eval ${CMD}

echo ""
echo "=========================================="
echo "Evaluation completed"
echo "=========================================="
echo "Results directory: ${BASE_DIR}/result_analysed/${EXP}/"
