#!/bin/bash
# Exp6 Base vs Instruct Model Comparison Evaluation Script
#
# Compare performance differences between Base models and Instruct-tuned models
#
# Usage:
#   bash scripts/evaluate_scripts/evaluate_exp6.sh                    # Evaluate all configurations (Base and Instruct)
#   MODEL=Instruct bash scripts/evaluate_scripts/evaluate_exp6.sh     # Evaluate specific configuration
#
# Prerequisites:
#   Directory structure should be:
#     result/exp6/N1/{Base,Instruct}/*.csv
#     result/exp6/N50/{Base,Instruct}/*.csv (Optional)
# ============================================================

set -e

# Get project root directory (scripts/evaluate_scripts/ -> scripts/ -> project root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

cd "${PROJECT_ROOT}"

# Allow external override of BASE_DIR
BASE_DIR=${BASE_DIR:-"/root/autodl-fs"}
EXP=${EXP:-"exp6"}

echo "=========================================="
echo "Exp6 Base vs Instruct Evaluation (Base vs Instruct Study)"
echo "=========================================="
echo "Experiment: ${EXP}"
if [ -n "${MODEL}" ]; then
    echo "Configuration: ${MODEL}"
else
    echo "Configuration: All (Base, Instruct)"
fi
echo "Project Path: ${PROJECT_ROOT}"
echo "Data Path: ${BASE_DIR}/result/${EXP}"
echo "=========================================="

# Check N1 directory structure (N1 is required)
N1_DIR="${BASE_DIR}/result/${EXP}/N1"

if [ ! -d "$N1_DIR" ]; then
    echo ""
    echo "Warning: N1 directory does not exist: ${N1_DIR}"
    echo "Please confirm if run_batch.sh results have been correctly generated."
    echo "Expected structure: ${BASE_DIR}/result/exp6/N1/{Base,Instruct}/*.csv"
    exit 1
fi

SUBDIR_COUNT=$(find "$N1_DIR" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
if [ "$SUBDIR_COUNT" -eq 0 ]; then
    echo ""
    echo "Error: No subdirectories found in N1 (should contain Base, Instruct, etc.)"
    ls -F "$N1_DIR"
    exit 1
fi

# Check N50 directory (N50 is optional, but provide a hint if found)
N50_DIR="${BASE_DIR}/result/${EXP}/N50"
if [ -d "$N50_DIR" ]; then
    N50_COUNT=$(find "$N50_DIR" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
    echo "Also found N50 data directory, containing ${N50_COUNT} configurations"
fi

echo ""
echo "Preparing to evaluate ${SUBDIR_COUNT} configurations (N1)"

# Build evaluation command
# --mode full: Execute GT merging -> Point-wise evaluation (N1) -> Distribution evaluation (if N50 exists)
CMD="python analysis/evaluators/evaluate.py --mode full --base-dir ${BASE_DIR} --exp ${EXP}"

if [ -n "${MODEL}" ]; then
    CMD="${CMD} --model ${MODEL}"
fi

echo "Executing command: ${CMD}"
echo ""

# Execute evaluation
eval ${CMD}

echo ""
echo "=========================================="
echo "Exp6 evaluation completed"
echo "=========================================="
echo "Results directory: ${BASE_DIR}/result_analysed/${EXP}/"
