#!/bin/bash
# ============================================================
# Exp8 Memory Mechanism Evaluation Script
#
# Compare the impact of memory enhancement (Memory vs NoMemory) on LLM predictions
#
# Usage:
#   bash scripts/evaluate_scripts/evaluate_exp8.sh                    # Evaluate all configurations
#   MODEL=Memory bash scripts/evaluate_scripts/evaluate_exp8.sh       # Evaluate specific configuration
#
# Prerequisites:
#   Directory structure should be: result/exp8/N1/{Memory,NoMemory}/*.csv
# ============================================================

set -e

# Get project root directory (scripts/evaluate_scripts/ -> scripts/ -> project root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

cd "${PROJECT_ROOT}"

# Allow external override of BASE_DIR
BASE_DIR=${BASE_DIR:-"/root/autodl-fs"}
EXP=${EXP:-"exp8"}

echo "=========================================="
echo "Exp8 Memory Mechanism Evaluation (Memory Study)"
echo "=========================================="
echo "Experiment: ${EXP}"
if [ -n "${MODEL}" ]; then
    echo "Configuration: ${MODEL}"
else
    echo "Configuration: All (Memory, NoMemory)"
fi
echo "Project Path: ${PROJECT_ROOT}"
echo "Data Path: ${BASE_DIR}/result/${EXP}"
echo "=========================================="

# Check directory structure
N1_DIR="${BASE_DIR}/result/${EXP}/N1"

if [ ! -d "$N1_DIR" ]; then
    echo ""
    echo "Warning: N1 directory does not exist: ${N1_DIR}"
    echo "Please confirm if run_batch.sh results have been correctly generated."
    echo "Expected structure: ${BASE_DIR}/result/exp8/N1/{Memory,NoMemory}/*.csv"
    exit 1
fi

SUBDIR_COUNT=$(find "$N1_DIR" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
if [ "$SUBDIR_COUNT" -eq 0 ]; then
    echo ""
    echo "Error: No subdirectories found in N1"
    ls -F "$N1_DIR"
    exit 1
fi

echo ""
echo "Found ${SUBDIR_COUNT} configurations"

CMD="python analysis/evaluators/evaluate.py --mode full --base-dir ${BASE_DIR} --exp ${EXP}"

if [ -n "${MODEL}" ]; then
    CMD="${CMD} --model ${MODEL}"
fi

echo "Executing command: ${CMD}"
echo ""

eval ${CMD}

echo ""
echo "=========================================="
echo "Exp8 evaluation completed"
echo "=========================================="
echo "Results directory: ${BASE_DIR}/result_analysed/${EXP}/"
