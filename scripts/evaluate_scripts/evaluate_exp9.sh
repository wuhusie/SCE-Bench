#!/bin/bash
# ============================================================
# Exp9 Teacher Guidance Mode Evaluation Script (Teacher-Student Evaluation)
#
# Compare the impact of teacher guidance (Teacher vs NoTeacher) on LLM predictions
#
# Usage:
#   bash scripts/evaluate_scripts/evaluate_exp9.sh                    # Evaluate all configurations
#   MODEL=Teacher bash scripts/evaluate_scripts/evaluate_exp9.sh      # Evaluate specific configuration
#
# Prerequisites:
#   Directory structure should be: result/exp9/N1/{Teacher,NoTeacher}/*.csv
# ============================================================

set -e

# Get project root directory (scripts/evaluate_scripts/ -> scripts/ -> project root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

cd "${PROJECT_ROOT}"

# Allow external override of BASE_DIR
BASE_DIR=${BASE_DIR:-"/root/autodl-fs"}
EXP=${EXP:-"exp9"}

echo "=========================================="
echo "Exp9 Teacher Guidance Mode Evaluation (Teacher Study)"
echo "=========================================="
echo "Experiment: ${EXP}"
if [ -n "${MODEL}" ]; then
    echo "Configuration: ${MODEL}"
else
    echo "Configuration: All (Teacher, NoTeacher)"
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
    echo "Expected structure: ${BASE_DIR}/result/exp9/N1/{Teacher,NoTeacher}/*.csv"
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
echo "Exp9 evaluation completed"
echo "=========================================="
echo "Results directory: ${BASE_DIR}/result_analysed/${EXP}/"
