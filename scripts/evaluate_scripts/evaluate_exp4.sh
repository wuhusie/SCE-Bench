#!/bin/bash
# ============================================================
# Exp4 Prompt Injection Evaluation Script
#
# Compare the impact of different prompt formats (Json-Language vs Nature-Language) on LLM predictions
#
# Usage:
#   bash scripts/evaluate_scripts/evaluate_exp4.sh                    # Evaluate all configurations (Json-Language and Nature-Language)
#   MODEL=Json-Language bash scripts/evaluate_scripts/evaluate_exp4.sh           # Evaluate specific configuration
#
# Prerequisites:
#   Directory structure should be: result/exp4/N1/{Json-Language,Nature-Language}/*.csv
# ============================================================

set -e

# Get project root directory (scripts/evaluate_scripts/ -> scripts/ -> project root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

cd "${PROJECT_ROOT}"

# Allow external override of BASE_DIR
BASE_DIR=${BASE_DIR:-"/root/autodl-fs"}
EXP=${EXP:-"exp4"}

echo "=========================================="
echo "Exp4 Prompt Injection Evaluation (Prompt Injection Study)"
echo "=========================================="
echo "Experiment: ${EXP}"
if [ -n "${MODEL}" ]; then
    echo "Configuration: ${MODEL}"
else
    echo "Configuration: All (Json-Language, Nature-Language)"
fi
echo "Project Path: ${PROJECT_ROOT}"
echo "Data Path: ${BASE_DIR}/result/${EXP}"
echo "=========================================="

# Check directory structure
N1_DIR="${BASE_DIR}/result/${EXP}/N1"

# If N1 directory does not exist, check if it's still in root
if [ ! -d "$N1_DIR" ]; then
    echo ""
    echo "Warning: N1 directory does not exist: ${N1_DIR}"
    echo "Please confirm if run_batch.sh results have been correctly generated."
    echo "Expected structure: ${BASE_DIR}/result/exp4/N1/{Json-Language,Nature-Language}/*.csv"
    
    # Simple error message
    echo ""
    echo "If your data is in result/exp4 but without N1 subdirectory,"
    echo "please manually move or adjust the directory structure."
    exit 1
fi

# Check for subdirectories
SUBDIR_COUNT=$(find "$N1_DIR" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
if [ "$SUBDIR_COUNT" -eq 0 ]; then
    echo ""
    echo "Error: No subdirectories found in N1 (should contain Json-Language, Nature-Language, etc.)"
    echo "Current N1 content:"
    ls -F "$N1_DIR"
    exit 1
fi

echo ""
echo "Found ${SUBDIR_COUNT} configurations"

# Build evaluation command
# --mode full: Execute GT merging -> Point-wise evaluation -> Distribution evaluation (if N50 exists)
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
echo "Exp4 evaluation completed"
echo "=========================================="
echo "Results directory: ${BASE_DIR}/result_analysed/${EXP}/"
