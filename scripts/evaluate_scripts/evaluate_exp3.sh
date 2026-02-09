#!/bin/bash
# Exp3 Perspective Evaluation Script
#
# Compare the impact of different perspectives (2p vs 3p) on LLM predictions
#
# Usage:
#   bash scripts/evaluate_scripts/evaluate_exp3.sh                    # Evaluate all configurations (2p and 3p)
#   MODEL=2p bash scripts/evaluate_scripts/evaluate_exp3.sh           # Evaluate specific configuration
#
# Prerequisites:
#   Directory structure should be: result/exp3/N1/{2p,3p}/*.csv
# ============================================================

set -e

# Get project root directory (scripts/evaluate_scripts/ -> scripts/ -> project root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

cd "${PROJECT_ROOT}"

# Allow external override of BASE_DIR
BASE_DIR=${BASE_DIR:-"/root/autodl-fs"}
EXP=${EXP:-"exp3"}

echo "=========================================="
echo "Exp3 Perspective Evaluation (Perspective Study)"
echo "=========================================="
echo "Experiment: ${EXP}"
if [ -n "${MODEL}" ]; then
    echo "Configuration: ${MODEL}"
else
    echo "Configuration: All (2p, 3p)"
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
    echo "Expected structure: ${BASE_DIR}/result/exp3/N1/{2p,3p}/*.csv"
    
    # Simple error message
    echo ""
    echo "If your data is in result/exp3 but without N1 subdirectory,"
    echo "please manually move or adjust the directory structure."
    exit 1
fi

# Check for subdirectories (2p/3p)
SUBDIR_COUNT=$(find "$N1_DIR" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
if [ "$SUBDIR_COUNT" -eq 0 ]; then
    echo ""
    echo "Error: No subdirectories found in N1 (should contain 2p, 3p, etc.)"
    echo "Current N1 content:"
    ls -F "$N1_DIR"
    exit 1
fi

echo ""
echo "Found ${SUBDIR_COUNT} configurations (2p/3p)"

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
echo "Exp3 evaluation completed"
echo "=========================================="
echo "Results directory: ${BASE_DIR}/result_analysed/${EXP}/"
