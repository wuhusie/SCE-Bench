#!/bin/bash
# ============================================================
# Exp5 Chain-of-Thought Evaluation Script
#
# Compare the impact of enabling/disabling Thinking Mode (Think vs NoThink) on LLM predictions
#
# Usage:
#   bash scripts/evaluate_scripts/evaluate_exp5.sh                    # Evaluate all configurations (Think and NoThink)
#   MODEL=Think bash scripts/evaluate_scripts/evaluate_exp5.sh        # Evaluate specific configuration
#
# Prerequisites:
#   Directory structure should be: result/exp5/N1/{Think,NoThink}/*.csv
# ============================================================

set -e

# Get project root directory (scripts/evaluate_scripts/ -> scripts/ -> project root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

cd "${PROJECT_ROOT}"

# Allow external override of BASE_DIR
BASE_DIR=${BASE_DIR:-"/root/autodl-fs"}
EXP=${EXP:-"exp5"}

echo "=========================================="
echo "Exp5 Thinking Mode Impact Evaluation (Thinking Mode Study)"
echo "=========================================="
echo "Experiment: ${EXP}"
if [ -n "${MODEL}" ]; then
    echo "Configuration: ${MODEL}"
else
    echo "Configuration: All (Think, NoThink)"
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
    echo "Expected structure: ${BASE_DIR}/result/exp5/N1/{Think,NoThink}/*.csv"
    
    # Simple error message
    echo ""
    echo "If your data is in result/exp5 but without N1 subdirectory,"
    echo "please manually move or adjust the directory structure."
    exit 1
fi

# Check for subdirectories
SUBDIR_COUNT=$(find "$N1_DIR" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
if [ "$SUBDIR_COUNT" -eq 0 ]; then
    echo ""
    echo "Error: No subdirectories found in N1 (should contain Think, NoThink, etc.)"
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
echo "Exp5 evaluation completed"
echo "=========================================="
echo "Results directory: ${BASE_DIR}/result_analysed/${EXP}/"
