#!/bin/bash
# Exp1.3 Batch Run Script - Unattended Mode
# Run 4 models: gpt_5_mini, gpt_5_mini_medium, gemini_3_flash_nothinking, gemini_3_flash_thinking

set -e  # Use set +e to continue on error, set -e to stop on error

# Configuration
BASE_OUTPUT_DIR="/root/autodl-fs/result/exp1/N50"
N_SAMPLES=50
EXPERIMENTS="labor credit spending"

# Model List
MODELS=(
    # "gpt_5_mini_minimal"
    # "gpt_5_mini_medium"
    # "gemini_3_flash_nothinking"
    # "gemini_3_flash_thinking"
    "gpt-3.5-turbo"
)

# Switch to script directory
cd "$(dirname "$0")"

echo "=============================================="
echo "Exp1.3 Batch Run - Start Time: $(date)"
echo "=============================================="

for model in "${MODELS[@]}"; do
    echo ""
    echo "======================================================"
    echo ">>> Start Running Model: $model"
    echo ">>> Time: $(date)"
    echo "======================================================"

    OUTPUT_DIR="${BASE_OUTPUT_DIR}/${model}"
    mkdir -p "$OUTPUT_DIR"

    LOG_FILE="${OUTPUT_DIR}/$(date +%Y_%m%d_%H%M)_exp1_N50_${model}.log"

    python run_N50.py \
        --provider "$model" \
        --n-samples $N_SAMPLES \
        --output-dir "$OUTPUT_DIR" \
        --experiment $EXPERIMENTS \
        2>&1 | tee -a "$LOG_FILE"

    echo ">>> Model $model Completed, Time: $(date)"
done

echo ""
echo "=============================================="
echo "All Completed! End Time: $(date)"
echo "=============================================="
