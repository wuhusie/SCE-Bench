#!/bin/bash

# Define paths
SRC_DIR="/root/autodl-tmp/src"
EXP4_DIR="${SRC_DIR}/analysis/exp4"
CONFIG_FILE="${EXP4_DIR}/batch_config_exp4.yaml"
MODIFY_SCRIPT="${EXP4_DIR}/modify_prompts.py"
RUN_BATCH_SCRIPT="${SRC_DIR}/run_batch.sh"

# Files to manage
MAIN_PY="${SRC_DIR}/sce/main.py"
PROMPT_UTILS_PY="${SRC_DIR}/sce/utils/prompt_utils.py"
CREDIT_PY="${SRC_DIR}/sce/experiments/credit.py"
LABOR_PY="${SRC_DIR}/sce/experiments/labor.py"
SPENDING_PY="${SRC_DIR}/sce/experiments/spending.py"

# Backup paths
MAIN_BAK="${MAIN_PY}.bak"
PROMPT_UTILS_BAK="${PROMPT_UTILS_PY}.bak"
CREDIT_BAK="${CREDIT_PY}.bak"
LABOR_BAK="${LABOR_PY}.bak"
SPENDING_BAK="${SPENDING_PY}.bak"

# Function to restore files on exit
cleanup() {
    echo "🧹 Restoring original files..."
    
    if [ -f "$MAIN_BAK" ]; then mv "$MAIN_BAK" "$MAIN_PY"; fi
    if [ -f "$PROMPT_UTILS_BAK" ]; then mv "$PROMPT_UTILS_BAK" "$PROMPT_UTILS_PY"; fi
    if [ -f "$CREDIT_BAK" ]; then mv "$CREDIT_BAK" "$CREDIT_PY"; fi
    if [ -f "$LABOR_BAK" ]; then mv "$LABOR_BAK" "$LABOR_PY"; fi
    if [ -f "$SPENDING_BAK" ]; then mv "$SPENDING_BAK" "$SPENDING_PY"; fi
    
    echo "✅ Restore completed."
}
trap cleanup EXIT

echo "🚀 Starting Exp 4 (JSON Prompts)"
echo "Config: $CONFIG_FILE"

# 1. Backup files
echo "💾 Backing up source files..."
if [ ! -f "$MAIN_BAK" ]; then cp "$MAIN_PY" "$MAIN_BAK"; fi
if [ ! -f "$PROMPT_UTILS_BAK" ]; then cp "$PROMPT_UTILS_PY" "$PROMPT_UTILS_BAK"; fi
if [ ! -f "$CREDIT_BAK" ]; then cp "$CREDIT_PY" "$CREDIT_BAK"; fi
if [ ! -f "$LABOR_BAK" ]; then cp "$LABOR_PY" "$LABOR_BAK"; fi
if [ ! -f "$SPENDING_BAK" ]; then cp "$SPENDING_PY" "$SPENDING_BAK"; fi

# 2. Modify Prompts
echo "📝 Injecting JSON Prompts..."
python3 "$MODIFY_SCRIPT"

if [ $? -ne 0 ]; then
    echo "❌ Prompt modification failed! Aborting."
    exit 1
fi

# 3. Execute run_batch.sh
echo "▶️  Executing batch runner..."
bash "$RUN_BATCH_SCRIPT" "$CONFIG_FILE"

echo "✅ Exp 4 pipeline finished."
# cleanup() will automatically run here
