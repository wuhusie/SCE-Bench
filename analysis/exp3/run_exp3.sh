#!/bin/bash

# Define paths matching the environment structure
SRC_DIR="/root/autodl-tmp/src"
EXP3_DIR="${SRC_DIR}/analysis/exp3"
CONFIG_FILE="${EXP3_DIR}/batch_config_exp3.yaml"
MODIFY_SCRIPT="${EXP3_DIR}/modify_prompts.py"
RUN_BATCH_SCRIPT="${SRC_DIR}/run_batch.sh"

# Files to manage
MAIN_PY="${SRC_DIR}/sce/main.py"
CREDIT_PY="${SRC_DIR}/sce/experiments/credit.py"
LABOR_PY="${SRC_DIR}/sce/experiments/labor.py"
SPENDING_PY="${SRC_DIR}/sce/experiments/spending.py"

# Backup paths
MAIN_BAK="${MAIN_PY}.bak"
CREDIT_BAK="${CREDIT_PY}.bak"
LABOR_BAK="${LABOR_PY}.bak"
SPENDING_BAK="${SPENDING_PY}.bak"

# Function to restore files on exit
cleanup() {
    echo "🧹 Restoring original files..."
    
    if [ -f "$MAIN_BAK" ]; then mv "$MAIN_BAK" "$MAIN_PY"; fi
    if [ -f "$CREDIT_BAK" ]; then mv "$CREDIT_BAK" "$CREDIT_PY"; fi
    if [ -f "$LABOR_BAK" ]; then mv "$LABOR_BAK" "$LABOR_PY"; fi
    if [ -f "$SPENDING_BAK" ]; then mv "$SPENDING_BAK" "$SPENDING_PY"; fi
    
    echo "✅ Restore completed."
}
trap cleanup EXIT

echo "🚀 Starting Exp 3 (Third Person Prompts)"
echo "Config: $CONFIG_FILE"

# 1. Backup files
echo "💾 Backing up source files..."
cp "$MAIN_PY" "$MAIN_BAK"
cp "$CREDIT_PY" "$CREDIT_BAK"
cp "$LABOR_PY" "$LABOR_BAK"
cp "$SPENDING_PY" "$SPENDING_BAK"

# 2. Modify Prompts
echo "📝 Injecting Third-Person Prompts..."
python3 "$MODIFY_SCRIPT"

if [ $? -ne 0 ]; then
    echo "❌ Prompt modification failed! Aborting."
    exit 1
fi

# 3. Execute run_batch.sh
echo "▶️  Executing batch runner..."
bash "$RUN_BATCH_SCRIPT" "$CONFIG_FILE"

echo "✅ Exp 3 pipeline finished."
# cleanup() will automatically run here
