#!/bin/bash

# Define paths
SRC_DIR="/root/autodl-tmp/src"
CONFIG_FILE="${SRC_DIR}/analysis/exp2/batch_config_ablation.yaml"
RUN_BATCH_SCRIPT="${SRC_DIR}/run_batch.sh"

# Note: Ensure 'conda activate acl' is run before this script, or handled by the parent shell.
# run_batch.sh handles internal environment activation for tmux sessions.

echo "🚀 Starting Ablation Study (Exp 2)"
echo "Config: $CONFIG_FILE"

# Execute run_batch.sh with the custom config
bash "$RUN_BATCH_SCRIPT" "$CONFIG_FILE"
