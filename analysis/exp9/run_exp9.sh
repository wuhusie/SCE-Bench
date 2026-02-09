#!/bin/bash
# ==============================================================================
# 实验9：真实人类回答记忆 (Real Human Answer Memory)
#
# 用法:
#   bash src/analysis/exp9/run_exp9.sh
#
# 说明:
#   直接调用 run_batch.sh 执行 batch_config_exp9.yaml 中定义的任务。
#   无需额外的 prompt 注入，所有逻辑通过 Exp9Mixin 在框架内实现。
# ==============================================================================

SRC_DIR="/root/autodl-tmp/src"
EXP9_DIR="${SRC_DIR}/analysis/exp9"
CONFIG_FILE="${EXP9_DIR}/batch_config_exp9.yaml"
RUN_BATCH_SCRIPT="${SRC_DIR}/run_batch.sh"

echo "🚀 Starting Exp 9 (Real Human Answer Memory)"
echo "Config: $CONFIG_FILE"
echo ""

# 执行批量运行器
bash "$RUN_BATCH_SCRIPT" "$CONFIG_FILE"

echo "✅ Exp 9 pipeline finished."