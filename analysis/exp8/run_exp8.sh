#!/bin/bash
# ==============================================================================
# 实验8：模型回答记忆 (LLM Response Memory)
#
# 用法:
#   bash src/analysis/exp8/run_exp8.sh
#
# 前置条件:
#   必须先跑完 exp1-1 全量实验，结果文件位于:
#   /root/autodl-fs/result/exp1.1/
#
# 说明:
#   直接调用 run_batch.sh 执行 batch_config_exp8.yaml 中定义的任务。
# ==============================================================================

SRC_DIR="/root/autodl-tmp/src"
EXP8_DIR="${SRC_DIR}/analysis/exp8"
CONFIG_FILE="${EXP8_DIR}/batch_config_exp8.yaml"
RUN_BATCH_SCRIPT="${SRC_DIR}/run_batch.sh"

echo "🚀 Starting Exp 8 (LLM Response Memory)"
echo "Config: $CONFIG_FILE"
echo ""

# 检查第一轮结果是否存在
PRIOR_DIR="/root/autodl-fs/result/exp1.1"
if [ ! -d "$PRIOR_DIR" ]; then
    echo "❌ 错误: 第一轮实验结果目录不存在: $PRIOR_DIR"
    echo "   请先运行 exp1-1 全量实验"
    exit 1
fi

# 执行批量运行器
bash "$RUN_BATCH_SCRIPT" "$CONFIG_FILE"

echo "✅ Exp 8 pipeline finished."
