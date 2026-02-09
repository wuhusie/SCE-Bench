#!/bin/bash
# ============================================
# ============================================
# Exp4: JSON Structured Prompts N50 Experiment
# ============================================
# Convert Profile and Environment from Natural Language to JSON Format
# Use runtime code injection, automatically restore after experiment
# ============================================

# ================== Path Config ==================
PROJECT_ROOT="/root/autodl-tmp"
SRC_DIR="${PROJECT_ROOT}/src"
EXP4_DIR="${SRC_DIR}/analysis/exp4"
EXPN50_DIR="${SRC_DIR}/analysis/expN50"
MODIFY_SCRIPT="${EXP4_DIR}/modify_prompts_N50.py"

# ================== Model Config ==================
MODEL_KEY="4"
MODEL_NAME="Qwen3-30B-A3B-Instruct-2507-FP8"

# ================== Experiment Config ==================
PROVIDER="local_vllm"
# THINK_MODE="think"
N_SAMPLES=50
SAMPLE_RATIO=0.1  # 10% Sampling
SEED=2026
EXPERIMENTS="labor credit spending"

# ================== Output Config ==================
OUTPUT_DIR="/root/autodl-fs/result/exp4/N50/Json-Language"

# ================== tmux Config ==================
SESSION_NAME="exp4_n50_qwen3"

# ================== Files to Backup ==================
PROMPT_UTILS_PY="${SRC_DIR}/sce/utils/prompt_utils.py"
PROMPT_UTILS_BAK="${PROMPT_UTILS_PY}.bak"

# ============================================
# ============================================
# Cleanup Function (Restore original files)
# ============================================
cleanup() {
    echo ""
    echo "🧹 Restoring original files..."

    if [ -f "$PROMPT_UTILS_BAK" ]; then
        mv "$PROMPT_UTILS_BAK" "$PROMPT_UTILS_PY" && echo "  ✅ prompt_utils.py"
    fi

    echo "✅ Restore completed."
}
trap cleanup EXIT

# ============================================
# ============================================
# Main Logic
# ============================================

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Log file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$OUTPUT_DIR/run_${TIMESTAMP}.log"

echo "=============================================="
echo "🚀 Exp4: JSON Structured Prompts N50 Experiment"
echo "=============================================="
echo "📦 Model: $MODEL_NAME"
echo "🧪 Experiment: $EXPERIMENTS"
echo "📊 N Samples: $N_SAMPLES"
echo "🎲 Sample Ratio: $SAMPLE_RATIO (Seed: $SEED)"
echo "📂 Output Dir: $OUTPUT_DIR"
echo "📝 Log File: $LOG_FILE"
echo "=============================================="

# 1. Backup files
echo ""
echo "💾 Backing up source files..."
cp "$PROMPT_UTILS_PY" "$PROMPT_UTILS_BAK" && echo "  ✅ prompt_utils.py"

# 2. Inject JSON Format Prompts
echo ""
echo "📝 Injecting JSON Structured Prompts..."
python3 "$MODIFY_SCRIPT"

if [ $? -ne 0 ]; then
    echo "❌ Prompt modification failed! Aborting."
    exit 1
fi

# 3. If already in tmux, run experiment directly
if [ -n "$TMUX" ]; then
    echo ""
    echo "Detected inside tmux, running experiment directly..."
    cd "$PROJECT_ROOT"

    THINK_OPT=""
    if [ -n "$THINK_MODE" ]; then
        THINK_OPT="--think-mode $THINK_MODE"
    fi

    for exp in $EXPERIMENTS; do
        echo ""
        echo "=========================================="
        echo "🔬 Running Experiment: $exp (JSON Format)"
        echo "=========================================="

        python src/analysis/expN50/run_N50.py \
            --experiment "$exp" \
            --provider "$PROVIDER" \
            --model "$MODEL_NAME" \
            $THINK_OPT \
            --n-samples "$N_SAMPLES" \
            --sample-ratio "$SAMPLE_RATIO" \
            --seed "$SEED" \
            --output-dir "$OUTPUT_DIR" \
            2>&1 | tee -a "$LOG_FILE"
    done

    echo ""
    echo "=========================================="
    echo "✅ Exp4 N50 Experiment Completed!"
    echo "=========================================="
    exit 0
fi

# 4. Non-tmux mode: Create tmux session
if ! command -v tmux &> /dev/null; then
    echo "❌ Error: tmux not installed"
    echo "   Please run: apt install tmux"
    exit 1
fi

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "📺 tmux session '$SESSION_NAME' exists, attaching..."
    tmux attach-session -t "$SESSION_NAME"
    exit 0
fi

echo ""
echo "Creating tmux session..."

# Left Pane: Start Model
LEFT_CMD="cd $PROJECT_ROOT && echo '🚀 Starting Model: $MODEL_NAME' && python src/server/launch_model.py $MODEL_KEY"

# Right Pane: Wait for model ready then run experiment
RIGHT_CMD="cd $PROJECT_ROOT && \
echo '⏳ Waiting for vLLM service...' && \
export NO_PROXY=localhost,127.0.0.1 && \
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY && \
while ! curl -s --max-time 2 http://localhost:8000/v1/models > /dev/null; do \
    sleep 3; \
    echo '   Still waiting...'; \
done && \
echo '✅ vLLM Service Ready!' && \
sleep 2 && \
echo '🔬 Keying Exp4 N50 Experiment...' && \
bash src/analysis/expN50/run_exp4_N50.sh"

# Create tmux session
tmux new-session -d -s "$SESSION_NAME" -x 200 -y 50
tmux send-keys -t "$SESSION_NAME" "$LEFT_CMD" C-m
tmux split-window -h -t "$SESSION_NAME"
tmux send-keys -t "$SESSION_NAME" "$RIGHT_CMD" C-m

echo "📺 Opening tmux session..."
echo "   Left: Model Service"
echo "   Right: Experiment Process"
echo ""
echo "⚠️  Note: Original files will be restored automatically after experiment"
echo "Tip: Press Ctrl+B then D to detach (run in background)"
tmux attach-session -t "$SESSION_NAME"
