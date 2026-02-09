#!/bin/bash
# ============================================
# ============================================
# Exp5: Think Mode N50 Experiment
# ============================================
# Evaluate impact of reasoning mode (think_mode) on model predictions
# Use Qwen3-30B-A3B-FP8 (Base) + Chain of Thought enabled
# ============================================

# ================== Path Config ==================
PROJECT_ROOT="/root/autodl-tmp"

# ================== Model Config ==================
MODEL_KEY="3"  # Qwen3-30B-A3B-FP8 (Base)
MODEL_NAME="Qwen3-30B-A3B-FP8"

# ================== Experiment Config ==================
PROVIDER="local_vllm"
THINK_MODE="think"  # Enable Chain of Thought
N_SAMPLES=50
SAMPLE_RATIO=0.1  # 10% Sampling
SEED=2026
EXPERIMENTS="labor credit spending"

# ================== Output Config ==================
# Output to Think subdir for comparison with NoThink
OUTPUT_DIR="/root/autodl-fs/result/exp5/N50/Think"

# ================== tmux Config ==================
SESSION_NAME="exp5_n50_think"

# ============================================
# ============================================
# Main Logic
# ============================================

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Log file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$OUTPUT_DIR/run_${TIMESTAMP}.log"

# If already in tmux, run experiment directly
if [ -n "$TMUX" ]; then
    echo "Detected inside tmux, running experiment directly..."
    cd "$PROJECT_ROOT"

    for exp in $EXPERIMENTS; do
        echo ""
        echo "=========================================="
        echo "🔬 Running Experiment: $exp (Think Mode)"
        echo "=========================================="

        python src/analysis/expN50/run_N50.py \
            --experiment "$exp" \
            --provider "$PROVIDER" \
            --model "$MODEL_NAME" \
            --think-mode "$THINK_MODE" \
            --n-samples "$N_SAMPLES" \
            --sample-ratio "$SAMPLE_RATIO" \
            --seed "$SEED" \
            --output-dir "$OUTPUT_DIR" \
            2>&1 | tee -a "$LOG_FILE"
    done

    echo ""
    echo "=========================================="
    echo "✅ Exp5 N50 (Think Mode) Experiment Completed!"
    echo "=========================================="
    exit 0
fi

# Check if tmux is installed
if ! command -v tmux &> /dev/null; then
    echo "❌ Error: tmux not installed"
    echo "   Please run: apt install tmux"
    exit 1
fi

# If session exists, attach to it
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "📺 tmux session '$SESSION_NAME' exists, attaching..."
    tmux attach-session -t "$SESSION_NAME"
    exit 0
fi

echo "=============================================="
echo "🚀 Exp5: Think Mode N50 Experiment"
echo "=============================================="
echo "📦 Model: $MODEL_NAME (key=$MODEL_KEY)"
echo "🧠 Think Mode: $THINK_MODE"
echo "🧪 Experiment: $EXPERIMENTS"
echo "📊 N Samples: $N_SAMPLES"
echo "🎲 Sample Ratio: $SAMPLE_RATIO (Seed: $SEED)"
echo "📂 Output Dir: $OUTPUT_DIR"
echo "📝 Log File: $LOG_FILE"
echo "=============================================="
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
echo '🔬 Starting Exp5 N50 Experiment (Think Mode)...' && \
bash src/analysis/expN50/run_exp5_N50.sh"

# Create tmux session
tmux new-session -d -s "$SESSION_NAME" -x 200 -y 50
tmux send-keys -t "$SESSION_NAME" "$LEFT_CMD" C-m
tmux split-window -h -t "$SESSION_NAME"
tmux send-keys -t "$SESSION_NAME" "$RIGHT_CMD" C-m

echo "📺 Opening tmux session..."
echo "   Left: Model Service"
echo "   Right: Experiment Process"
echo ""
echo "Tip: Press Ctrl+B then D to detach (run in background)"
tmux attach-session -t "$SESSION_NAME"
