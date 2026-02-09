#!/bin/bash
# ============================================
# ============================================
# Exp9: Human GT Memory N50 Experiment
# ============================================
# Add previous Human GT answer as memory based on the data
# previousGt calculated by shifting data's GT column, no external file needed
# ============================================

# ================== Path Config ==================
PROJECT_ROOT="/root/autodl-tmp"

# ================== Model Config ==================
MODEL_KEY="4"  # MODEL_REGISTRY key: 4 = Qwen3-30B-A3B-FP8
MODEL_NAME="Qwen3-30B-A3B-Instruct-2507-FP8"

# ================== Experiment Config ==================
PROVIDER="local_vllm"
# THINK_MODE="think"
N_SAMPLES=50

# ================== Exp9 Config ==================
# Experiment Types: exp9_labor, exp9_credit, exp9_spending
EXPERIMENTS="exp9_labor exp9_credit exp9_spending"

# ================== Output Config ==================
OUTPUT_DIR="/root/autodl-fs/result/exp9/N50/Teacher"

# ================== tmux Config ==================
SESSION_NAME="exp9_n50_qwen3"

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
        echo "🔬 Running Experiment: $exp"
        echo "=========================================="

        python src/analysis/expN50/run_N50.py \
            --experiment "$exp" \
            --provider "$PROVIDER" \
            --model "$MODEL_NAME" \
            ${THINK_MODE:+--think-mode "$THINK_MODE"} \
            --n-samples "$N_SAMPLES" \
            --output-dir "$OUTPUT_DIR" \
            2>&1 | tee -a "$LOG_FILE"
    done
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
echo "🚀 Exp9: Human GT Memory N50 Experiment"
echo "=============================================="
echo "📦 Model: $MODEL_NAME (key=$MODEL_KEY)"
echo "🧪 Experiment Type: $EXPERIMENTS"
echo "🧠 Think Mode: ${THINK_MODE:-Disabled}"
echo "📊 N Samples: $N_SAMPLES"
echo "📂 Output Dir: $OUTPUT_DIR"
echo "📝 Log File: $LOG_FILE"
echo "=============================================="
echo ""
echo "Creating tmux session..."

THINK_OPT=""
if [ -n "$THINK_MODE" ]; then
    THINK_OPT="--think-mode $THINK_MODE"
fi

# Build run command
RUN_EXPERIMENTS_CMD=""
for exp in $EXPERIMENTS; do
    RUN_EXPERIMENTS_CMD="$RUN_EXPERIMENTS_CMD
echo '' && \
echo '==========================================' && \
echo '🔬 Running Experiment: $exp' && \
echo '==========================================' && \
python src/analysis/expN50/run_N50.py \
    --experiment $exp \
    --provider $PROVIDER \
    --model $MODEL_NAME \
    $THINK_OPT \
    --n-samples $N_SAMPLES \
    --output-dir $OUTPUT_DIR \
    2>&1 | tee -a $LOG_FILE && "
done
# Remove trailing " && "
RUN_EXPERIMENTS_CMD="${RUN_EXPERIMENTS_CMD% && }"

# Left Pane: Command to start model
LEFT_CMD="cd $PROJECT_ROOT && echo '🚀 Starting Model: $MODEL_NAME' && python src/server/launch_model.py $MODEL_KEY"

# Right Pane: Wait for model ready then run experiment
RIGHT_CMD="cd $PROJECT_ROOT && \
echo '⏳ Waiting for vLLM service...' && \
export NO_PROXY=localhost,127.0.0.1 && \
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY && \
while ! curl -s --max-time 2 http://localhost:8000/v1/models > /dev/null; do \
    sleep 3; \
    echo '   Still waiting... (If no response for long time, check left pane for errors)'; \
done && \
echo '✅ vLLM Service Ready!' && \
sleep 2 && \
echo '🔬 Starting Exp9 N50 Experiment...' \
$RUN_EXPERIMENTS_CMD"

# Create tmux session and split
tmux new-session -d -s "$SESSION_NAME" -x 200 -y 50

# Run model in left pane
tmux send-keys -t "$SESSION_NAME" "$LEFT_CMD" C-m

# Split horizontally (left/right)
tmux split-window -h -t "$SESSION_NAME"

# Run experiment in right pane
tmux send-keys -t "$SESSION_NAME" "$RIGHT_CMD" C-m

# Attach to session
echo "📺 Opening tmux session..."
echo "   Left: Model Service"
echo "   Right: Experiment Process"
echo ""
echo "Tip: Press Ctrl+B then D to detach (run in background)"
tmux attach-session -t "$SESSION_NAME"
