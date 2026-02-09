#!/bin/bash
# ============================================
# ============================================
# Exp2: Feature Ablation N50 Experiment
# ============================================
# Leave-One-Out Feature Ablation Experiment
# 3 Tasks × 13 Features = 39 Experiment Combinations
# ============================================

# ================== Path Config ==================
PROJECT_ROOT="/root/autodl-tmp"

# ================== Model Config ==================
MODEL_KEY="4"
MODEL_NAME="Qwen3-30B-A3B-Instruct-2507-FP8"

# ================== Experiment Config ==================
PROVIDER="local_vllm"
# THINK_MODE="think"
N_SAMPLES=50
SAMPLE_RATIO=0.1  # 10% Sampling
SEED=2026

# ================== Output Config ==================
OUTPUT_BASE="/root/autodl-fs/result/exp2/N50"

# ================== tmux Config ==================
SESSION_NAME="exp2_n50_qwen3"

# ================== Feature Definition ==================
# Profile Features: 1-10
# 1=age, 2=gender, 3=education, 4=marital_status, 5=state_residence
# 6=housing_status, 7=own_other_home, 8=health_status, 9=employment_status, 10=income

# Environment Features: 1-3
# 1=inflation, 2=unemployment, 3=interest_rate

TASKS="labor credit spending"
PROFILE_FEATURES="1 2 3 4 5 6 7 8 9 10"
ENV_FEATURES="1 2 3"

# Feature name maps for output naming
declare -A PROFILE_MAP=(
    [1]="age" [2]="gender" [3]="education" [4]="marital_status" [5]="state_residence"
    [6]="housing_status" [7]="own_other_home" [8]="health_status" [9]="employment_status" [10]="income"
)
declare -A ENV_MAP=(
    [1]="inflation" [2]="unemployment" [3]="interest_rate"
)

# ============================================
# ============================================
# Helper Functions
# ============================================

# Generate feature list excluding a specific feature
get_profile_without() {
    local exclude=$1
    local result=""
    for f in $PROFILE_FEATURES; do
        if [ "$f" != "$exclude" ]; then
            [ -n "$result" ] && result="$result,"
            result="$result$f"
        fi
    done
    echo "$result"
}

get_env_without() {
    local exclude=$1
    local result=""
    for f in $ENV_FEATURES; do
        if [ "$f" != "$exclude" ]; then
            [ -n "$result" ] && result="$result,"
            result="$result$f"
        fi
    done
    echo "$result"
}

# ============================================
# ============================================
# Main Logic
# ============================================

# Create output directory
mkdir -p "$OUTPUT_BASE"

# Log file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$OUTPUT_BASE/run_${TIMESTAMP}.log"

# If already in tmux, run experiment directly
if [ -n "$TMUX" ]; then
    echo "Detected inside tmux, running experiment directly..."
    cd "$PROJECT_ROOT"

    THINK_OPT=""
    if [ -n "$THINK_MODE" ]; then
        THINK_OPT="--think-mode $THINK_MODE"
    fi

    # Iterate over all tasks
    for task in $TASKS; do
        # 1. Profile Feature Ablation (Remove 1 profile feature)
        for fid in $PROFILE_FEATURES; do
            feat_name="${PROFILE_MAP[$fid]}"
            profile_subset=$(get_profile_without $fid)
            output_dir="$OUTPUT_BASE/NO${feat_name}"
            mkdir -p "$output_dir"

            echo ""
            echo "=========================================="
            echo "🔬 $task - NO${feat_name}"
            echo "   Profile: $profile_subset"
            echo "   Output: $output_dir"
            echo "=========================================="

            python src/analysis/expN50/run_N50.py \
                --experiment "$task" \
                --provider "$PROVIDER" \
                --model "$MODEL_NAME" \
                $THINK_OPT \
                --n-samples "$N_SAMPLES" \
                --sample-ratio "$SAMPLE_RATIO" \
                --seed "$SEED" \
                --output-dir "$output_dir" \
                --profile-features "$profile_subset" \
                --env-features "all" \
                2>&1 | tee -a "$LOG_FILE"
        done

        # 2. Environment Feature Ablation (Remove 1 env feature)
        for fid in $ENV_FEATURES; do
            feat_name="${ENV_MAP[$fid]}"
            env_subset=$(get_env_without $fid)
            output_dir="$OUTPUT_BASE/NO${feat_name}"
            mkdir -p "$output_dir"

            echo ""
            echo "=========================================="
            echo "🔬 $task - NO${feat_name}"
            echo "   Env: $env_subset"
            echo "   Output: $output_dir"
            echo "=========================================="

            python src/analysis/expN50/run_N50.py \
                --experiment "$task" \
                --provider "$PROVIDER" \
                --model "$MODEL_NAME" \
                $THINK_OPT \
                --n-samples "$N_SAMPLES" \
                --sample-ratio "$SAMPLE_RATIO" \
                --seed "$SEED" \
                --output-dir "$output_dir" \
                --profile-features "all" \
                --env-features "$env_subset" \
                2>&1 | tee -a "$LOG_FILE"
        done
    done

    echo ""
    echo "=========================================="
    echo "✅ Exp2 N50 Experiment Completed!"
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
echo "🚀 Exp2: Feature Ablation N50 Experiment"
echo "=============================================="
echo "📦 Model: $MODEL_NAME (key=$MODEL_KEY)"
echo "🧪 Task: $TASKS"
echo "📊 Ablation Combinations: 3 Tasks × 13 Features = 39"
echo "🧠 Think Mode: ${THINK_MODE:-Disabled}"
echo "📊 N Samples: $N_SAMPLES"
echo "🎲 Sample Ratio: $SAMPLE_RATIO (Seed: $SEED)"
echo "📂 Output Dir: $OUTPUT_BASE/NO{feature}/"
echo "📝 Log File: $LOG_FILE"
echo "=============================================="
echo ""
echo "Creating tmux session..."

# Left Pane: Command to start model
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
echo '🔬 Starting Exp2 N50 Experiment...' && \
bash src/analysis/expN50/run_exp2_N50.sh"

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
