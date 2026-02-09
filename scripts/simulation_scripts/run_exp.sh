#!/bin/bash

# =============================================================================
# One-click Experiment Script (Supports SCE / CEX datasets, local vLLM / API, Qwen3 thinking mode)
# =============================================================================

SESSION_NAME="exp"
CONDA_ENV="/root/autodl-tmp/agent_env"
SRC_DIR="/root/autodl-tmp/src"
LOG_DIR="${SRC_DIR}/log"
SERVER_LOG_DIR="${LOG_DIR}/server"

# SCE / CEX experiment list
SCE_EXPERIMENTS="credit spending labor"
CEX_EXPERIMENTS="living_cost"

mkdir -p "$SERVER_LOG_DIR"

# [Important] Disable local proxy to prevent 503 error when connecting to vLLM
export no_proxy="localhost,127.0.0.1,0.0.0.0"

# =============================================================================
# Model Number -> Name Mapping (consistent with launch_model.py MODEL_REGISTRY)
# =============================================================================
declare -A MODEL_NAMES
MODEL_NAMES[1]="Meta-Llama-3.1-70B-bnb-4bit"
MODEL_NAMES[2]="Meta-Llama-3.1-70B-Instruct-bnb-4bit"
MODEL_NAMES[3]="Qwen3-30B-A3B-FP8"
MODEL_NAMES[4]="Qwen3-30B-A3B-Instruct-2507-FP8"
MODEL_NAMES[5]="Qwen3-0.6B"
MODEL_NAMES[6]="Meta-Llama-3.1-70B-Instruct-GPTQ-INT4"

# =============================================================================
# Common Functions
# =============================================================================

# Get model name (replace / with -) for log filename
get_safe_model_name() {
    local model_id="$1"
    local name="${MODEL_NAMES[$model_id]}"
    if [ -z "$name" ]; then
        name="unknown-model${model_id}"
    fi
    echo "$name" | tr '/' '-' | tr '\\' '-'
}

build_think_arg() {
    local think_mode="$1"
    if [ -n "$think_mode" ]; then
        echo "--think-mode $think_mode"
    fi
}

# =============================================================================
# Interactive Menu
# =============================================================================

clear
echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                                                                  ║"
echo "║           🧪      Experiment Runner  (SCE / CEX)      🧪         ║"
echo "║                                                                  ║"
echo "╚══════════════════════════════════════════════════════════════════╝"

# ─────────────────────────────────────────────────────────────────────────────
# 📂 Dataset Selection
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "┌──────────────────────────────────────────────────────────────────┐"
echo "│  📂 Dataset Selection                                            │"
echo "└──────────────────────────────────────────────────────────────────┘"
echo ""
echo "  1) SCE (credit, spending, labor)"
echo "  2) CEX (living_cost)"
echo "  3) All (SCE + CEX)"
echo ""
read -p "👉 Enter 1/2/3 [Default: 1]: " DATASET_MODE
DATASET_MODE=${DATASET_MODE:-1}

case "$DATASET_MODE" in
    2) DATASETS="cex" ;;
    3) DATASETS="sce cex" ;;
    *) DATASETS="sce" ;;
esac

# ─────────────────────────────────────────────────────────────────────────────
# 📋 Experimental Task Selection
# ─────────────────────────────────────────────────────────────────────────────
SCE_SELECTED_EXPS="$SCE_EXPERIMENTS"
CEX_SELECTED_EXPS="$CEX_EXPERIMENTS"

if [[ "$DATASETS" == *"sce"* ]]; then
    echo ""
    echo "┌──────────────────────────────────────────────────────────────────┐"
    echo "│  📋 SCE Experimental Task Selection                              │"
    echo "└──────────────────────────────────────────────────────────────────┘"
    echo ""
    echo "  1) credit   - 💳 Credit Expectations"
    echo "  2) spending - 🛒 Spending Expectations"
    echo "  3) labor    - 👔 Labor Expectations"
    echo "  4) All"
    echo ""
    read -p "👉 Enter Choice [Default: 4]: " SCE_TASK_OPT
    SCE_TASK_OPT=${SCE_TASK_OPT:-4}
    case "$SCE_TASK_OPT" in
        1) SCE_SELECTED_EXPS="credit" ;;
        2) SCE_SELECTED_EXPS="spending" ;;
        3) SCE_SELECTED_EXPS="labor" ;;
        *) SCE_SELECTED_EXPS="credit spending labor" ;;
    esac
fi

if [[ "$DATASETS" == *"cex"* ]]; then
    echo ""
    echo "┌──────────────────────────────────────────────────────────────────┐"
    echo "│  📋 CEX Experimental Task Selection                              │"
    echo "└──────────────────────────────────────────────────────────────────┘"
    echo ""
    echo "  1) living_cost - 💰 Living Cost Expectations"
    echo "  2) All"
    echo ""
    read -p "👉 Enter Choice [Default: 2]: " CEX_TASK_OPT
    CEX_TASK_OPT=${CEX_TASK_OPT:-2}
    case "$CEX_TASK_OPT" in
        1) CEX_SELECTED_EXPS="living_cost" ;;
        *) CEX_SELECTED_EXPS="living_cost" ;;
    esac
fi


# --- Feature Selection ---

# =============================================================================
# SCE Feature Definition (Number -> Feature Name)
# =============================================================================
# Profile Features:
#   1) age            - Age
#   2) gender         - Gender (Male/Female)
#   3) education      - Education Level
#   4) marital_status - Marital Status
#   5) state_residence- State of Residence
#   6) housing_status - Housing Status (Own/Rent)
#   7) own_other_home - Ownership of other homes
#   8) health_status  - Self-reported Health Status
#   9) employment_status - Employment Status
#  10) income         - Household Income Range
#
# Environment Features:
#   1) inflation      - CPI Inflation Rate (past 4 months)
#   2) unemployment   - Unemployment Rate (past 4 months)
#   3) interest_rate  - Federal Funds Rate (past 4 months)
# =============================================================================

declare -a SCE_PROFILE_KEYS=("age" "gender" "education" "marital_status" "state_residence" "housing_status" "own_other_home" "health_status" "employment_status" "income")
declare -a SCE_PROFILE_DESC=("Age" "Gender (Male/Female)" "Education Level" "Marital Status" "State of Residence" "Housing Status (Own/Rent)" "Ownership of other homes" "Self-reported Health" "Employment Status" "Household Income Range")

declare -a SCE_ENV_KEYS=("inflation" "unemployment" "interest_rate")
declare -a SCE_ENV_DESC=("CPI Inflation Rate (past 4 months)" "Unemployment Rate (past 4 months)" "Federal Funds Rate (past 4 months)")

# =============================================================================
# CEX Feature Definition (Number -> Feature Name)
# =============================================================================
# Profile Features:
#   1) age            - Age of reference person
#   2) gender         - Gender
#   3) marital_status - Marital Status
#   4) education      - Education Level
#   5) race           - Race
#   6) household_size - Household Size
#   7) children       - Number of children under 18
#   8) region         - Region (Northeast/Midwest/South/West)
#   9) income         - Annual after-tax household income
#  10) previous_cost  - Previous period consumption (Lagged feature)
#
# Environment Features:
#   1) inflation      - InflationRate YoY
#   2) unemployment   - Unemployment Rate
#   3) interest_rate  - Federal Funds Rate
#   4) gdp            - Quarterly GDP Growth Rate
# =============================================================================

declare -a CEX_PROFILE_KEYS=("age" "gender" "marital_status" "education" "race" "household_size" "children" "region" "income" "previous_cost")
declare -a CEX_PROFILE_DESC=("Age of reference person" "Gender" "Marital status" "Education level" "Race" "Number of people in family" "Number of children under 18" "Region (NE/MW/S/W)" "Annual after-tax household income" "Previous period consumption (Lag)")

declare -a CEX_ENV_KEYS=("inflation" "unemployment" "interest_rate" "gdp")
declare -a CEX_ENV_DESC=("Inflation Rate YoY" "Unemployment Rate" "Federal Funds Rate" "Quarterly GDP Growth")

# =============================================================================
# Feature Selection Function: Display menu and convert numbers to feature names
# =============================================================================
# Parameters: $1=Feature type label, $2=keys array name, $3=desc array name, $4=emoji icon
# Returns: Returns comma-separated feature names via SELECTED_FEATURES variable
select_features() {
    local label="$1"
    local -n keys_ref=$2
    local -n desc_ref=$3
    local emoji="$4"
    local count=${#keys_ref[@]}

    echo ""
    echo "  ${emoji} ${label}"
    echo "  ────────────────────────────────────────"
    echo "     0) ✨ all (All Features)"
    for (( i=0; i<count; i++ )); do
        printf "    %2d) %-18s - %s\n" "$((i+1))" "${keys_ref[$i]}" "${desc_ref[$i]}"
    done
    echo ""
    read -p "  👉 Enter numbers (space-separated, Enter=All) [Default: 0]: " INPUT
    INPUT=${INPUT:-0}

    if [ "$INPUT" = "0" ]; then
        SELECTED_FEATURES="all"
        return
    fi

    # Parse input numbers and convert to feature names
    local result=""
    for num in $INPUT; do
        if [[ "$num" =~ ^[0-9]+$ ]] && [ "$num" -ge 1 ] && [ "$num" -le "$count" ]; then
            local idx=$((num - 1))
            if [ -n "$result" ]; then
                result="${result},${keys_ref[$idx]}"
            else
                result="${keys_ref[$idx]}"
            fi
        fi
    done

    if [ -z "$result" ]; then
        SELECTED_FEATURES="all"
    else
        SELECTED_FEATURES="$result"
    fi
}

# ─────────────────────────────────────────────────────────────────────────────
# 🎯 Feature Selection
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "┌──────────────────────────────────────────────────────────────────┐"
echo "│  🎯 Feature Selection                                            │"
echo "└──────────────────────────────────────────────────────────────────┘"

# Select features based on dataset
if [[ "$DATASETS" == "sce" ]]; then
    # SCE Only
    select_features "SCE Profile Features (Personal Features)" SCE_PROFILE_KEYS SCE_PROFILE_DESC "👤"
    PROFILE_FEAT_INPUT="$SELECTED_FEATURES"

    select_features "SCE Environment Features (Macro Environment)" SCE_ENV_KEYS SCE_ENV_DESC "🌍"
    ENV_FEAT_INPUT="$SELECTED_FEATURES"

elif [[ "$DATASETS" == "cex" ]]; then
    # CEX Only
    select_features "CEX Profile Features (Personal Features)" CEX_PROFILE_KEYS CEX_PROFILE_DESC "👤"
    PROFILE_FEAT_INPUT="$SELECTED_FEATURES"

    select_features "CEX Environment Features (Macro Environment)" CEX_ENV_KEYS CEX_ENV_DESC "🌍"
    ENV_FEAT_INPUT="$SELECTED_FEATURES"

else
    # All (SCE + CEX) - Select separately
    echo ""
    echo "  ══════════════════ SCE Dataset ══════════════════"
    select_features "SCE Profile Features (Personal Features)" SCE_PROFILE_KEYS SCE_PROFILE_DESC "👤"
    SCE_PROFILE_FEAT="$SELECTED_FEATURES"

    select_features "SCE Environment Features (Macro Environment)" SCE_ENV_KEYS SCE_ENV_DESC "🌍"
    SCE_ENV_FEAT="$SELECTED_FEATURES"

    echo ""
    echo "  ══════════════════ CEX Dataset ══════════════════"
    select_features "CEX Profile Features (Personal Features)" CEX_PROFILE_KEYS CEX_PROFILE_DESC "👤"
    CEX_PROFILE_FEAT="$SELECTED_FEATURES"

    select_features "CEX Environment Features (Macro Environment)" CEX_ENV_KEYS CEX_ENV_DESC "🌍"
    CEX_ENV_FEAT="$SELECTED_FEATURES"

    # For multiple datasets, use all (handled by each main.py during execution)
    PROFILE_FEAT_INPUT="all"
    ENV_FEAT_INPUT="all"
    echo ""
    echo "  ⚠️  In multi-dataset mode, default feature sets (all) will be used for each dataset."
fi

echo ""
echo "  ✅ Selected Profile Features:     $PROFILE_FEAT_INPUT"
echo "  ✅ Selected Environment Features: $ENV_FEAT_INPUT"

# Build argument string
FEAT_ARGS="--profile-features \"$PROFILE_FEAT_INPUT\" --env-features \"$ENV_FEAT_INPUT\""

# ─────────────────────────────────────────────────────────────────────────────
# 🖥️ Running Mode Selection
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "┌──────────────────────────────────────────────────────────────────┐"
echo "│  🖥️  Running Mode Selection                                      │"
echo "└──────────────────────────────────────────────────────────────────┘"
echo ""
echo "  1) 🏠 Local vLLM (Iterate through multiple models)"
echo "  2) ☁️  API Provider (e.g., openai_proxy)"
echo ""
read -p "👉 Enter 1 or 2 [Default: 1]: " RUN_MODE
RUN_MODE=${RUN_MODE:-1}



# --- Get experiment list and main.py path based on dataset ---
get_experiments() {
    local ds="$1"
    if [ "$ds" = "sce" ]; then
        echo "$SCE_SELECTED_EXPS"
    else
        echo "$CEX_SELECTED_EXPS"
    fi
}

get_main_script() {
    local ds="$1"
    echo "${SRC_DIR}/${ds}/main.py"
}

# =============================================================================
# Mode 1: Local vLLM
# =============================================================================
run_local_vllm() {
    # ─────────────────────────────────────────────────────────────────────────
    # 🤖 Model Selection
    # ─────────────────────────────────────────────────────────────────────────
    echo ""
    echo "┌──────────────────────────────────────────────────────────────────┐"
    echo "│  🤖 Model Selection (Multi-select possible, space-separated)     │"
    echo "└──────────────────────────────────────────────────────────────────┘"
    echo ""
    echo "  1) 🦙 Meta-Llama-3.1-70B-bnb-4bit"
    echo "  2) 🦙 Meta-Llama-3.1-70B-Instruct-bnb-4bit"
    echo "  3) 🔮 Qwen3-30B-A3B-FP8"
    echo "  4) 🔮 Qwen3-30B-A3B-Instruct-2507-FP8"
    echo "  5) 🔮 Qwen3-0.6B (Lightweight Test)"
    echo ""
    read -p "👉 Model Numbers [Default: 3 4 5]: " MODEL_INPUT
    MODEL_INPUT=${MODEL_INPUT:-"3 4 5"}
    read -ra MODELS <<< "$MODEL_INPUT"

    # Check if Qwen3 models that support think mode are included (Model 4 Instruct-2507 defaults to no_think and does not support think)
    HAS_QWEN3=false
    for m in "${MODELS[@]}"; do
        if [[ "$m" == "3" || "$m" == "5" ]]; then
            HAS_QWEN3=true
            break
        fi
    done

    THINK_MODE=""
    if $HAS_QWEN3; then
        # ─────────────────────────────────────────────────────────────────────
        # 🧠 Qwen3 Thinking Mode
        # ─────────────────────────────────────────────────────────────────────
        echo ""
        echo "┌──────────────────────────────────────────────────────────────────┐"
        echo "│  🧠 Qwen3 Thinking Mode                                          │"
        echo "└──────────────────────────────────────────────────────────────────┘"
        echo ""
        echo "  1) 💭 think    - Enable deep thinking (More accurate, slower)"
        echo "  2) ⚡ no_think - Disable thinking (Faster)"
        echo ""
        read -p "👉 Thinking Mode [Default: 2]: " THINK_CHOICE
        THINK_CHOICE=${THINK_CHOICE:-2}
        if [ "$THINK_CHOICE" = "1" ]; then
            THINK_MODE="think"
        else
            THINK_MODE="no_think"
        fi
        echo ""
        echo "  ✅ Qwen3 Thinking Mode: $THINK_MODE"
    fi

    # Ensure log directories
    for ds in $DATASETS; do
        mkdir -p "${LOG_DIR}/${ds}"
    done

    # ─────────────────────────────────────────────────────────────────────────
    # 🚀 Start Execution
    # ─────────────────────────────────────────────────────────────────────────
    echo ""
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║  🚀 Starting Experimental Execution                              ║"
    echo "╚══════════════════════════════════════════════════════════════════╝"

    # Clean up old sessions
    tmux kill-session -t $SESSION_NAME 2>/dev/null
    echo ""
    echo "  ⏳ Initializing Tmux Session: $SESSION_NAME"
    tmux new-session -d -s $SESSION_NAME -n "Work"
    tmux split-window -h -t $SESSION_NAME:Work

    for MODEL_ID in "${MODELS[@]}"; do
        echo ""
        echo "┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄"
        LOG_START_TIME=$(date +%Y%m%d%H%M%S)
        SAFE_NAME=$(get_safe_model_name "$MODEL_ID")
        # Build server log suffix (Model 4 does not support think mode, no suffix added)
        SRV_THINK_SUFFIX=""
        if [[ "$MODEL_ID" == "3" || "$MODEL_ID" == "5" ]] && [ -n "$THINK_MODE" ]; then
            SRV_THINK_SUFFIX="_${THINK_MODE}"
        fi
        CURRENT_LOG_FILE="${SERVER_LOG_DIR}/${LOG_START_TIME}_server_${SAFE_NAME}${SRV_THINK_SUFFIX}.log"
        echo "  🤖 [$(date +%T)] Preparing Model: ${SAFE_NAME}"
        echo "  📝 Server Log: $CURRENT_LOG_FILE"

        # Launch model
        tmux send-keys -t $SESSION_NAME:Work.0 "source activate $CONDA_ENV && python -u ${SRC_DIR}/server/launch_model.py $MODEL_ID > $CURRENT_LOG_FILE 2>&1" C-m

        # Wait for readiness
        echo "  ⏳ Waiting for model to be ready..."
        sleep 2
        ( timeout 300s tail -f "$CURRENT_LOG_FILE" | grep -q "Application startup complete" )

        if [ $? -ne 0 ]; then
            echo "  ❌ Error: Model startup timed out or failed"
            echo "     Please check the log: $CURRENT_LOG_FILE"
            tmux send-keys -t $SESSION_NAME:Work.0 C-c
            continue
        fi
        echo "  ✅ Model is ready, starting experimental tasks..."

        # Determine if current model is a Qwen3 that supports think mode (Model 4 is not supported)
        CURRENT_THINK_ARG=""
        if [[ "$MODEL_ID" == "3" || "$MODEL_ID" == "5" ]]; then
            CURRENT_THINK_ARG=$(build_think_arg "$THINK_MODE")
        fi

        # Build log suffix: Only append think mode for Qwen3 models that support it
        THINK_SUFFIX=""
        if [[ "$MODEL_ID" == "3" || "$MODEL_ID" == "5" ]] && [ -n "$THINK_MODE" ]; then
            THINK_SUFFIX="_${THINK_MODE}"
        fi

        # Generate independent logs for each experiment, chain commands
        EXP_CMD=""
        for ds in $DATASETS; do
            MAIN_SCRIPT=$(get_main_script "$ds")
            EXPS=$(get_experiments "$ds")
            for exp in $EXPS; do
                EXP_LOG_FILE="${LOG_DIR}/${ds}/${LOG_START_TIME}_${exp}_${SAFE_NAME}${THINK_SUFFIX}.log"
                echo "     📊 [${ds}/${exp}] → $EXP_LOG_FILE"
                if [ -n "$EXP_CMD" ]; then
                    EXP_CMD="$EXP_CMD && "
                fi
                EXP_CMD="${EXP_CMD}python -u $MAIN_SCRIPT --experiment $exp --provider local_vllm $CURRENT_THINK_ARG $FEAT_ARGS 2>&1 | tee -a $EXP_LOG_FILE"
            done
        done

        # Run experiment
        DONE_FLAG="${SRC_DIR}/.finished"
        rm -f "$DONE_FLAG"

        tmux send-keys -t $SESSION_NAME:Work.1 "source activate $CONDA_ENV && cd $SRC_DIR && ($EXP_CMD); touch $DONE_FLAG" C-m

        # Wait for completion
        while [ ! -f "$DONE_FLAG" ]; do
            sleep 5
        done

        # Clean-up
        echo "  ✅ Current experimental round completed, shutting down model and releasing GPU memory..."
        tmux send-keys -t $SESSION_NAME:Work.0 C-c
        rm -f "$DONE_FLAG"
        sleep 10
    done

    # Clean up tmux session, release all resources
    echo ""
    echo "  🧹 Cleaning up Tmux session..."
    tmux kill-session -t $SESSION_NAME 2>/dev/null
    sleep 2

    echo ""
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║  🎉 All local model tasks have been executed successfully!       ║"
    echo "╚══════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "  📁 Server Log: $SERVER_LOG_DIR"
    echo "  📁 Experiment Log:    $LOG_DIR"
    echo ""
}

# =============================================================================
# Mode 2: API Provider
# =============================================================================
run_api_provider() {
    # ─────────────────────────────────────────────────────────────────────────
    # ☁️ API Provider Configuration
    # ─────────────────────────────────────────────────────────────────────────
    echo ""
    echo "┌──────────────────────────────────────────────────────────────────┐"
    echo "│  ☁️  API Provider Configuration                                  │"
    echo "└──────────────────────────────────────────────────────────────────┘"
    echo ""
    read -p "👉 Enter Provider Name [Default: openai_proxy]: " PROVIDER
    PROVIDER=${PROVIDER:-openai_proxy}

    # Check if model name contains qwen3
    MODEL_NAME=$(python -c "
import sys; sys.path.insert(0, '${SRC_DIR}')
from server.config import load_provider_config
cfg = load_provider_config('$PROVIDER')
print(cfg.get('default_model', ''))
" 2>/dev/null)

    THINK_MODE=""
    if echo "$MODEL_NAME" | grep -iq "qwen3" && ! echo "$MODEL_NAME" | grep -iq "instruct"; then
        # ─────────────────────────────────────────────────────────────────────
        # 🧠 Qwen3 Thinking Mode
        # ─────────────────────────────────────────────────────────────────────
        echo ""
        echo "┌──────────────────────────────────────────────────────────────────┐"
        echo "│  🧠 Qwen3 Thinking Mode                                          │"
        echo "└──────────────────────────────────────────────────────────────────┘"
        echo ""
        echo "  1) 💭 think    - Enable deep thinking (More accurate, slower)"
        echo "  2) ⚡ no_think - Disable thinking (Faster)"
        echo ""
        read -p "👉 Thinking Mode [Default: 2]: " THINK_CHOICE
        THINK_CHOICE=${THINK_CHOICE:-2}
        if [ "$THINK_CHOICE" = "1" ]; then
            THINK_MODE="think"
        else
            THINK_MODE="no_think"
        fi
        echo ""
        echo "  ✅ Qwen3 Thinking Mode: $THINK_MODE"
    fi

    THINK_ARG=$(build_think_arg "$THINK_MODE")

    # Use safe model name for logs
    SAFE_MODEL_NAME=$(echo "$MODEL_NAME" | tr '/' '-' | tr '\\' '-')
    API_THINK_SUFFIX=""
    if [ -n "$THINK_MODE" ]; then
        API_THINK_SUFFIX="_${THINK_MODE}"
    fi

    # ─────────────────────────────────────────────────────────────────────────
    # 🚀 Start Execution
    # ─────────────────────────────────────────────────────────────────────────
    LOG_START_TIME=$(date +%Y%m%d%H%M%S)
    echo ""
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║  🚀 Starting Experimental Execution                              ║"
    echo "╚══════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "  ☁️  Provider: $PROVIDER"
    echo "  🤖 Model:     $MODEL_NAME"
    echo "  📂 Dataset:   $DATASETS"
    echo ""
    echo "┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄"

    for ds in $DATASETS; do
        MAIN_SCRIPT=$(get_main_script "$ds")
        EXPS=$(get_experiments "$ds")
        mkdir -p "${LOG_DIR}/${ds}"

        for exp in $EXPS; do
            EXP_LOG_FILE="${LOG_DIR}/${ds}/${LOG_START_TIME}_${exp}_${SAFE_MODEL_NAME}${API_THINK_SUFFIX}.log"
            echo ""
            echo "  📊 [$(date +%T)] Running ${ds}/${exp}"
            echo "     📝 Log: $EXP_LOG_FILE"
                python -u "$MAIN_SCRIPT" \
                    --experiment $exp \
                    --provider $PROVIDER \
                    $THINK_ARG \
                    $FEAT_ARGS \
                    2>&1 | tee -a "$EXP_LOG_FILE"
            echo "  ✅ [$(date +%T)] ${ds}/${exp} Completed"
        done
    done

    echo ""
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║  🎉 All API experimental tasks have been executed successfully!  ║"
    echo "╚══════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "  📁 Log Directory: $LOG_DIR"
    echo ""
}

# =============================================================================
# Main Entry Point
# =============================================================================
if [ "$RUN_MODE" = "2" ]; then
    run_api_provider
else
    run_local_vllm
fi