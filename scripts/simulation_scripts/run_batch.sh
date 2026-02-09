#!/bin/bash

# =============================================================================
# Batch Experiment Runner — Reads batch_config.yaml, fully automated sequential execution of all jobs
# =============================================================================

SESSION_NAME="exp"
CONDA_ENV="/root/autodl-tmp/agent_env"
SRC_DIR="/root/autodl-tmp/src"
LOG_DIR="${SRC_DIR}/log"
SERVER_LOG_DIR="${LOG_DIR}/server"
# Supports passing configuration file path from command line (default batch_config.yaml)
BATCH_CONFIG="${1:-${SRC_DIR}/batch_config.yaml}"

# SCE / CEX experiment list
SCE_EXPERIMENTS="credit spending labor"
CEX_EXPERIMENTS="living_cost"

# Model number -> Name mapping
declare -A MODEL_NAMES
MODEL_NAMES[1]="Meta-Llama-3.1-70B-bnb-4bit"
MODEL_NAMES[2]="Meta-Llama-3.1-70B-Instruct-bnb-4bit"
MODEL_NAMES[3]="Qwen3-30B-A3B-FP8"
MODEL_NAMES[4]="Qwen3-30B-A3B-Instruct-2507-FP8"
MODEL_NAMES[5]="Qwen3-0.6B"
MODEL_NAMES[6]="Meta-Llama-3.1-70B-Instruct-GPTQ-INT4"

mkdir -p "$SERVER_LOG_DIR"

# [Important] Disable local proxy to prevent 503 errors when connecting to vLLM
export no_proxy="localhost,127.0.0.1,0.0.0.0"

# =============================================================================
# Feature Definitions (Number -> Feature Name)
# =============================================================================
# SCE Profile: 1-10, SCE Env: 1-3
# CEX Profile: 1-10, CEX Env: 1-4
# =============================================================================

declare -a SCE_PROFILE_KEYS=("age" "gender" "education" "marital_status" "state_residence" "housing_status" "own_other_home" "health_status" "employment_status" "income")
declare -a SCE_ENV_KEYS=("inflation" "unemployment" "interest_rate")

declare -a CEX_PROFILE_KEYS=("age" "gender" "marital_status" "education" "race" "household_size" "children" "region" "income" "previous_cost")
declare -a CEX_ENV_KEYS=("inflation" "unemployment" "interest_rate" "gdp")

# =============================================================================
# Common Functions
# =============================================================================

get_safe_model_name() {
    local model_id="$1"
    local name="${MODEL_NAMES[$model_id]}"
    if [ -z "$name" ]; then
        name="unknown-model${model_id}"
    fi
    echo "$name" | tr '/' '-' | tr '\\' '-'
}

get_experiments() {
    local ds="$1"
    if [ "$ds" = "sce" ]; then
        echo "$SCE_EXPERIMENTS"
    else
        echo "$CEX_EXPERIMENTS"
    fi
}

get_main_script() {
    local ds="$1"
    echo "${SRC_DIR}/${ds}/main.py"
}

is_qwen3_model() {
    # Only returns Qwen3 models that support think mode (Model 4 Instruct-2507 defaults to no_think and does not support think)
    local model_id="$1"
    [[ "$model_id" == "3" || "$model_id" == "5" ]]
}

# =============================================================================
# Feature Conversion Function: Convert numerical index to feature name
# =============================================================================
# Parameters: $1=Input value (number or feature name), $2=Dataset (sce/cex), $3=Feature type (profile/env)
# Returns: Returns converted feature name (comma-separated) or "all" via echo
convert_features() {
    local input="$1"
    local dataset="$2"
    local feat_type="$3"

    # Empty value or all, return immediately
    if [ -z "$input" ] || [ "$input" = "all" ]; then
        echo "all"
        return
    fi

    # Select corresponding feature array
    local -n keys_ref
    if [ "$dataset" = "sce" ]; then
        if [ "$feat_type" = "profile" ]; then
            keys_ref=SCE_PROFILE_KEYS
        else
            keys_ref=SCE_ENV_KEYS
        fi
    else
        if [ "$feat_type" = "profile" ]; then
            keys_ref=CEX_PROFILE_KEYS
        else
            keys_ref=CEX_ENV_KEYS
        fi
    fi
    local count=${#keys_ref[@]}

    # Check if input is pure numeric format (e.g., "1,2,3" or "1 2 3")
    local normalized=$(echo "$input" | tr ',' ' ')
    local is_numeric=true
    for item in $normalized; do
        if ! [[ "$item" =~ ^[0-9]+$ ]]; then
            is_numeric=false
            break
        fi
    done

    # If not pure numeric, return original value (assuming it's a feature name)
    if [ "$is_numeric" = false ]; then
        echo "$input"
        return
    fi

    # Convert number to feature name
    local result=""
    for num in $normalized; do
        if [ "$num" -ge 1 ] && [ "$num" -le "$count" ]; then
            local idx=$((num - 1))
            if [ -n "$result" ]; then
                result="${result},${keys_ref[$idx]}"
            else
                result="${keys_ref[$idx]}"
            fi
        fi
    done

    if [ -z "$result" ]; then
        echo "all"
    else
        echo "$result"
    fi
}

# =============================================================================
# Parse batch_config.yaml (Read using Python, output in shell-parsable format)
# =============================================================================

TOTAL_JOBS=$(python3 -c "
import yaml, sys
with open('${BATCH_CONFIG}') as f:
    cfg = yaml.safe_load(f)
print(len(cfg.get('jobs', [])))
" 2>/dev/null)

if [ -z "$TOTAL_JOBS" ] || [ "$TOTAL_JOBS" -eq 0 ]; then
    echo ""
    echo "❌ Error: jobs not found in batch_config.yaml, or file parsing failed"
    exit 1
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║           🔄         Batch Experiment Runner          🔄         ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "  📁 Config File: $BATCH_CONFIG"
echo "  📋 Total Tasks: $TOTAL_JOBS"
echo ""
echo "──────────────────────────────────────────────────────────────────────"

# =============================================================================
# Main Loop: Execute each job sequentially
# =============================================================================

CURRENT_VLLM_MODEL=""  # Record currently launched local model ID to avoid redundant startups/shutdowns

for (( JOB_IDX=0; JOB_IDX<TOTAL_JOBS; JOB_IDX++ )); do
    # Parse fields of the current job using Python
    # Rules:
    #   - Experiment lists (DATASETS, SCE_EXPS, CEX_EXPS): Space-separated (for shell loops)
    #   - Feature lists (PROFILE_FEAT, ENV_FEAT): Comma-separated (passed to Python scripts)
    eval "$(python3 -c "
import yaml

def to_space_sep(val):
    '''List -> Space-separated string (for shell loops)'''
    if val is None:
        return ''
    if isinstance(val, list):
        return ' '.join(str(x) for x in val)
    return str(val)

def to_comma_sep(val):
    '''List -> Comma-separated string (for Python parameters)'''
    if val is None:
        return ''
    if isinstance(val, list):
        return ','.join(str(x) for x in val)
    return str(val)

with open('${BATCH_CONFIG}') as f:
    cfg = yaml.safe_load(f)
job = cfg['jobs'][$JOB_IDX]

# Base fields
print(f\"JOB_NAME=\\\"{job.get('name', 'job_$JOB_IDX')}\\\"\" )
print(f\"JOB_MODE=\\\"{job.get('mode', 'local_vllm')}\\\"\" )
print(f\"JOB_MODEL_ID=\\\"{job.get('model_id', '')}\\\"\" )
print(f\"JOB_PROVIDER=\\\"{job.get('provider', '')}\\\"\" )
print(f\"JOB_THINK_MODE=\\\"{job.get('think_mode', '')}\\\"\" )
print(f\"JOB_DEBUG_LIMIT=\\\"{job.get('debug_limit', '')}\\\"\" )

# Feature configuration (comma-separated, passed to Python scripts)
print(f\"JOB_PROFILE_FEAT=\\\"{to_comma_sep(job.get('profile_features'))}\\\"\" )
print(f\"JOB_ENV_FEAT=\\\"{to_comma_sep(job.get('env_features'))}\\\"\" )

# [New] Sampling parameters
print(f\"JOB_SAMPLE_RATIO=\\\"{job.get('sample_ratio', '')}\\\"\" )
print(f\"JOB_SEED=\\\"{job.get('seed', '')}\\\"\" )
# [New] Output directory
print(f\"JOB_OUTPUT_DIR=\\\"{job.get('output_dir', '')}\\\"\" )

# [New] Custom suffix (YAML suffix priority, otherwise auto-generated from job name)
suffix = job.get('suffix', '')
if not suffix:
    # Extract 'No xxx' part from job name, generate 'NOxxx_SEEDyyy' format
    import re
    name = job.get('name', '')
    match = re.search(r'No[_ ](\w+)', name, re.IGNORECASE)
    if match:
        feature = match.group(1)
        seed = job.get('seed', '')
        suffix = f'NO{feature}'
        if seed:
            suffix += f'_SEED{seed}'
print(f\"JOB_SUFFIX=\\\"{suffix}\\\"\" )

# Experimental task configuration (space-separated, for shell loops)
sce_exps = job.get('sce_experiments')
cex_exps = job.get('cex_experiments')

# SCE: 'all' expands to all experiment names
if sce_exps == 'all':
    sce_str = 'credit spending labor'
else:
    sce_str = to_space_sep(sce_exps)

# CEX: 'all' expands to all experiment names
if cex_exps == 'all':
    cex_str = 'living_cost'
else:
    cex_str = to_space_sep(cex_exps)

print(f\"JOB_SCE_EXPS=\\\"{sce_str}\\\"\" )
print(f\"JOB_CEX_EXPS=\\\"{cex_str}\\\"\" )

# Infer dataset from experiment configuration (space-separated)
datasets = []
if sce_str:
    datasets.append('sce')
if cex_str:
    datasets.append('cex')
print(f\"JOB_DATASETS=\\\"{' '.join(datasets)}\\\"\" )
")"

    echo ""
    echo "┌──────────────────────────────────────────────────────────────────┐"
    echo "│  📌 Task [$((JOB_IDX+1))/$TOTAL_JOBS]: $JOB_NAME"
    echo "├──────────────────────────────────────────────────────────────────┤"
    if [ "$JOB_MODE" = "local_vllm" ]; then
        echo "│  🖥️  Mode: Local vLLM"
        echo "│  🤖 Model: $(get_safe_model_name $JOB_MODEL_ID) (#$JOB_MODEL_ID)"
    else
        echo "│  ☁️  Mode: Remote API"
        echo "│  🔌 Provider: $JOB_PROVIDER"
    fi
    [ -n "$JOB_THINK_MODE" ] && echo "│  🧠 Think Mode: $JOB_THINK_MODE"
    [ -n "$JOB_DEBUG_LIMIT" ] && echo "│  🐛 Debug Limit: $JOB_DEBUG_LIMIT"
    [ -n "$JOB_SCE_EXPS" ] && echo "│  📊 SCE Experiments: $JOB_SCE_EXPS"
    [ -n "$JOB_CEX_EXPS" ] && echo "│  📈 CEX Experiments: $JOB_CEX_EXPS"
    [ -n "$JOB_PROFILE_FEAT" ] && echo "│  👤 Profile: $JOB_PROFILE_FEAT"
    [ -n "$JOB_ENV_FEAT" ] && echo "│  🌍 Env: $JOB_ENV_FEAT"
    echo "└──────────────────────────────────────────────────────────────────┘"

    LOG_START_TIME=$(date +%Y%m%d%H%M%S)

    # Get first dataset for feature conversion (multi-dataset tasks use the first as reference)
    FIRST_DATASET=$(echo "$JOB_DATASETS" | awk '{print $1}')

    # Convert features (supports numerical index or feature names)
    CONVERTED_PROFILE_FEAT=$(convert_features "$JOB_PROFILE_FEAT" "$FIRST_DATASET" "profile")
    CONVERTED_ENV_FEAT=$(convert_features "$JOB_ENV_FEAT" "$FIRST_DATASET" "env")

    # Display converted features
    if [ -n "$JOB_PROFILE_FEAT" ] && [ "$JOB_PROFILE_FEAT" != "$CONVERTED_PROFILE_FEAT" ]; then
        echo "  🔀 Profile Conversion: $JOB_PROFILE_FEAT → $CONVERTED_PROFILE_FEAT"
    fi
    if [ -n "$JOB_ENV_FEAT" ] && [ "$JOB_ENV_FEAT" != "$CONVERTED_ENV_FEAT" ]; then
        echo "  🔀 Env Conversion: $JOB_ENV_FEAT → $CONVERTED_ENV_FEAT"
    fi

    # Build general feature parameters (comma-separated feature names, no quotes needed)
    FEAT_ARGS=""
    if [ "$CONVERTED_PROFILE_FEAT" != "all" ] && [ -n "$CONVERTED_PROFILE_FEAT" ]; then
        FEAT_ARGS="$FEAT_ARGS --profile-features $CONVERTED_PROFILE_FEAT"
    fi
    if [ "$CONVERTED_ENV_FEAT" != "all" ] && [ -n "$CONVERTED_ENV_FEAT" ]; then
        FEAT_ARGS="$FEAT_ARGS --env-features $CONVERTED_ENV_FEAT"
    fi

    # [New] Build sampling parameters
    SAMPLE_ARGS=""
    if [ -n "$JOB_SAMPLE_RATIO" ]; then
        SAMPLE_ARGS="--sample-ratio $JOB_SAMPLE_RATIO"
        if [ -n "$JOB_SEED" ]; then
            SAMPLE_ARGS="$SAMPLE_ARGS --seed $JOB_SEED"
        fi
    fi

    # [New] Custom output directory
    OUTPUT_ARG=""
    if [ -n "$JOB_OUTPUT_DIR" ]; then
        OUTPUT_ARG="--output-dir $JOB_OUTPUT_DIR"
    fi

    # [New] Custom suffix
    SUFFIX_ARG=""
    if [ -n "$JOB_SUFFIX" ]; then
        SUFFIX_ARG="--suffix $JOB_SUFFIX"
    fi

    # ------------------------------------------------------------------
    # 本地 vLLM 模式
    # ------------------------------------------------------------------
    if [ "$JOB_MODE" = "local_vllm" ]; then

        SAFE_NAME=$(get_safe_model_name "$JOB_MODEL_ID")

        # Determine if restart is needed (restart only if model differs from previous job)
        if [ "$CURRENT_VLLM_MODEL" != "$JOB_MODEL_ID" ]; then

            # Shut down old model
            if [ -n "$CURRENT_VLLM_MODEL" ]; then
                echo "  ⏹️  Shutting down previous model..."
                tmux send-keys -t $SESSION_NAME:Work.0 C-c
                sleep 10
            else
                # First time: Initialize tmux
                tmux kill-session -t $SESSION_NAME 2>/dev/null
                tmux new-session -d -s $SESSION_NAME -n "Work"
                tmux split-window -h -t $SESSION_NAME:Work
            fi

            # Launch new model
            THINK_SUFFIX=""
            if is_qwen3_model "$JOB_MODEL_ID" && [ -n "$JOB_THINK_MODE" ]; then
                THINK_SUFFIX="_${JOB_THINK_MODE}"
            fi
            SERVER_LOG="${SERVER_LOG_DIR}/${LOG_START_TIME}_server_${SAFE_NAME}${THINK_SUFFIX}.log"
            echo ""
            echo "  🚀 Launching model: ${SAFE_NAME}"
            echo "  📝 Server Log: $SERVER_LOG"

            tmux send-keys -t $SESSION_NAME:Work.0 "source activate $CONDA_ENV && python -u ${SRC_DIR}/server/launch_model.py $JOB_MODEL_ID > $SERVER_LOG 2>&1" C-m

            echo "  ⏳ Waiting for model to be ready..."
            sleep 2
            ( timeout 300s tail -f "$SERVER_LOG" | grep -q "Application startup complete" )

            if [ $? -ne 0 ]; then
                echo "  ❌ Error: Model startup timed out or failed, skipping this task"
                tmux send-keys -t $SESSION_NAME:Work.0 C-c
                CURRENT_VLLM_MODEL=""
                continue
            fi
            echo "  ✅ Model is ready"
            CURRENT_VLLM_MODEL="$JOB_MODEL_ID"
        else
            echo ""
            echo "  ♻️  Model ${SAFE_NAME} is already running, reusing"
        fi

        # Build think parameters
        THINK_ARG=""
        if is_qwen3_model "$JOB_MODEL_ID" && [ -n "$JOB_THINK_MODE" ]; then
            THINK_ARG="--think-mode $JOB_THINK_MODE"
        fi

        # Build debug-limit parameters
        DEBUG_ARG=""
        if [ -n "$JOB_DEBUG_LIMIT" ]; then
            DEBUG_ARG="--debug-limit $JOB_DEBUG_LIMIT"
        fi

        # Build log suffix
        THINK_SUFFIX=""
        if is_qwen3_model "$JOB_MODEL_ID" && [ -n "$JOB_THINK_MODE" ]; then
            THINK_SUFFIX="_${JOB_THINK_MODE}"
        fi

        # Build experiment command chain
        EXP_CMD=""
        for ds in $JOB_DATASETS; do
            mkdir -p "${LOG_DIR}/${ds}"
            MAIN_SCRIPT=$(get_main_script "$ds")
            # Get experiment list for this dataset
            if [ "$ds" = "sce" ]; then
                EXPS="$JOB_SCE_EXPS"
            else
                EXPS="$JOB_CEX_EXPS"
            fi
            for exp in $EXPS; do
                EXP_LOG="${LOG_DIR}/${ds}/${LOG_START_TIME}_${exp}_${SAFE_NAME}${THINK_SUFFIX}.log"
                echo "  📄 [${ds}/${exp}] Log: $EXP_LOG"
                if [ -n "$EXP_CMD" ]; then
                    EXP_CMD="$EXP_CMD && "
                fi
                EXP_CMD="${EXP_CMD}python -u $MAIN_SCRIPT --experiment $exp --provider local_vllm $THINK_ARG $DEBUG_ARG $FEAT_ARGS $SAMPLE_ARGS $OUTPUT_ARG $SUFFIX_ARG 2>&1 | tee -a $EXP_LOG"
            done
        done

        # Execute
        echo ""
        echo "  ▶️  Starting experimental execution..."
        DONE_FLAG="${SRC_DIR}/.finished_batch"
        rm -f "$DONE_FLAG"
        tmux send-keys -t $SESSION_NAME:Work.1 "source activate $CONDA_ENV && cd $SRC_DIR && ($EXP_CMD); touch $DONE_FLAG" C-m

        while [ ! -f "$DONE_FLAG" ]; do
            sleep 5
        done
        rm -f "$DONE_FLAG"
        echo ""
        echo "  ✅ Task [$((JOB_IDX+1))/$TOTAL_JOBS] Completed"
        echo "──────────────────────────────────────────────────────────────────────"

    # ------------------------------------------------------------------
    # API 模式
    # ------------------------------------------------------------------
    elif [ "$JOB_MODE" = "api" ]; then

        # If local model is running, shut it down first
        if [ -n "$CURRENT_VLLM_MODEL" ]; then
            echo "  ⏹️  Shutting down local model, switching to API mode..."
            tmux send-keys -t $SESSION_NAME:Work.0 C-c
            sleep 10
            CURRENT_VLLM_MODEL=""
        fi

        # Get model name
        MODEL_NAME=$(python3 -c "
import sys; sys.path.insert(0, '${SRC_DIR}')
from server.config import load_provider_config
cfg = load_provider_config('$JOB_PROVIDER')
print(cfg.get('default_model', ''))
" 2>/dev/null)
        SAFE_MODEL_NAME=$(echo "$MODEL_NAME" | tr '/' '-' | tr '\\' '-')

        THINK_ARG=""
        THINK_SUFFIX=""
        if [ -n "$JOB_THINK_MODE" ]; then
            THINK_ARG="--think-mode $JOB_THINK_MODE"
            THINK_SUFFIX="_${JOB_THINK_MODE}"
        fi

        DEBUG_ARG=""
        if [ -n "$JOB_DEBUG_LIMIT" ]; then
            DEBUG_ARG="--debug-limit $JOB_DEBUG_LIMIT"
        fi

        echo ""
        echo "  🔌 Provider: $JOB_PROVIDER"
        echo "  🤖 Model: $MODEL_NAME"
        echo ""
        echo "  ▶️  Starting experimental execution..."

        for ds in $JOB_DATASETS; do
            mkdir -p "${LOG_DIR}/${ds}"
            MAIN_SCRIPT=$(get_main_script "$ds")
            # Get experiment list for this dataset
            if [ "$ds" = "sce" ]; then
                EXPS="$JOB_SCE_EXPS"
            else
                EXPS="$JOB_CEX_EXPS"
            fi
            for exp in $EXPS; do
                EXP_LOG="${LOG_DIR}/${ds}/${LOG_START_TIME}_${exp}_${SAFE_MODEL_NAME}${THINK_SUFFIX}.log"
                echo "  📄 [${ds}/${exp}] Log: $EXP_LOG"
                python -u "$MAIN_SCRIPT" \
                    --experiment $exp \
                    --provider $JOB_PROVIDER \
                    $THINK_ARG $DEBUG_ARG $FEAT_ARGS $SAMPLE_ARGS $OUTPUT_ARG $SUFFIX_ARG \
                    2>&1 | tee -a "$EXP_LOG"
            done
        done

        echo ""
        echo "  ✅ Task [$((JOB_IDX+1))/$TOTAL_JOBS] Completed"
        echo "──────────────────────────────────────────────────────────────────────"

    else
        echo "  ⚠️  Unknown mode: $JOB_MODE, skipping"
    fi
done

# =============================================================================
# Wrap-up: Close residual models and clean up tmux session
# =============================================================================
echo ""
echo "🧹 Cleaning up Tmux session and releasing resources..."
tmux kill-session -t $SESSION_NAME 2>/dev/null
sleep 2

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                  🎉  All Batch Tasks Completed!  🎉              ║"
echo "╠══════════════════════════════════════════════════════════════════╣"
echo "║  📋 Completed tasks: $TOTAL_JOBS"
echo "║  📂 Log directory: $LOG_DIR"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
