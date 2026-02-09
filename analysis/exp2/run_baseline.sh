#!/bin/bash

# Configuration
SESSION="exp2_baseline"
MODEL_ID=4
SRC_DIR="/root/autodl-tmp/src"

# 1. Create Detached Session
tmux new-session -d -s $SESSION

# 2. Split Window (Left: Model, Right: Task)
tmux split-window -h -t $SESSION:0

# 3. Left Pane: Launch Model
# Use 'python -u' to unbuffer output
tmux send-keys -t $SESSION:0.0 "cd $SRC_DIR && python -u server/launch_model.py $MODEL_ID" C-m

# 4. Right Pane: Run Experiment (Wait -> Run)
# Use 'python -u' and wait a bit before starting ensuring potential race conditions
tmux send-keys -t $SESSION:0.1 "export no_proxy='localhost,127.0.0.1,0.0.0.0' && cd $SRC_DIR && python -u analysis/exp2/run_baseline_jobs.py" C-m

# 5. Attach to Session
# This brings the user into the tmux view immediately
tmux attach -t $SESSION
