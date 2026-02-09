# Exp1 N50: Distribution Probe (Full Features)

## 1. Experiment Overview

Use **Batch Sampling Mode** to probe distributions for `labor`, `credit`, and `spending` tasks.

| Configuration Item | Value |
|--------------------|-------|
| Sampling Rate      | 10% (Stratified by date) |
| Random Seed        | 2026 |
| Samples per User   | 50 (N=50) |
| Profile Features   | All 10 |
| Environment Features | All 3 |

## 2. File Structure

```
src/analysis/exp1.3/
├── run_N50.py           # Main run script
├── run_all_models.sh    # Batch run script (Unattended)
├── exp1_3_config.yaml   # Config file (Optional)
├── append_ground_truth.py
├── evaluate_distribution.py
└── README.md
```

## 3. Supported Models

Configured in `llm_providers.yaml`:

| Provider Name | Model | Description |
|---------------|-------|-------------|
| `gpt_5_mini_minimal_minimal` | gpt-5-mini | reasoning_effort: minimal |
| `gpt_5_mini_medium` | gpt-5-mini | reasoning_effort: medium |
| `gemini_3_flash_nothinking` | gemini-3-flash-preview-nothinking | No thinking mode |
| `gemini_3_flash_thinking` | gemini-3-flash-preview-thinking | Thinking mode |

## 4. How to Run

### 4.1 Quick Test

```bash
cd src/analysis/exp1.3

# Dry-run (No LLM calls)
python run_N50.py \
    --provider gpt_5_mini_minimal \
    --experiment labor \
    --output-dir /root/autodl-fs/result/exp1/N50/gpt_5_mini_minimal \
    --dry-run

# Real call, limit to 2 users
python run_N50.py \
    --provider gpt_5_mini_minimal \
    --experiment labor \
    --output-dir /root/autodl-fs/result/exp1/N50/gpt_5_mini_minimal \
    --debug-limit 2
```

### 4.2 Single Model Full Run

```bash
python run_N50.py \
    --provider gpt_5_mini_minimal \
    --output-dir /root/autodl-fs/result/exp1/N50/gpt_5_mini_minimal
```

### 4.3 Batch Run (Unattended)

```bash
cd src/analysis/exp1.3
chmod +x run_all_models.sh
nohup ./run_all_models.sh > batch_run.log 2>&1 &
```

The script will run 4 models sequentially, 3 tasks per model.

### 4.4 Monitor Progress

```bash
# View overall progress
tail -f batch_run.log

# View specific model log
tail -f /root/autodl-fs/result/exp1/N50/gpt_5_mini_minimal/*.log
```

## 5. CLI Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--provider` | LLM provider name | local_vllm |
| `--experiment` | Experiment name (multiple allowed) | labor credit spending |
| `--output-dir` | Output directory | /root/autodl-fs/result/exp1.3 |
| `--n-samples` | Samples per user | 50 |
| `--temperature` | Override temperature | (Use provider config) |
| `--dry-run` | Test mode, no LLM calls | false |
| `--debug-limit` | Limit number of users | (No limit) |
| `--config` | Config file path | (None) |
| `--model` | Override model name | (Use provider config) |
| `--think-mode` | Qwen3 Thinking Mode | (None) |

**Priority**: CLI Arguments > Config File > DEFAULT_CONFIG

## 6. Output

### 6.1 Directory Structure

```
/root/autodl-fs/result/exp1/N50/
├── gpt_5_mini_minimal/
│   ├── labor_gpt-5-mini_probe_N50.csv
│   ├── credit_gpt-5-mini_probe_N50.csv
│   ├── spending_gpt-5-mini_probe_N50.csv
│   └── 2026_0207_0400_exp1_N50_gpt_5_mini_minimal.log
├── gpt_5_mini_medium/
├── gemini_3_flash_nothinking/
└── gemini_3_flash_thinking/
```

### 6.2 CSV Fields

| Field | Description |
|-------|-------------|
| userid | User ID |
| date | Date |
| llm_response | LLM Response (JSON list, 50 values) |
| latency | Latency (seconds) |
| prompt_tokens | Input Tokens |
| completion_tokens | Output Tokens |
| total_tokens | Total Tokens |
| system_prompt | Complete System Prompt |
| user_prompt | Complete User Prompt |

### 6.3 Log Naming

Format: `{YYYY}_{MMDD}_{HHMM}_exp1_N50_{model}.log`

Example: `2026_0207_0400_exp1_N50_gpt_5_mini_minimal.log`

## 7. Post-processing

### Append Ground Truth

```bash
python src/analysis/exp1.3/append_ground_truth.py
```

Output to `/root/autodl-fs/result_cleaned/exp1.3/`, suffix `_withHumanData.csv`.

### Evaluate Distribution

```bash
python src/analysis/exp1.3/evaluate_distribution.py
```

## 8. Configuration Details

### DEFAULT_CONFIG (Built-in run_N50.py)

```python
DEFAULT_CONFIG = {
    'sample_ratio': 0.1,      # Sampling Rate
    'seed': 2026,             # Random Seed
    'n_samples': 50,          # Samples per user
    'default_provider': 'local_vllm',
    'output_dir': '/root/autodl-fs/result/exp1.3',
    'experiments': ['labor', 'credit', 'spending'],
    'profile_features': 'all',
    'env_features': 'all'
}
```

### run_all_models.sh Configuration

```bash
BASE_OUTPUT_DIR="/root/autodl-fs/result/exp1/N50"
N_SAMPLES=50
EXPERIMENTS="labor credit spending"
MODELS=("gpt_5_mini_minimal" "gpt_5_mini_medium" "gemini_3_flash_nothinking" "gemini_3_flash_thinking")
```
