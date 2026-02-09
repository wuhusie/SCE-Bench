import yaml
from pathlib import Path

# Paths
BASE_DIR = Path(r"/root/autodl-tmp/src")
OUTPUT_FILE = BASE_DIR / "analysis" / "exp2" / "batch_config_baseline.yaml"

# Configuration Constants
MODEL_ID = 4  # Qwen3-30B-Instruct
MODE = "local_vllm"
SAMPLE_RATIO = 0.1
SEED = 2026
OUTPUT_DIR = "result/sce/exp2"

TASKS = ["credit", "spending", "labor"]

def generate_config():
    jobs = []
    
    # Generate Baseline Jobs for each Task
    for task in TASKS:
        job = {
            "name": f"Exp2 Baseline - {task}",
            "mode": MODE,
            "model_id": MODEL_ID,
            "sce_experiments": [task],
            "profile_features": "all",
            "env_features": "all",
            "sample_ratio": SAMPLE_RATIO,
            "seed": SEED,
            "output_dir": OUTPUT_DIR
        }
        jobs.append(job)

    # Construct final yaml structure
    config = {
        "description": "Experiment 2: Baseline (Non-Ablation) with 10% Sampling",
        "jobs": jobs
    }
    
    # Save
    if not OUTPUT_FILE.parent.exists():
        OUTPUT_FILE.parent.mkdir(parents=True)
        
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, sort_keys=False, allow_unicode=True)
    
    print(f"Generated {len(jobs)} jobs in {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_config()
