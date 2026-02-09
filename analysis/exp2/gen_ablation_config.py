import yaml
from pathlib import Path

# Paths
BASE_DIR = Path(r"/root/autodl-tmp/src")
OUTPUT_FILE = BASE_DIR / "analysis" / "exp2" / "batch_config_full.yaml"

# Configuration Constants
MODEL_ID = 4  # Qwen3-30B-Instruct
MODE = "local_vllm"
SAMPLE_RATIO = 0.1
MODE = "local_vllm"
SAMPLE_RATIO = 0.1
SEED = 2026
OUTPUT_DIR = "result/exp2/N1/Full"

TASKS = ["credit", "spending", "labor"]

# Feature Definitions (Numeric IDs based on batch_config.yaml)
# Profile: 1-10
# Env: 1-3
PROFILE_IDS = list(range(1, 11))
ENV_IDS = list(range(1, 4))

# Optional Maps for better naming (Job Name only)
PROFILE_MAP = {
    1: "age", 2: "gender", 3: "education", 4: "marital_status", 5: "state_residence",
    6: "housing_status", 7: "own_other_home", 8: "health_status", 9: "employment_status", 10: "income"
}
ENV_MAP = {
    1: "inflation", 2: "unemployment", 3: "interest_rate"
}

def generate_config():
    jobs = []
    
    # Generate Jobs for each Task
    for task in TASKS:
        # 1. Ablation on Profile Features (Leave one out)
        for fid in PROFILE_IDS:
            # Create list excluding current feature
            subset = [i for i in PROFILE_IDS if i != fid]
            feat_name = PROFILE_MAP.get(fid, str(fid))
            
            job = {
                "name": f"Exp2 Ablation - {task} - No {feat_name}",
                "mode": MODE,
                "model_id": MODEL_ID,
                "sce_experiments": [task],
                "profile_features": subset,
                "env_features": "all", # Or list(range(1,4))
                "sample_ratio": SAMPLE_RATIO,
                "seed": SEED,
                "output_dir": OUTPUT_DIR
            }
            jobs.append(job)

        # 2. Ablation on Env Features (Leave one out)
        for fid in ENV_IDS:
            subset = [i for i in ENV_IDS if i != fid]
            feat_name = ENV_MAP.get(fid, str(fid))
            
            job = {
                "name": f"Exp2 Ablation - {task} - No {feat_name}",
                "mode": MODE,
                "model_id": MODEL_ID,
                "sce_experiments": [task],
                "profile_features": "all",
                "env_features": subset,
                "sample_ratio": SAMPLE_RATIO,
                "seed": SEED,
                "output_dir": OUTPUT_DIR
            }
            jobs.append(job)

    # Construct final yaml structure
    config = {
        "description": "Experiment 2: Ablation Study (Leave-One-Out) with 10% Sampling",
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
