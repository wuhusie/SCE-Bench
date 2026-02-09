import yaml
import time
import requests
import subprocess
import sys
from pathlib import Path

# Config
BASE_DIR = Path("/root/autodl-tmp/src")
CONFIG_FILE = BASE_DIR / "analysis" / "exp2" / "batch_config_baseline.yaml"
MAIN_SCRIPT = BASE_DIR / "sce" / "main.py"

def wait_for_server(url="http://localhost:8000/v1/models", timeout=300):
    print(f"⏳ Waiting for Model Server at {url}...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print("✅ Model Server is ready!")
                return True
        except requests.RequestException:
            pass
        
        print(f"   ... waiting (elapsed: {int(time.time() - start_time)}s)")
        time.sleep(5)
    
    print("❌ Error: Model Server timed out.")
    return False

def run_jobs():
    if not CONFIG_FILE.exists():
        print(f"❌ Config file not found: {CONFIG_FILE}")
        return

    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    jobs = config.get('jobs', [])
    print(f"📋 Found {len(jobs)} jobs in config.")

    # 1. Wait for Server (Once)
    if not wait_for_server():
        return

    # 2. Execute Jobs
    for i, job in enumerate(jobs):
        print(f"\n▶️  Running Job {i+1}/{len(jobs)}: {job['name']}")
        
        # Build Command
        cmd = [
            "python", str(MAIN_SCRIPT),
            "--experiment", job['sce_experiments'][0], # Assuming list
            "--provider", job['mode'],
            "--sample-ratio", str(job['sample_ratio']),
            "--seed", str(job['seed']),
            "--output-dir", job['output_dir']
        ]

        if job.get('profile_features') == 'all':
             cmd.extend(["--profile-features", "all"]) # main.py supports 'all' or needs explicit? main.py defaults None. 
             # Wait, main.py expects string arguments.
             pass # If None/Default is handled as all? 
             # main.py: p_features = args.profile_features.split(',') if args.profile_features else None
             # If None, Experiment class usually defaults to all? 
             # Let's check main.py or Experiment.
             # Experiment.prepare_prompts defaults to all if None?
             # Checking run_batch.sh: it constructs --profile-features if not empty.
             # In main.py: experiment.prepare_prompts(df, profile_features=...)
        else:
            # Handle list
             if isinstance(job['profile_features'], list):
                 cmd.extend(["--profile-features", ",".join(map(str, job['profile_features']))])
        
        if job.get('env_features') and job['env_features'] != 'all':
             if isinstance(job['env_features'], list):
                 cmd.extend(["--env-features", ",".join(map(str, job['env_features']))])

        # Add explicit args if needed (model_id is handled by server, main.py just queries)
        
        print(f"   Command: {' '.join(cmd)}")
        
        try:
            subprocess.run(cmd, check=True)
            print(f"✅ Job {i+1} completed.")
        except subprocess.CalledProcessError as e:
            print(f"❌ Job {i+1} failed with error: {e}")
            # Optionally continue or break
            # break 

if __name__ == "__main__":
    run_jobs()
