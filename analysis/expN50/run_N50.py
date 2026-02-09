"""
Exp1.3: Distribution Probe Experiment (Full Features)

Core Idea: Refer to the batch sampling mode of probe_distribution.py,
Perform distribution probing for labor, credit, spending.

Configuration:
- Sampling Ratio: 10% per period
- Random Seed: 2026
- Features: All attributes (10 + 3)
- Default Model: Qwen3-30B-Instruct (ID=4)
"""

import argparse
import asyncio
import sys
import pandas as pd
import yaml
from pathlib import Path
from tqdm.asyncio import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent  # src/
sys.path.insert(0, str(project_root))

# [Fix] Ensure localhost is not proxied
import os
os.environ["no_proxy"] = os.environ.get("no_proxy", "") + ",localhost,127.0.0.1,0.0.0.0"

from sce.config import load_experiment_config
from server.config import load_provider_config, MAX_CONCURRENT_REQUESTS, ACTIVE_PROVIDER
from server.llm_client import get_llm_response
from sce.experiments import (
    SpendingBatchExperiment, CreditBatchExperiment, LaborBatchExperiment,
    Exp8SpendingBatchExperiment, Exp8CreditBatchExperiment, Exp8LaborBatchExperiment,
    Exp9SpendingBatchExperiment, Exp9CreditBatchExperiment, Exp9LaborBatchExperiment
)

# Experiment Class Mapping (Use Batch Version)
EXPERIMENT_CLASSES = {
    'spending': SpendingBatchExperiment,
    'credit': CreditBatchExperiment,
    'labor': LaborBatchExperiment,
    'exp8_spending': Exp8SpendingBatchExperiment,
    'exp8_credit': Exp8CreditBatchExperiment,
    'exp8_labor': Exp8LaborBatchExperiment,
    'exp9_spending': Exp9SpendingBatchExperiment,
    'exp9_credit': Exp9CreditBatchExperiment,
    'exp9_labor': Exp9LaborBatchExperiment,
}

# Default Configuration
DEFAULT_CONFIG = {
    'sample_ratio': 1.0,  # 1.0 = Full Data
    'seed': 2026,
    'n_samples': 50,
    'default_provider': 'local_vllm',
    'output_dir': '/root/autodl-fs/result/exp1.3',
    'experiments': ['labor', 'credit', 'spending'],
    'profile_features': 'all',
    'env_features': 'all'
}



def process_profile(text):
    """Convert second-person profile to third-person."""
    replace_map = [
        ("Your", "The participant's"),
        ("You live", "The participant lives"),
        ("You don't own", "The participant does not own"),
        ("You are", "The participant is"),
        ("You have", "The participant has"),
        ("You", "The participant")
    ]
    for old, new in replace_map:
        text = text.replace(old, new)
    return text

async def process_user_batch(row, system_prompt, experiment, model_name=None, think_mode=None, **kwargs):
    """
    Generate N samples for a single user (one call).
    """
    # 1. Construct User Prompt
    user_prompt = (
            f"{row['env_prompt']}\n\n"
            "The following is a profile of a real participant."
            "Based on the profile, predict how the participant would answer the survey above, giving the best reasonable estimates.\n\n"
            f"{process_profile(row['profile_prompt'])}"
    )

    # 2. Add Question Prompt (if any)
    if hasattr(experiment, 'get_question_prompt'):
        user_prompt += f"\n\n---\n\nBased on the profile above, please answer the survey question:\n{experiment.get_question_prompt()}"
    
    # 3. Batch Sampling Instruction (Included in system_prompt, not repeated here)
    # batch_instruction removed because Batch experiment class system_prompt already contains output format requirements

    # 4. Call LLM
    try:
        result = await get_llm_response(system_prompt, user_prompt, model=model_name, think_mode=think_mode, **kwargs)
    except Exception as e:
        print(f"❌ [User {row['userid']}] LLM Call Failed: {e}")
        return None

    # Check for error string in content (from llm_client)
    if "ERROR: Failed" in result.get('content', ''):
        print(f"⚠️ [User {row['userid']}] LLM Returned Error: {result['content']}")
        # We optionally decide whether to keep this record.
        # If we keep it, it might skew analysis. Let's return None to skip this user.
        return None
    
    return {
        'userid': row['userid'],
        'date': row['date'],
        'llm_response': result['content'],
        'latency': result['latency'],
        'prompt_tokens': result['prompt_tokens'],
        'completion_tokens': result['completion_tokens'],
        'total_tokens': result['total_tokens'],
        'system_prompt': system_prompt,
        'user_prompt': user_prompt
    }


async def run_probe(experiment_name, config, provider_name=None, model_name=None, think_mode=None, dry_run=False, debug_limit=None, **kwargs):
    """
    Run distribution probe for a single experiment.
    """
    # 1. Load Experiment Config (Support config_name mapping, e.g., exp8_spending -> spending)
    ExperimentClass = EXPERIMENT_CLASSES.get(experiment_name)
    if not ExperimentClass:
        raise ValueError(f"Unknown experiment: {experiment_name}")
    config_key = getattr(ExperimentClass, 'config_name', None) or experiment_name
    print(f"🔧 Loading experiment config: {config_key}...")
    exp_config = load_experiment_config(config_key)
    
    # 2. Provider Config (Priority: CLI > config file > System Default)
    effective_provider = provider_name or config.get('default_provider')

    if effective_provider:
        import server.config as config_module
        provider_cfg = load_provider_config(effective_provider)
        config_module.ACTIVE_PROVIDER = provider_cfg
        import server.llm_client as llm_module
        llm_module.client = llm_module._create_client(provider_cfg)
        llm_module.ACTIVE_PROVIDER = provider_cfg
        provider_name = effective_provider
    else:
        provider_cfg = ACTIVE_PROVIDER
        provider_name = provider_cfg.get('provider_name', 'unknown')

    print(f"🔌 Current Provider: {provider_name}")

    # 3. Initialize Experiment Class
    # Get n_samples for initializing experiment class
    n_samples = config.get('n_samples', 50)

    experiment = ExperimentClass(exp_config, exp_config, experiment_name, n_samples=n_samples)

    # [exp8] Set first round experiment result path
    prior_result = kwargs.get('prior_result')
    if prior_result:
        experiment.prior_result_path = prior_result
        print(f"📁 [exp8] First round result file: {prior_result}")

    print(f"🚀 Start {experiment_name.capitalize()} Probe 🚀")
    print(f"📂 Data File: {exp_config['data_file']}")

    # 4. Load Data
    print("⏳ Loading data...")
    df = experiment.load_data()

    # Sort
    if 'date' in df.columns:
        df = df.sort_values(by=['date', 'userid'])

    initial_count = len(df)

    # 5. Stratified Sampling
    sample_ratio = config.get('sample_ratio', 0.1)
    seed = config.get('seed', 2026)
    print(f"🎲 Sampling Data: Ratio={sample_ratio}, Seed={seed}, Stratified by 'date'")

    try:
        df = df.groupby('date', group_keys=False).apply(lambda x: x.sample(frac=sample_ratio, random_state=seed))
        print(f"   Sampling Complete: {initial_count} -> {len(df)} rows")
    except Exception as e:
        print(f"⚠️ Sampling Failed: {e}, using full data.")

    # 5.1 Debug Limit (Limit processed users)
    if debug_limit and debug_limit > 0:
        df = df.head(debug_limit)
        print(f"🐛 [Debug Mode] Limit processed users: {debug_limit}")

    # 6. Generate Prompts (All Features)
    profile_features = config.get('profile_features', 'all')
    env_features = config.get('env_features', 'all')

    # Handle feature parameters
    p_features = None if profile_features == 'all' else profile_features
    e_features = None if env_features == 'all' else env_features

    print(f"⚡ [On-the-fly] Generating prompts (Profile: {profile_features}, Env: {env_features})...")
    df = experiment.prepare_prompts(df, profile_features=p_features, env_features=e_features)

    system_prompt = experiment.get_system_prompt()

    # 7. Determine Model Name
    current_model_name = model_name

    if provider_name == "local_vllm" and not current_model_name:
        print("🔍 Connecting to local model service (vLLM)...")
        try:
            import server.llm_client as llm_module
            # Increase timeout to 30s to allow for slow startup
            models = await asyncio.wait_for(llm_module.client.models.list(), timeout=30)
            if models.data:
                current_model_name = models.data[0].id
                print(f"✅ Connected! Model detected: {current_model_name}")

                # Load model default parameters
                try:
                    from server.launch_model import MODEL_REGISTRY
                    for v in MODEL_REGISTRY.values():
                        if v['id'] == current_model_name or v['name'] == current_model_name:
                            if 'sampling_params' in v:
                                print(f"📥 Loaded model default parameters: {v['sampling_params']}")
                                provider_cfg.update(v['sampling_params'])
                            break
                except:
                    pass
        except Exception as e:
            print(f"❌ Unable to connect to vLLM (http://localhost:8000/v1): {repr(e)}")
            print("   💡 Please check: 1. vLLM service started 2. no_proxy env var (auto-set tried) 3. Port correct")
            return
    elif not current_model_name:
        current_model_name = provider_cfg.get('default_model') or "default-model"
        print(f"🤖 Using model: {current_model_name}")

    # DEBUG: Print Prompt Example
    sample_row = df.iloc[0]
    debug_user_prompt = (
        f"{sample_row.get('env_prompt', '')}\n\n"
        "Now imagine you are a real person with the following demographic profile. "
        "Answer the survey above as this person would, giving your best reasonable estimates.\n\n"
        f"{sample_row['profile_prompt']}"
    )
    if hasattr(experiment, 'get_question_prompt'):
        debug_user_prompt += f"\n\n---\n\nBased on the profile above, please answer the survey question:\n{experiment.get_question_prompt()}"

    print("\n" + "="*20 + " DEBUG: PROMPT (FULL - BATCH MODE) " + "="*20)
    print(f"--- SYSTEM PROMPT ---\n{system_prompt}")
    print(f"--- USER PROMPT ---\n{debug_user_prompt}")
    print("="*55 + "\n")

    # Inference Config
    actual_params = {
        'temperature': kwargs.get('temperature') if kwargs.get('temperature') is not None else provider_cfg.get('temperature', 1),
        'top_p': provider_cfg.get('top_p', 1.0),
        'top_k': provider_cfg.get('top_k'),
        'min_p': provider_cfg.get('min_p'),
        'max_tokens': provider_cfg.get('max_tokens', 4096)
    }
    actual_params = {k: v for k, v in actual_params.items() if v is not None}

    print("\n" + "=" * 60)
    print(f"🧠 [Inference Config] Final Effective Parameters:")
    print(f"   Model       : {current_model_name}")
    print(f"   Param       : {actual_params}")
    print(f"   Mode        : 1 Call per User -> {n_samples} Samples in List")
    print("=" * 60 + "\n")

    # Dry-run Mode
    if dry_run:
        print("🔍 [Dry-Run] Mode enabled, no actual LLM calls.")
        print(f"   Number of users to process: {len(df)}")
        return

    # 8. Prepare Output Path
    output_dir = Path(config.get('output_dir', '/root/autodl-fs/result/exp1.3'))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use provider name as filename identifier (Distinguish same model different configs)
    safe_provider_name = provider_name.replace("/", "-").replace("\\", "-")
    output_filename = f"{experiment_name}_{safe_provider_name}_probe_N{n_samples}.csv"
    output_path = output_dir / output_filename

    print(f"📝 Final Output File: {output_path}")

    # 9. Concurrent Execution
    max_concurrent = provider_cfg.get('max_concurrent', MAX_CONCURRENT_REQUESTS)
    sem = asyncio.Semaphore(max_concurrent)

    async def semaphore_wrapper(row_data):
        async with sem:
            return await process_user_batch(
                row_data, system_prompt, experiment,
                model_name=current_model_name, think_mode=think_mode, **kwargs
            )

    print(f"🔥 Start Concurrent Processing (Pool Size: {max_concurrent})...")
    print(f"📊 Total Requests: {len(df)} Users (Expected {n_samples} samples per user)")

    all_tasks = [semaphore_wrapper(row) for _, row in df.iterrows()]

    results = []
    save_interval = 10

    for f in tqdm(asyncio.as_completed(all_tasks), total=len(all_tasks), desc=f"{experiment_name} Probe Batch"):
        try:
            res = await f
            if res:
                results.append(res)

                # [Checkpoint] Save every 10 successful results
                if len(results) % save_interval == 0:
                    pd.DataFrame(results).to_csv(output_path, index=False)
        except Exception as e:
            print(f"❌ Worker Exception: {e}")

    if len(results) > 0:
        res_df = pd.DataFrame(results)
        res_df.to_csv(output_path, index=False)
        print(f"✅ Probe Experiment Complete. Results saved to: {output_path}")
        print(f"📊 Valid Samples: {len(results)} / {len(df)}")
    else:
        print("⚠️ No results generated (Valid count = 0). Please check errors in log.")


async def main_async(args, config):
    """Async Main Function"""
    experiments = args.experiment if args.experiment else config.get('experiments', ['labor', 'credit', 'spending'])
    
    if isinstance(experiments, str):
        experiments = [experiments]
    
    for exp_name in experiments:
        print(f"\n{'='*60}")
        print(f"🎯 Start Experiment: {exp_name}")
        print(f"{'='*60}\n")
        
        # debug_limit Priority: CLI > config file
        effective_debug_limit = args.debug_limit if args.debug_limit is not None else config.get('debug_limit')
        if effective_debug_limit is not None:
             print(f"🐛 [Debug Mode] Global limit processed users: {effective_debug_limit}")
        
        await run_probe(
            exp_name,
            config,
            provider_name=args.provider,
            model_name=args.model,
            think_mode=args.think_mode,
            dry_run=args.dry_run,
            debug_limit=effective_debug_limit,
            temperature=args.temperature,
            prior_result=args.prior_result
        )


def main():
    parser = argparse.ArgumentParser(description="Exp1.3: Distribution Probe (Full Features)")
    parser.add_argument('--experiment', type=str, nargs='+', default=None,
                        choices=['spending', 'credit', 'labor',
                                 'exp8_spending', 'exp8_credit', 'exp8_labor',
                                 'exp9_spending', 'exp9_credit', 'exp9_labor'],
                        help="Experiment(s) to run (default: all). Use exp8_* for LLM memory, exp9_* for human GT memory.")
    parser.add_argument('--provider', type=str, default=None,
                        help="LLM provider (e.g. local_vllm, openai_proxy)")
    parser.add_argument('--model', type=str, default=None,
                        help="Model name (overrides auto-detection)")
    parser.add_argument('--n-samples', type=int, default=None,
                        help="Number of samples per user (default: 50)")
    parser.add_argument('--think-mode', type=str, default=None,
                        choices=['think', 'no_think'],
                        help="Qwen3 thinking mode")
    parser.add_argument('--temperature', type=float, default=None,
                        help="Override temperature")
    parser.add_argument('--dry-run', action='store_true',
                        help="Dry-run mode (no LLM calls)")
    parser.add_argument('--config', type=str, default=None,
                        help="Path to config YAML file")
    parser.add_argument('--output-dir', type=str, default=None,
                        help="Output directory (overrides config)")
    parser.add_argument('--debug-limit', type=int, default=None,
                        help="Limit number of users to process (for debugging)")
    parser.add_argument('--prior-result', type=str, default=None,
                        help="Path to prior experiment result CSV (for exp8, LLM memory)")
    parser.add_argument('--profile-features', type=str, default=None,
                        help="Profile features: 'all' or comma-separated indices (e.g., '1,2,3,4,5')")
    parser.add_argument('--env-features', type=str, default=None,
                        help="Environment features: 'all' or comma-separated indices (e.g., '1,2,3')")
    parser.add_argument('--sample-ratio', type=float, default=None,
                        help="Sample ratio (default: 1.0, use 0.1 for 10%% sampling)")
    parser.add_argument('--seed', type=int, default=None,
                        help="Random seed for sampling (default: 2026)")

    args = parser.parse_args()

    # Load Config
    config = DEFAULT_CONFIG.copy()
    
    if args.config:
        config_path = Path(args.config)
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = yaml.safe_load(f)
                if user_config:
                    config.update(user_config)
            print(f"📄 Config file loaded: {config_path}")
        else:
            print(f"❌ Error: Config file not found: {config_path}")
            sys.exit(1)
    
    # CLI Argument Override
    if args.n_samples:
        config['n_samples'] = args.n_samples
    if args.output_dir:
        config['output_dir'] = args.output_dir
    if args.profile_features:
        # Parse: 'all' or '1,2,3,4,5' -> ['all'] or [1,2,3,4,5]
        if args.profile_features.lower() == 'all':
            config['profile_features'] = 'all'
        else:
            config['profile_features'] = [int(x.strip()) for x in args.profile_features.split(',')]
    if args.env_features:
        if args.env_features.lower() == 'all':
            config['env_features'] = 'all'
        else:
            config['env_features'] = [int(x.strip()) for x in args.env_features.split(',')]
    if args.sample_ratio is not None:
        config['sample_ratio'] = args.sample_ratio
    if args.seed is not None:
        config['seed'] = args.seed

    # Windows Asyncio Policy
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(main_async(args, config))


if __name__ == "__main__":
    main()
