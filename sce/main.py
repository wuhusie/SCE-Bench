import argparse
import asyncio
import sys
import pandas as pd
import os
from pathlib import Path

# Add project root directory to Python path
# Project root directory: /root/autodl-tmp/src
# Current file: /root/autodl-tmp/src/sce/main.py
project_root = Path(__file__).parent.parent  # One level up to src/
sys.path.insert(0, str(project_root))

from tqdm.asyncio import tqdm
from sce.config import load_experiment_config
from server.config import load_provider_config, MAX_CONCURRENT_REQUESTS, ACTIVE_PROVIDER
from server.llm_client import get_llm_response, client as llm_client_instance
from sce.experiments import SpendingExperiment, CreditExperiment, LaborExperiment
from sce.experiments.exp8 import Exp8SpendingExperiment, Exp8CreditExperiment, Exp8LaborExperiment
from sce.experiments.exp9 import Exp9SpendingExperiment, Exp9CreditExperiment, Exp9LaborExperiment

# Experiment class mapping factory
EXPERIMENT_CLASSES = {
    'spending': SpendingExperiment,
    'credit': CreditExperiment,
    'labor': LaborExperiment,
    'exp8_spending': Exp8SpendingExperiment,
    'exp8_credit': Exp8CreditExperiment,
    'exp8_labor': Exp8LaborExperiment,
    'exp9_spending': Exp9SpendingExperiment,
    'exp9_credit': Exp9CreditExperiment,
    'exp9_labor': Exp9LaborExperiment,
}

async def process_single_row(row, sem, pbar, experiment, system_prompt, model_name=None, think_mode=None, **kwargs):
    """
    Asynchronous function to process a single row of data.
    Constructs prompt via experiment.build_row_prompts(); behavior can be overridden by subclasses.
    """
    async with sem:
        userid = row['userid']
        date = row['date']

        # Experiment class decides the prompt construction method
        final_system, user_prompt, extra_fields = experiment.build_row_prompts(row, system_prompt)

        # Call LLM
        result = await get_llm_response(final_system, user_prompt, model=model_name, think_mode=think_mode, **kwargs)

        pbar.update(1)

        output = {
            'userid': userid,
            'date': date,
            'llm_response': result['content'],
            'latency': result['latency'],
            'prompt_tokens': result['prompt_tokens'],
            'completion_tokens': result['completion_tokens'],
            'total_tokens': result['total_tokens'],
            'system_prompt': final_system,
            'user_prompt': user_prompt
        }
        output.update(extra_fields)
        return output

async def run_experiment(experiment_name, debug_limit=None, provider_name=None, think_mode=None, profile_features=None, env_features=None, **kwargs):
    # 1. Load configuration (supports config_name mapping)
    ExperimentClass = EXPERIMENT_CLASSES.get(experiment_name)
    config_key = getattr(ExperimentClass, 'config_name', None) or experiment_name
    print(f"🔧 Loading experiment configuration: {config_key}...")
    config = load_experiment_config(config_key)
    
    # Manually override debug_limit if passed from CLI
    if debug_limit is not None:
        config['debug_limit'] = debug_limit

    # 2. Load provider configuration
    if provider_name:
        import server.config as config_module
        provider_cfg = load_provider_config(provider_name)
        config_module.ACTIVE_PROVIDER = provider_cfg
        # Re-create global client
        import server.llm_client as llm_module
        llm_module.client = llm_module._create_client(provider_cfg)
        llm_module.ACTIVE_PROVIDER = provider_cfg
    else:
        provider_cfg = ACTIVE_PROVIDER
        provider_name = provider_cfg['provider_name']

    print(f"🔌 Current Provider: {provider_name}")

    # 3. Initialize experiment class
    experiment = ExperimentClass(config, config, experiment_name)

    # [exp8] Set first-round experiment result path
    if kwargs.get('prior_result'):
        experiment.prior_result_path = kwargs['prior_result']
        print(f"📁 First-round result file: {kwargs['prior_result']}")

    print(f"🚀 Starting {experiment_name.capitalize()} experiment 🚀")
    print(f"📂 Data file: {config['data_file']}")


    # 4. Determine model name
    current_model_name = None

    if provider_name == "local_vllm":
        # Local vLLM: Auto-detect model
        print("🔍 Connecting to local model service (vLLM)...")
        try:
            import server.llm_client as llm_module
            models = await asyncio.wait_for(llm_module.client.models.list(), timeout=10)
            if models.data:
                current_model_name = models.data[0].id
                print(f"✅ Connection successful! Detected model: {current_model_name}")
                
                # [Core Fix] Attempt to load default parameters for this model from Registry
                try:
                    from server.launch_model import MODEL_REGISTRY
                    registry_entry = None
                    # Find by ID
                    for v in MODEL_REGISTRY.values():
                        # [Fix] Must match both id (Internal ID) and name (vLLM display name)
                        if v['id'] == current_model_name or v['name'] == current_model_name:
                            registry_entry = v
                            break
                    
                    if registry_entry and 'sampling_params' in registry_entry:
                        print(f"📥 Loaded model default parameters: {registry_entry['sampling_params']}")
                        # Merge these default parameters into provider_cfg
                        provider_cfg.update(registry_entry['sampling_params'])
                except ImportError as e:
                    print(f"❌ Failed to load MODEL_REGISTRY: {e}")
                    pass
                except Exception as e:
                    print(f"❌ Unexpected error loading MODEL_REGISTRY: {e}")
                    pass
            else:
                raise RuntimeError("Service connected but returned an empty model list.")
        except Exception as e:
            print("\n" + "!"*60)
            print("❌ [CRITICAL ERROR] Failed to connect to local model service (vLLM)")
            print(f"   Error details: {e}")
            print("-" * 60)
            print("💡 Please check the following:")
            print("   1. Is 'python src/server/launch_model.py' running?")
            print("   2. Does the launch_model.py window show 'Uvicorn running on http://0.0.0.0:8000'?")
            print("   3. Check system proxy settings (e.g., no_proxy=localhost)?")
            print("   4. Do NOT close the launch_model.py window! Two terminals must run simultaneously.")
            print("!"*60 + "\n")
            return
    else:
        # Remote API: Use default_model from configuration
        current_model_name = provider_cfg.get('default_model') or "default-model"
        print(f"🤖 Using model: {current_model_name}")

    # =========================================================================
    # [Optimization] Print Inference Config in advance
    # =========================================================================
    # Calculate effectively applied sampling parameters
    actual_params = {
        'temperature': kwargs.get('temperature') if kwargs.get('temperature') is not None else provider_cfg.get('temperature', 1),
        'top_p': kwargs.get('top_p') if kwargs.get('top_p') is not None else provider_cfg.get('top_p', 1.0),
        'top_k': kwargs.get('top_k') if kwargs.get('top_k') is not None else provider_cfg.get('top_k'),
        'min_p': kwargs.get('min_p') if kwargs.get('min_p') is not None else provider_cfg.get('min_p'),
        'max_tokens': provider_cfg.get('max_tokens', 500)
    }
    # Filter out None values
    actual_params = {k: v for k, v in actual_params.items() if v is not None}
    
    print("\n" + "=" * 60)
    print(f"🧠 [Inference Config] Final effective parameters:")
    print(f"   Model       : {current_model_name}")
    print(f"   Provider    : {provider_name}")
    print(f"   Params      : {actual_params}")
    if debug_limit:
        print(f"   Debug Mode  : Only processing first {debug_limit} rows")
    print("=" * 60 + "\n")

    # 4. Load data
    df = experiment.load_data()
    
    # [New] Data sampling (Stratified by Date)
    if kwargs.get('sample_ratio'):
        ratio = kwargs['sample_ratio']
        seed = kwargs.get('seed', 42)
        print(f"🎲 Performing data sampling: Ratio={ratio}, Seed={seed}, Stratified by 'date'")
        initial_count = len(df)
        
        # Stratified sampling by date
        try:
            df = df.groupby('date', group_keys=False).apply(lambda x: x.sample(frac=ratio, random_state=seed))
            print(f"   Sampling completed: {initial_count} -> {len(df)} rows")
        except Exception as e:
            print(f"⚠️ Sampling failed: {e}, using full dataset instead.")
    
    # 5. Generate Prompts (Pass CLI parameters)
    df = experiment.prepare_prompts(df, profile_features=profile_features, env_features=env_features)
    
    # 6. Prepare System Prompt
    system_prompt = experiment.get_system_prompt()
    
    # 7. Prepare output path & Resume from breakpoint
    # Dynamically build filename: {experiment_name}_{model_name}.csv
    # Clean illegal characters in model_name (e.g., /)
    safe_model_name = current_model_name.replace("/", "-").replace("\\", "-")
    if think_mode:
        output_filename = f"{experiment_name}_{safe_model_name}_{think_mode}.csv"
    else:
        output_filename = f"{experiment_name}_{safe_model_name}.csv"

    # [New] Custom suffix (for ablation studies, etc.)
    if kwargs.get('suffix'):
        output_filename = output_filename.replace('.csv', f'_{kwargs["suffix"]}.csv')

    # [New] Custom output directory
    if kwargs.get('output_dir'):
        # Supports absolute or relative paths (relative to base_dir)
        custom_out = Path(kwargs['output_dir'])
        if custom_out.is_absolute():
            out_dir = custom_out
        else:
            out_dir = experiment.base_dir / custom_out
    else:
        out_dir = experiment.base_dir / getattr(experiment, 'default_output_dir', 'result/exp1.1')

    # Ensure directory exists
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 Created output directory: {out_dir}")

    output_path = out_dir / output_filename
    
    print(f"📝 Final output file: {output_path}")
    processed_set = set()
    
    if output_path.exists():
        try:
            existing = pd.read_csv(output_path)
            processed_set = set(zip(existing['userid'], existing['date']))
            print(f"✅ Found existing results. Processed records: {len(processed_set)}")
        except Exception:
             print("⚠️ Error reading existing results file, starting fresh.")
    
    # 8. Filter tasks to process (Experiments class may declare extra columns)
    base_cols = ['userid', 'date', 'profile_prompt', 'env_prompt']
    extra_cols = [c for c in getattr(experiment, 'extra_process_columns', []) if c in df.columns]
    df_to_process = df[base_cols + extra_cols].copy()
    
    # [Optimization] Sort by date (Prefix Caching)
    df_to_process.sort_values(by='date', inplace=True)
    
    # Filter processed
    initial_len = len(df_to_process)
    keys_to_process = set(zip(df_to_process['userid'], df_to_process['date']))
    keys_remaining = keys_to_process - processed_set
    
    if len(keys_remaining) < initial_len:
         df_to_process = df_to_process[df_to_process.apply(lambda x: (x['userid'], x['date']) in keys_remaining, axis=1)]
    
    print(f"Skipped: {initial_len - len(df_to_process)}, Remaining: {len(df_to_process)}")

    limit = config.get('debug_limit')
    if limit:
        df_to_process = df_to_process.head(limit)
        
    total_tasks = len(df_to_process)
    if total_tasks == 0:
        print("✅ All tasks completed.")
        return

    # 9. Concurrent processing
    max_concurrent = provider_cfg.get('max_concurrent', MAX_CONCURRENT_REQUESTS)
    sem = asyncio.Semaphore(max_concurrent)
    tasks = []

    print(f"🔥 Starting concurrent processing (Pool Size: {max_concurrent})...")
    
    with tqdm(total=total_tasks, desc=f"{experiment_name} LLM Processing") as pbar:
        for _, row in df_to_process.iterrows():
            task = asyncio.create_task(
                process_single_row(row, sem, pbar, experiment, system_prompt, model_name=current_model_name, think_mode=think_mode, **kwargs)
            )
            tasks.append(task)
            
        pending_results = []
        save_every = config.get('save_every', 100)
        
        for future in asyncio.as_completed(tasks):
            try:
                res = await future
                pending_results.append(res)
            except Exception as e:
                print(f"⚠️ Task exception: {e}")
                continue
            
            # Batch save
            if len(pending_results) >= save_every:
                new_df = pd.DataFrame(pending_results)
                write_header = not output_path.exists()
                new_df.to_csv(output_path, mode='a', header=write_header, index=False, escapechar='\\', doublequote=False)
                pending_results = []

        # Save remaining
        if pending_results:
            new_df = pd.DataFrame(pending_results)
            write_header = not output_path.exists()
            new_df.to_csv(output_path, mode='a', header=write_header, index=False, escapechar='\\', doublequote=False)
            
    print(f"✅ {experiment_name} experiment processing completed.")

def main():
    parser = argparse.ArgumentParser(description="SCE Experiment Runner")
    parser.add_argument('--experiment', type=str, required=True,
                        choices=list(EXPERIMENT_CLASSES.keys()),
                        help="Choose the experiment to run")
    parser.add_argument('--debug-limit', type=int, default=None,
                        help="Limit number of rows for debugging")
    parser.add_argument('--provider', type=str, default=None,
                        help="LLM provider name (e.g. local_vllm, openai_proxy, openai)")
    parser.add_argument('--think-mode', type=str, default=None,
                        choices=['think', 'no_think'],
                        help="Qwen3 thinking mode: think or no_think")
    
    # [New] Feature selection parameters
    parser.add_argument('--profile-features', type=str, default=None,
                        help="Comma-separated list of profile features to include (or 'all')")
    parser.add_argument('--env-features', type=str, default=None,
                        help="Comma-separated list of environment features to include (or 'all')")

    # [New] Sampling parameter override
    parser.add_argument('--temperature', type=float, default=None, help="Override temperature")
    parser.add_argument('--top-p', type=float, default=None, help="Override top_p")
    parser.add_argument('--top-k', type=int, default=None, help="Override top_k")
    parser.add_argument('--min-p', type=float, default=None, help="Override min_p")

    # [New] Sampling parameters (Experiment 2: Ablation Study)
    parser.add_argument('--sample-ratio', type=float, default=None, help="Ratio of data to sample per date (e.g. 0.1)")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for sampling")
    parser.add_argument('--output-dir', type=str, default=None, help="Custom output directory")
    parser.add_argument('--suffix', type=str, default=None, help="Custom suffix for output filename (e.g. NOage_SEED2026)")

    # [exp8] First-round experiment result path
    parser.add_argument('--prior-result', type=str, default=None, help="Path to prior experiment result CSV (for exp8)")

    args = parser.parse_args()

    # 解析特征列表
    p_features = args.profile_features.split(',') if args.profile_features else None
    e_features = args.env_features.split(',') if args.env_features else None

    # Qwen3 interactive prompt
    think_mode = args.think_mode
    if think_mode is None and args.provider:
        from server.config import load_provider_config as _load_cfg
        _cfg = _load_cfg(args.provider)
        _model = _cfg.get('default_model') or ''
        if 'qwen3' in _model.lower() and 'instruct' not in _model.lower():
            print("Qwen3 model detected, please choose thinking mode:")
            print("  1. think (Enable thinking)")
            print("  2. no_think (Disable thinking)")
            choice = input("Please input 1 or 2: ").strip()
            think_mode = 'think' if choice == '1' else 'no_think'

    # Windows Asyncio Policy
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(run_experiment(
        args.experiment, 
        args.debug_limit, 
        args.provider, 
        think_mode,
        profile_features=p_features,
        env_features=e_features,
        # 传递采样参数
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        # 传递采样参数
        sample_ratio=args.sample_ratio,
        seed=args.seed,
        output_dir=args.output_dir,
        suffix=args.suffix,
        # exp8: 第一轮结果路径
        prior_result=args.prior_result
    ))

if __name__ == "__main__":
    main()
