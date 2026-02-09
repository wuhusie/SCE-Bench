import os
import yaml
from pathlib import Path

# =============================================================================
# LLM Provider Configuration
# =============================================================================

def load_provider_config(provider_name=None):
    """
    Load provider configuration from llm_providers.yaml.

    Arguments:
        provider_name: Specified provider name; if None, uses active_provider from yaml.

    Returns:
        dict: Contains api_base, api_key, default_model, max_tokens, temperature
    """
    config_path = Path(__file__).parent / "llm_providers.yaml"

    with open(config_path, 'r', encoding='utf-8') as f:
        raw = yaml.safe_load(f)

    if provider_name is None:
        provider_name = raw['active_provider']

    # [Modified] Dynamic Provider generation logic
    # Check if a specific local model is requested (Format: local_MODELID)
    if provider_name.startswith("local_") and provider_name != "local_vllm":
         # Attempt to parse from MODEL_REGISTRY
        try:
            from server.launch_model import MODEL_REGISTRY
            model_id = provider_name.replace("local_", "")
            
            # Find in Registry (iterate through values or directly keys)
            registry_entry = None
            for v in MODEL_REGISTRY.values():
                if v['id'] == model_id:
                    registry_entry = v
                    break
            
            if registry_entry:
                # Model found, clone configuration based on local_vllm
                base_cfg = raw['providers']['local_vllm'].copy()
                base_cfg['provider_name'] = provider_name
                base_cfg['default_model'] = registry_entry['id'] # Bind model ID
                
                # Merge sampling parameters from registry
                if 'sampling_params' in registry_entry:
                    base_cfg.update(registry_entry['sampling_params'])
                    
                return base_cfg
                
        except ImportError:
            pass # server.launch_model might be unavailable

    providers = raw['providers']
    if provider_name not in providers:
        raise ValueError(f"Unknown provider: {provider_name}. Available: {list(providers.keys())}")

    cfg = providers[provider_name].copy()

    # Parse API key: Environment variable priority, fallback to literal value
    if 'api_key_env' in cfg:
        env_key = os.environ.get(cfg['api_key_env'])
        if env_key:
            cfg['api_key'] = env_key
        elif 'api_key' not in cfg:
            cfg['api_key'] = ''
        del cfg['api_key_env']

    cfg['provider_name'] = provider_name
    return cfg


# Load default provider configuration
ACTIVE_PROVIDER = load_provider_config()

# Backward compatibility variables
OPENAI_API_BASE = ACTIVE_PROVIDER['api_base']
OPENAI_API_KEY = ACTIVE_PROVIDER['api_key']
MODEL_NAME = ACTIVE_PROVIDER.get('default_model') or "default-model"
MAX_CONCURRENT_REQUESTS = ACTIVE_PROVIDER.get('max_concurrent', 150)
