import yaml
from pathlib import Path

# =============================================================================
#  Configuration Loading
# =============================================================================

def load_experiment_config(experiment_name: str) -> dict:
    config_path = Path(__file__).parent / "experiments.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"配置文件未找到: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        full_config = yaml.safe_load(f)

    if experiment_name not in full_config['experiments']:
        raise ValueError(f"未知的实验名称: {experiment_name}. 可用实验: {list(full_config['experiments'].keys())}")

    common_cfg = full_config['common']
    exp_cfg = full_config['experiments'][experiment_name]

    merged_config = common_cfg.copy()
    merged_config.update(exp_cfg)

    return merged_config
