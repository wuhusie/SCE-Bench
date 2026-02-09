from abc import ABC, abstractmethod
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys

# Add project root directory to Python path
# Project root directory: /root/autodl-tmp/src
# Current file: /root/autodl-tmp/src/sce/experiments/base.py
project_root = Path(__file__).parent.parent.parent  # Two levels up to src/
sys.path.insert(0, str(project_root))

from sce.utils.data_utils import load_with_cache, load_macro_indicators
from sce.utils.prompt_utils import build_profile_prompt, build_environment_prompt

class BaseExperiment(ABC):
    # --- Class attributes that can be overridden by subclasses ---
    config_name = None            # Configuration key name; uses experiment_name if None
    extra_process_columns = []    # Extra columns needed for process_single_row
    default_output_dir = "result/exp1.1"  # Default output directory (relative to base_dir)

    def __init__(self, config, common_config, experiment_name):
        """
        Initialize the experiment base class.

        Arguments:
            config: Specific configuration for the current experiment (dict).
            common_config: Common configuration (dict).
            experiment_name: Experiment name (spending/credit/labor).
        """
        self.config = config
        self.common_config = common_config
        self.experiment_name = experiment_name
        
        # Path configuration
        self.base_dir = Path(common_config['base_dir'])
        self.cache_dir = self.base_dir / common_config['cache_dir']
        
        # Parse data paths
        self.profile_path = self.base_dir / common_config['profile_path']
        self.data_path = self.base_dir / config['data_file']
        self.macro_dir = self.base_dir / common_config['macro_dir']
        
        # Parameter configuration
        self.cutoff_date = common_config.get('cutoff_date', 0)
        
    def load_data(self):
        """
        General data loading workflow:
        1. Load Profile data
        2. Load experiment-specific data
        3. Load Macro data
        4. Merge
        5. Filter
        """
        # 1. Load Profile
        profile_df = load_with_cache(
            self.profile_path, 
            "Profile Data", 
            self.cache_dir
        )
        
        # 2. Load Domain Specific Data
        domain_df = load_with_cache(
            self.data_path, 
            f"{self.experiment_name.capitalize()} Data", 
            self.cache_dir
        )
        
        # 3. Load Macro Data
        self.macro_data = load_macro_indicators(self.macro_dir, self.cache_dir)
        
        # 4. Merge
        print(f"[{self.experiment_name}] Merging data...")
        merged_df = pd.merge(
            profile_df,
            domain_df,
            on=['userid', 'date'],
            how='inner'
        )
        
        # 5. Filter by Date
        if "date" in merged_df.columns:
            merged_df['date_int'] = pd.to_numeric(merged_df['date'], errors='coerce').fillna(0).astype(int)
            merged_df = merged_df[merged_df["date_int"] > self.cutoff_date]
            merged_df = merged_df.drop(columns=['date_int'])
            print(f"Data filtering completed (Date > {self.cutoff_date}). Remaining records: {len(merged_df)}")
            
        return merged_df

    def prepare_prompts(self, df, profile_features=None, env_features=None):
        """
        Construct all Prompts and add to DataFrame (no disk caching, real-time construction).
        """
        # Removed original prompt_cache_path logic
        # Because dynamic feature selection is now supported, and construction speed Is extremely fast after optimization, caching prompt strings is no longer required.
        
        print("⚡ [On-the-fly] Starting real-time prompt generation (O(1) optimization enabled)...")
        
        # 1. Optimize macro data (DataFrame -> Dict)
        if not hasattr(self, 'macro_data_optimized'):
             from sce.utils.prompt_utils import optimize_macro_data
             self.macro_data_optimized = optimize_macro_data(self.macro_data)

        # 2. Generate Profile Prompts
        print(f"   - Generating profile prompts (Features: {profile_features or 'All'})...")
        tqdm.pandas(desc="Profile Prompt")
        df['profile_prompt'] = df.progress_apply(
            lambda row: build_profile_prompt(row, selected_features=profile_features), 
            axis=1
        )
        
        # 3. Generate Environment Prompts
        print(f"   - Generating environment prompts (Features: {env_features or 'All'})...")
        tqdm.pandas(desc="Environment Prompt")
        df['env_prompt'] = df.progress_apply(
            lambda row: build_environment_prompt(
                row.get('date'), 
                self.macro_data_optimized, 
                selected_features=env_features
            ), 
            axis=1
        )
        
        return df

    def build_row_prompts(self, row, system_prompt):
        """
        Construct (final_system_prompt, user_prompt, extra_output_fields) for a single row.
        Can be overridden by subclasses to change prompt structure.

        Default behavior (exp1-1):
            system = questionnaire question
            user   = env + profile
        """
        user_prompt = (
            f"{row['env_prompt']}\n\n"
            "Now you are a real person with the following demographic profile."
            "Answer the survey above, giving your best reasonable estimates.\n\n"
            f"{row['profile_prompt']}"
        )
        return system_prompt, user_prompt, {}

    @abstractmethod
    def get_system_prompt(self):
        """
        Get the System Instruction Prompt for the current experiment.
        Must be implemented by subclasses.
        """
        pass
