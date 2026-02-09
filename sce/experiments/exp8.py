# -*- coding: utf-8 -*-
"""
Experiment 8: LLM Response Memory

Based on the exp1-1 baseline experiment, if the user has the model response from the previous period (t-1),
then memory information is appended to the end of the user prompt.

Difference from exp9:
  - exp9: previousGt = Human ground truth response (calculated by shifting the GT column)
  - exp8: previousLLM = Model's previous response (searched from the first-round experiment result file)

Usage Process:
  1. Run the exp1-1 full experiment first to obtain the result file.
  2. Run exp8, specifying the first-round result file path via the --prior-result argument.
"""

import pandas as pd
from pathlib import Path
from .spending import SpendingExperiment
from .credit import CreditExperiment
from .labor import LaborExperiment

# ---------- Memory Induction ----------
MEMORY_INTRO = (
    "\n\nWhen making the previous decision four months ago, the responses of you "
    "are as follows. Consider the environmental changes over the past four months "
    "to determine your decision outcome this time:\n"
)


def _is_missing(value):
    """Check if previousLLM is missing"""
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    return False


def _build_prior_response_map(prior_result_path):
    """
    Construct (userid, date) → llm_response mapping from the first-round experiment result file.

    Returns:
        dict: {(userid, date): llm_response}
    """
    if not prior_result_path or not Path(prior_result_path).exists():
        print(f"[exp8] WARNING: First-round result file does not exist: {prior_result_path}")
        return {}

    print(f"[exp8] Loading first-round results: {prior_result_path}")
    prior_df = pd.read_csv(prior_result_path)

    # 构建映射
    response_map = {}
    for _, row in prior_df.iterrows():
        key = (row['userid'], row['date'])
        response_map[key] = row['llm_response']

    print(f"[exp8] Constructed {len(response_map)} mapping entries of (userid, date) → llm_response")
    return response_map


# ============================================================
# Mixin: Shared logic for all exp8 experiment classes
# ============================================================
class Exp8Mixin:
    """
    Inject two capabilities via Mixin:
      1. Calculate previousLLM based on first-round results after load_data()
      2. build_row_prompts() appends memory (if any) to the end of user prompt
    """
    extra_process_columns = ['previousLLM']
    default_output_dir = "result/exp8"

    # Subclass or runtime settings
    prior_result_path = None  # Path to the first-round experiment results
    def load_data(self):
        df = super().load_data()

        # Check if first-round result path is set
        prior_path = getattr(self, 'prior_result_path', None)
        if not prior_path:
            print(f"[exp8] WARNING: prior_result_path not set, previousLLM will all be empty")
            df['previousLLM'] = pd.NA
            return df

        # Build first-round result mapping
        response_map = _build_prior_response_map(prior_path)
        if not response_map:
            df['previousLLM'] = pd.NA
            return df

        # Sort by userid and date
        df.sort_values(['userid', 'date'], inplace=True)

        # Get all dates for each user to find the previous period
        print(f"[exp8] Calculating previousLLM...")

        def get_previous_llm(row, user_dates_map, response_map):
            """Find the llm_response of the user's previous period"""
            userid = row['userid']
            current_date = row['date']

            # Get all dates for this user (sorted)
            user_dates = user_dates_map.get(userid, [])

            # Find position of current date in the list
            try:
                idx = user_dates.index(current_date)
            except ValueError:
                return pd.NA

            # If it's the first period, there's no previous period
            if idx == 0:
                return pd.NA

            # Get date of the previous period
            prev_date = user_dates[idx - 1]

            # Find previous llm_response from mapping
            return response_map.get((userid, prev_date), pd.NA)

        # Pre-build date list for each user
        user_dates_map = df.groupby('userid')['date'].apply(lambda x: sorted(x.unique().tolist())).to_dict()

        # Calculate previousLLM
        df['previousLLM'] = df.apply(
            lambda row: get_previous_llm(row, user_dates_map, response_map),
            axis=1
        )

        n_with = df['previousLLM'].notna().sum()
        print(f"[exp8] With memory: {n_with}, No memory (first time): {len(df) - n_with}")

        return df

    def build_row_prompts(self, row, system_prompt):
        """
        exp8 prompt construction (consistent with exp1-1 structure, only appends memory at the end):
          system = questionnaire question (unchanged)
          user   = env + profile [+ memory intro + previousLLM]
        """
        # Call base class default logic to construct user prompt
        final_system, base_user, _ = super().build_row_prompts(row, system_prompt)

        # Check for previous period model response
        previous_llm = row.get('previousLLM')
        has_memory = not _is_missing(previous_llm)

        # If memory exists, append to the end of user prompt
        if has_memory:
            final_user = f"{base_user}{MEMORY_INTRO}{previous_llm}"
        else:
            final_user = base_user

        extra = {
            'has_memory': has_memory,
            'previousLLM': str(previous_llm) if has_memory else None,
        }
        return final_system, final_user, extra


# ============================================================
# Specific Experiment Classes (Inherit from Mixin + corresponding baseline experiment)
# ============================================================
class Exp8SpendingExperiment(Exp8Mixin, SpendingExperiment):
    config_name = 'spending'


class Exp8CreditExperiment(Exp8Mixin, CreditExperiment):
    config_name = 'credit'


class Exp8LaborExperiment(Exp8Mixin, LaborExperiment):
    config_name = 'labor'


# ============================================================
# Batch Versions (For N50 distribution probing)
# ============================================================
from .spending_batch import SpendingBatchExperiment
from .credit_batch import CreditBatchExperiment
from .labor_batch import LaborBatchExperiment


class Exp8SpendingBatchExperiment(Exp8Mixin, SpendingBatchExperiment):
    config_name = 'spending'


class Exp8CreditBatchExperiment(Exp8Mixin, CreditBatchExperiment):
    config_name = 'credit'


class Exp8LaborBatchExperiment(Exp8Mixin, LaborBatchExperiment):
    config_name = 'labor'
