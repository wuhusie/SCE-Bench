# -*- coding: utf-8 -*-
"""
Experiment 9: Real Human Answer Memory

Based on the exp1-1 baseline experiment, if the user has the actual human answer from the previous period (t-1),
then memory information is appended to the end of the user prompt.

Prompt structure is identical to exp1-1:
  - System prompt: Questionnaire questions
  - User prompt: env + profile [+ memory intro + previousGt]
"""

import pandas as pd
from .spending import SpendingExperiment
from .credit import CreditExperiment
from .labor import LaborExperiment

# ---------- Memory Induction ----------
MEMORY_INTRO = (
    "\n\nWhen making the previous decision four months ago, the responses from "
    "real individuals with the same profile background as yours were as follows. "
    "You should carefully review and learn from these responses, then based on "
    "these references, consider the environmental changes over the past four "
    "months to determine your decision outcome this time:\n"
)


def _is_missing(value):
    """Check if previousGt is missing"""
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    return False


# ============================================================
# Mixin: Shared logic for all exp9 experiment classes
# ============================================================
class Exp9Mixin:
    """
    Inject two capabilities via Mixin:
      1. Automatically calculate previousGt after load_data()
      2. build_row_prompts() appends memory (if any) to the end of user prompt
    """
    extra_process_columns = ['previousGt']
    default_output_dir = "result/exp9"
    GT_COLUMN = None   # Must be set by subclasses

    def load_data(self):
        df = super().load_data()

        gt_col = self.GT_COLUMN
        if 'previousGt' in df.columns:
            print(f"[exp9] Using existing 'previousGt' column from data")
        elif gt_col and gt_col in df.columns:
            print(f"[exp9] Calculating previousGt from '{gt_col}' (groupby userid, shift 1)...")
            df.sort_values(['userid', 'date'], inplace=True)
            df['previousGt'] = df.groupby('userid')[gt_col].shift(1)
            n_with = df['previousGt'].notna().sum()
            print(f"[exp9] With memory: {n_with}, No memory (first time): {len(df) - n_with}")
        else:
            print(f"[exp9] WARNING: GT column '{gt_col}' not found, previousGt will all be empty")
            df['previousGt'] = pd.NA

        return df

    def build_row_prompts(self, row, system_prompt):
        """
        exp9 prompt construction (consistent with exp1-1 structure, only appends memory at the end):
          system = questionnaire question (unchanged)
          user   = env + profile [+ memory intro + previousGt]
        """
        # Call base class default logic to construct user prompt
        final_system, base_user, _ = super().build_row_prompts(row, system_prompt)

        # Check for previous period human ground truth
        previous_gt = row.get('previousGt')
        has_memory = not _is_missing(previous_gt)

        # If memory exists, append to the end of user prompt
        if has_memory:
            final_user = f"{base_user}{MEMORY_INTRO}{previous_gt}"
        else:
            final_user = base_user

        extra = {
            'has_memory': has_memory,
            'previousGt': str(previous_gt) if has_memory else None,
        }
        return final_system, final_user, extra


# ============================================================
# Specific Experiment Classes (Inherit from Mixin + corresponding baseline experiment)
# ============================================================
class Exp9SpendingExperiment(Exp9Mixin, SpendingExperiment):
    config_name = 'spending'
    GT_COLUMN = 'Q26v2part2'


class Exp9CreditExperiment(Exp9Mixin, CreditExperiment):
    config_name = 'credit'
    GT_COLUMN = 'N17b_2'


class Exp9LaborExperiment(Exp9Mixin, LaborExperiment):
    config_name = 'labor'
    GT_COLUMN = 'oo2c3'


# ============================================================
# Batch Versions (For N50 distribution probing)
# ============================================================
from .spending_batch import SpendingBatchExperiment
from .credit_batch import CreditBatchExperiment
from .labor_batch import LaborBatchExperiment


class Exp9SpendingBatchExperiment(Exp9Mixin, SpendingBatchExperiment):
    config_name = 'spending'
    GT_COLUMN = 'Q26v2part2'


class Exp9CreditBatchExperiment(Exp9Mixin, CreditBatchExperiment):
    config_name = 'credit'
    GT_COLUMN = 'N17b_2'


class Exp9LaborBatchExperiment(Exp9Mixin, LaborBatchExperiment):
    config_name = 'labor'
    GT_COLUMN = 'oo2c3'
