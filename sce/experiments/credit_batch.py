"""
Credit Experiment - Batch Mode (for Distribution Probe)
Output Format: JSON list, supports n_samples parameter
"""
from .base import BaseExperiment


class CreditBatchExperiment(BaseExperiment):
    """Credit Experiment - Batch Sampling Mode"""
    
    def __init__(self, config, common_config, experiment_name, n_samples=50):
        super().__init__(config, common_config, experiment_name)
        self.n_samples = n_samples
    
    def get_system_prompt(self):
        return f"""The participant is asked to answer a credit survey. The participant should answer clearly as instructed.
---
**Question**: Over the next 12 months, **what does the participant think is the percent chance** that they will Apply for a mortgage or home-based loan?
---
**Output Requirement:**
To simulate variability, generate {self.n_samples} independent samples of probability estimates.
Each sample should represent one possible answer from a person with the participant's profile.
* **Format:** Output a list of {self.n_samples} numbers, each between 0 and 100.
* **Example:** [X1, X2, ...]
* **Constraint:** Output ONLY the list. No explanation, no text, no markdown."""

    def get_question_prompt(self):
        """Returns the question text (for use in user_prompt)"""
        return "Over the next 12 months, what does the participant think is the percent chance that they will Apply for a mortgage or home-based loan?"
