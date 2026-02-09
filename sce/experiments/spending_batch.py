"""
Spending Experiment - Batch Mode (for Distribution Probe)
Output Format: JSON list, supports n_samples parameter
"""
from .base import BaseExperiment


class SpendingBatchExperiment(BaseExperiment):
    """Spending Experiment - Batch Sampling Mode"""
    
    def __init__(self, config, common_config, experiment_name, n_samples=50):
        super().__init__(config, common_config, experiment_name)
        self.n_samples = n_samples
    
    def get_system_prompt(self):
        return f"""The participant is asked to answer a spending survey. The participant should answer clearly as instructed.
---
Now think about the participant's total household spending, including groceries, clothing, personal care, housing (such as rent, mortgage payments, utilities, maintenance, home improvements), medical expenses (including health insurance), transportation, recreation and entertainment, education, and any large items (such as home appliances, electronics, furniture, or car payments).
---
**Question**: What does the participant expect will happen to the total spending of all members of their household (including the participant) over the next 12 months?
---
**Output Requirement:**
To simulate variability, generate {self.n_samples} independent samples of percentage change estimates.
Each sample should represent one possible answer from a person with the participant's profile.
* **Format:** Output a list of {self.n_samples} numbers. Positive for increase, negative for decrease.
* **Example:** [X1,X2, ...]
* **Constraint:** Output ONLY the list. No explanation, no text, no markdown."""

    def get_question_prompt(self):
        """Returns the question text (for use in user_prompt)"""
        return "What does the participant expect will happen to the total spending of all members of their household (including the participant) over the next 12 months? (Express as percentage change)"
