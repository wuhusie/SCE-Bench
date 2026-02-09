from .base import BaseExperiment

class SpendingExperiment(BaseExperiment):
    def get_system_prompt(self):
        return """
You are asked to answer a spending survey. Please answer clearly as instructed.
---
Now think about your total household spending, including groceries, clothing, personal care, housing (such as rent, mortgage payments, utilities, maintenance, home improvements), medical expenses (including health insurance), transportation, recreation and entertainment, education, and any large items (such as home appliances, electronics, furniture, or car payments).
---
**Question**: What do you expect will happen to the total spending of all members of your household (including you) over the next 12 months?
---
**Output Requirement:**
Please output **only a single numerical value** representing the percentage change based on your best estimate.
* **Format:** Positive number for increase, negative number for decrease.
* **Examples of format:** If you expect a X% increase, output `X`. If you expect a X% decrease, output `-X`.
* **Constraint:** Please provide the specific number that reflects your situation.
* **Strictly NO** % symbol, words, or JSON. Just the number.
"""