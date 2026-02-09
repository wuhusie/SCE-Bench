from .base import BaseExperiment

class CreditExperiment(BaseExperiment):
    def get_system_prompt(self):
        return """
You are asked to answer a credit survey. Please answer clearly as instructed.
---
**Question**: Over the next 12 months, **what do you think is the percent chance** that you will Apply for a mortgage or home-based loan?
---
**Output Requirement:**
Please output **only a single numerical value** representing the probability (0-100).
* **Format:** A number between 0 and 100.
* **Examples of format:** If you think there is a X% chance, output `X`. 
* **Constraint:** Please provide the specific number that reflects your situation.
* **Strictly NO** % symbol, words, or JSON. Just the number.
"""