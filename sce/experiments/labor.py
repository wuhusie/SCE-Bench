from .base import BaseExperiment

class LaborExperiment(BaseExperiment):
    def get_system_prompt(self):
        return """
You are asked to answer a labor survey. Please answer clearly as instructed.
---
**Question**: Think about the job offers that you may receive within the coming four months. If the job with the best offer has an annual salary for the first year that equals 0.9-1.0 times your expected  annual salary, what do you think is the percent chance that you will accept the job?
---
**Output Requirement:**
Please output **only a single numerical value** representing the percentage chance (probability) based on your best estimate.
* **Range:** The value must be between 0 and 100 (inclusive).
* **Examples of format:** If you think there is a X% chance, output `X`. 
* **Constraint:** Please provide the specific number that reflects your situation.
* **Strictly NO** % symbol, words, or JSON. Just the number.
"""