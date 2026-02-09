import sys
import pandas as pd
import json
from pathlib import Path

# Add src to path
# This script is in src/analysis/exp4, so src is ../../
SRC_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SRC_DIR))

try:
    from sce.utils import prompt_utils
except ImportError:
    print("Error importing sce.utils.prompt_utils. Ensure src is in PYTHONPATH.")
    sys.exit(1)

def verify():
    print("--- Verifying Profile JSON ---")
    row = {
        'Q32': 74,
        'Q33': 1,   # Female
        'Q36': 3,   # Some college
        'Q38': 1,   # Married
        '_STATE': 'TX', 'Q42': 20,
        'Q43': 1,   # Own
        'Q44': 2,   # No other home
        'Q45b': 4,  # Fair (Health)
        'Q10_7': 1, # Retired
        'Q47': 4    # 30-40k
    }
    
    profile_json = prompt_utils.build_profile_prompt(row)
    print(f"Raw Output: {profile_json}")
    try:
        parsed = json.loads(profile_json)
        print("✅ Profile JSON is valid.")
        print(json.dumps(parsed, indent=2))
    except json.JSONDecodeError:
        print("❌ Profile Output is NOT valid JSON.")

    print("\n--- Verifying Environment JSON ---")
    # Mock macro data structure
    # { 'cpi': {'date': val}, ... } optimized
    mock_macro = {
        'cpi': {pd.Timestamp('2024-01-01'): 1.2, pd.Timestamp('2023-12-01'): 1.1, pd.Timestamp('2023-11-01'): 1.0, pd.Timestamp('2023-10-01'): 0.9},
        'interest': {pd.Timestamp('2024-01-01'): 5.0, pd.Timestamp('2023-12-01'): 5.0, pd.Timestamp('2023-11-01'): 4.9, pd.Timestamp('2023-10-01'): 4.9},
        'unemployment': {pd.Timestamp('2024-01-01'): 3.5, pd.Timestamp('2023-12-01'): 3.6, pd.Timestamp('2023-11-01'): 3.7, pd.Timestamp('2023-10-01'): 3.8}
    }
    
    # Date: 202402 (so ref_date 2024-02-01, past 4 months: Jan, Dec, Nov, Oct)
    env_json = prompt_utils.build_environment_prompt(202402, mock_macro)
    print(f"Raw Output: {env_json}")
    try:
        parsed = json.loads(env_json)
        print("✅ Environment JSON is valid.")
        print(json.dumps(parsed, indent=2))
    except json.JSONDecodeError:
        print("❌ Environment Output is NOT valid JSON.")

if __name__ == "__main__":
    verify()
