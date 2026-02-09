"""
Exp3 N50: Third-Person Prompt Injection for N50 Experiments

Modifies run_N50.py and *_batch.py files to use third-person perspective.
"""
import sys
from pathlib import Path

# Determine project root
CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent.parent
SCE_DIR = SRC_DIR / "sce"
ANALYSIS_DIR = SRC_DIR / "analysis"

FILES_TO_MODIFY = {
    "run_N50.py": ANALYSIS_DIR / "expN50" / "run_N50.py",
    "credit_batch.py": SCE_DIR / "experiments" / "credit_batch.py",
    "labor_batch.py": SCE_DIR / "experiments" / "labor_batch.py",
    "spending_batch.py": SCE_DIR / "experiments" / "spending_batch.py",
}


def modify_file(filepath, replacements):
    print(f"Modifying {filepath}...")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content
        for old, new in replacements:
            if old not in content:
                print(f"  WARNING: String not found: '{old[:50]}...'")
            content = content.replace(old, new)

        if content == original_content:
            print("  No changes made.")
        else:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print("  Changes applied.")

    except Exception as e:
        print(f"  ERROR modifying {filepath}: {e}")
        sys.exit(1)


def main():
    # 1. Modify run_N50.py
    # Inject helper function and modify user prompt
    helper_function_code = (
        '\n'
        'def process_profile(text):\n'
        '    """Convert second-person profile to third-person."""\n'
        '    replace_map = [\n'
        '        ("Your", "The participant\'s"),\n'
        '        ("You live", "The participant lives"),\n'
        '        ("You don\'t own", "The participant does not own"),\n'
        '        ("You are", "The participant is"),\n'
        '        ("You have", "The participant has"),\n'
        '        ("You", "The participant")\n'
        '    ]\n'
        '    for old, new in replace_map:\n'
        '        text = text.replace(old, new)\n'
        '    return text\n\n'
        'async def process_user_batch'
    )

    run_n50_replacements = [
        # Inject helper function before process_user_batch
        (
            'async def process_user_batch',
            helper_function_code
        ),
        # Modify user prompt
        (
            '"Now you are a real person with the following demographic profile."',
            '"The following is a profile of a real participant."'
        ),
        (
            '"Answer the survey above, giving your best reasonable estimates.\\n\\n"',
            '"Based on the profile, predict how the participant would answer the survey above, giving the best reasonable estimates.\\n\\n"'
        ),
        # Use helper function for profile prompt
        (
            'f"{row[\'profile_prompt\']}"',
            'f"{process_profile(row[\'profile_prompt\'])}"'
        ),
    ]
    modify_file(FILES_TO_MODIFY["run_N50.py"], run_n50_replacements)

    # 2. Modify credit_batch.py
    credit_replacements = [
        (
            'You are asked to answer a credit survey. Please answer clearly as instructed.',
            'The participant is asked to answer a credit survey. The participant should answer clearly as instructed.'
        ),
        (
            '**what do you think is the percent chance** that you will Apply',
            '**what does the participant think is the percent chance** that they will Apply'
        ),
        (
            'from a person with your profile',
            'from a person with the participant\'s profile'
        ),
        (
            'what do you think is the percent chance that you will Apply',
            'what does the participant think is the percent chance that they will Apply'
        ),
    ]
    modify_file(FILES_TO_MODIFY["credit_batch.py"], credit_replacements)

    # 3. Modify labor_batch.py
    labor_replacements = [
        (
            'You are asked to answer a labor survey. Please answer clearly as instructed.',
            'The participant is asked to answer a labor survey. The participant should answer clearly as instructed.'
        ),
        (
            'Think about the job offers that you may receive',
            'Think about the job offers that the participant may receive'
        ),
        (
            'equals 0.9-1.0 times your expected',
            'equals 0.9-1.0 times their expected'
        ),
        (
            'what do you think is the percent chance that you will accept',
            'what does the participant think is the percent chance that they will accept'
        ),
        (
            'from a person with your profile',
            'from a person with the participant\'s profile'
        ),
    ]
    modify_file(FILES_TO_MODIFY["labor_batch.py"], labor_replacements)

    # 4. Modify spending_batch.py
    spending_replacements = [
        (
            'You are asked to answer a spending survey. Please answer clearly as instructed.',
            'The participant is asked to answer a spending survey. The participant should answer clearly as instructed.'
        ),
        (
            'Now think about your total household spending',
            "Now think about the participant's total household spending"
        ),
        (
            '(including you)',
            '(including the participant)'
        ),
        (
            'What do you expect will happen to the total spending of all members of your household',
            "What does the participant expect will happen to the total spending of all members of their household"
        ),
        (
            'from a person with your profile',
            'from a person with the participant\'s profile'
        ),
    ]
    modify_file(FILES_TO_MODIFY["spending_batch.py"], spending_replacements)

    print("\nAll N50 modifications completed for Exp3 (Third-Person).")


if __name__ == "__main__":
    main()
