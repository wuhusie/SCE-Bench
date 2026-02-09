import os
import sys
from pathlib import Path

# Determine project root relative to this script
CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent.parent
SCE_DIR = SRC_DIR / "sce"

def get_file_path(filename):
    if filename == "main.py":
        return SCE_DIR / "main.py"
    else:
        return SCE_DIR / "experiments" / filename

FILES_TO_MODIFY = {
    "main.py": get_file_path("main.py"),
    "credit.py": get_file_path("credit.py"),
    "labor.py": get_file_path("labor.py"),
    "spending.py": get_file_path("spending.py"),
}

def modify_file(filepath, replacements):
    print(f"Modifying {filepath}...")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        for old, new in replacements:
            if old not in content:
                print(f"  WARNING: String not found: '{old}'")
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
    # 1. modify main.py
    
    # Define the helper function to inject (condensed layout)
    helper_function_code = (
        '\n'
        'def process_profile(text):\n'
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
        'async def process_single_row'
    )

    main_replacements = [
        # Inject helper function
        (
            'async def process_single_row',
            helper_function_code
        ),
        # Replace introductory text
        (
            'Now you are a real person with the following demographic profile.', 
            'The following is a profile of a real participant.'
        ),
        (
            'Answer the survey above, giving your best reasonable estimates.', 
            'Based on the profile, predict how the participant with such a profile would answer the survey above, giving the best reasonable estimates.'
        ),
        # Use helper function for profile prompt
        (
            "f\"{row['profile_prompt']}\"",
            "f\"{process_profile(row['profile_prompt'])}\""
        )
    ]
    modify_file(FILES_TO_MODIFY["main.py"], main_replacements)

    # 2. modify credit.py
    credit_replacements = [
        ('You are asked to answer a credit survey. Please answer clearly as instructed.', 
         'The participant is asked to answer a credit survey. The participant should answer clearly as instructed.'),
        ('**what do you think is the percent chance** that you will Apply', 
         '**what does the participant think is the percent chance** that they will Apply'),
        ('reflects your situation', 'reflects their situation'),
        ('If you think there is', 'If the participant thinks there is') 
    ]
    modify_file(FILES_TO_MODIFY["credit.py"], credit_replacements)

    # 3. modify labor.py
    labor_replacements = [
        ('You are asked to answer a labor survey. Please answer clearly as instructed.', 
         'The participant is asked to answer a labor survey. The participant should answer clearly as instructed.'),
        ('Think about the job offers that you may receive', 
         'Think about the job offers that the participant may receive'),
        ('equals 0.9-1.0 times your expected', 'equals 0.9-1.0 times their expected'),
        ('what do you think is the percent chance that you will accept', 
         'what does the participant think is the percent chance that they will accept'),
        ('reflects your situation', 'reflects their situation'),
        ('If you think there is', 'If the participant thinks there is'),
        ('based on your best estimate', "based on the participant's best estimate")
    ]
    modify_file(FILES_TO_MODIFY["labor.py"], labor_replacements)

    # 4. modify spending.py
    spending_replacements = [
        ('You are asked to answer a spending survey. Please answer clearly as instructed.', 
         'The participant is asked to answer a spending survey. The participant should answer clearly as instructed.'),
        ('Now think about your total household spending', 
         "Now think about the participant's total household spending"),
        ('(including you)', '(including the participant)'),
        ('What do you expect will happen', 'What does the participant expect will happen'),
        ('members of your household', 'members of their household'),
        ('based on your best estimate', "based on the participant's best estimate"),
        ('If you expect a', 'If the participant expects a'),
        ('reflects your situation', 'reflects their situation')
    ]
    modify_file(FILES_TO_MODIFY["spending.py"], spending_replacements)

    print("All modifications completed.")

if __name__ == "__main__":
    main()
