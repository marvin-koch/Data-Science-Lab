import os
from pathlib import Path

# Define the path to the prompts text file relative to this script's location
PROMPTS_FILE_PATH = Path(__file__).parent / "prompts.txt"

def load_prompts_from_file(file_path: Path) -> list[str]:
    """
    Reads prompts from a text file, where each non-empty line is a prompt.

    Args:
        file_path: The Path object pointing to the text file.

    Returns:
        A list of prompts (strings). Returns an empty list if the file
        is not found or an error occurs.
    """
    prompts = []
    if not file_path.is_file():
        print(f"❌ Error: Prompts file not found at '{file_path}'")
        print("Please create the prompts.txt file with one prompt per line in the same directory.")
        # Or raise FileNotFoundError("Prompts file not found") if you prefer the script to halt immediately
        return []

    try:
        print(f"Attempting to load prompts from: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Clean up lines: strip leading/trailing whitespace and ignore empty lines
        prompts = [line.strip() for line in lines if line.strip()]

        if not prompts:
             print(f"⚠️ Warning: No valid prompts found in {file_path}. The file might be empty or contain only whitespace.")
        else:
             print(f"✅ Successfully loaded {len(prompts)} prompts from {file_path}")

    except Exception as e:
        print(f"❌ Error reading prompts file {file_path}: {e}")
        return [] # Return empty list on error

    return prompts

# Load the prompts into the required variable name when this module is imported
PROMPT_LIST = load_prompts_from_file(PROMPTS_FILE_PATH)

if not PROMPT_LIST:
    raise ValueError("Failed to load any prompts. Halting execution.")
