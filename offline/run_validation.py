import os
import json
import csv
from datetime import datetime
from diffusers import StableDiffusionXLPipeline
import torch
from PIL import Image
from rapidata import RapidataClient

# Define storage directories
IMAGE_DIR = "images_validation"  # Updated directory name
METADATA_DIR = "metadata_validation"  # Updated directory name
ORDER_RESULTS_DIR = "validation_results"

os.makedirs(ORDER_RESULTS_DIR, exist_ok=True)

# Path for metadata files
MAPPING_FILE = os.path.join(METADATA_DIR, "image_mapping_validation.json")  # Updated file name
TRUTHS_FILE = os.path.join(METADATA_DIR, "better_images_validation.json")  # Updated file name

PROMPTS_CSV = os.path.join(METADATA_DIR, "prompts_validation.csv")  # Updated file name


def load_image_mapping(file_path):
    """Loads image mapping from JSON file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ Mapping file not found: {file_path}")

    with open(file_path, "r") as f:
        image_mapping = json.load(f)

    print(f"🟢 Loaded image mapping from {file_path}")
    return image_mapping

def create_rapidata_validation(rapi: RapidataClient):
    """Reads the JSON mapping and submits a batch order to Rapidata."""
    image_mapping = load_image_mapping(MAPPING_FILE)
    truths_mapping = load_image_mapping(TRUTHS_FILE)


    prompts = [entry["original_prompt"] for entry in image_mapping["data"]] #gets the original prompt from the mapping file.
    truths = [entry["better_image"] for entry in truths_mapping["data"]] #gets the original prompt from the mapping file.

    datapoints = [[os.path.join(IMAGE_DIR, img) for img in entry["images"]] for entry in image_mapping["data"]]

    validation_set = rapi.validation.create_compare_set(
     name="Example Compare Validation Set",
     instruction="Which image follows the prompt more accurately?",
     contexts=prompts,
     datapoints=datapoints, 
     truths=truths
    )


if __name__ == "__main__":
    rapi = RapidataClient()
    # Step 2: Submit order to Rapidata
    order = create_rapidata_validation(rapi)

    print("✅ Validation complete!")