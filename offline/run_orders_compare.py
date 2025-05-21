import os
import json
import csv
from datetime import datetime
from diffusers import StableDiffusionXLPipeline
import torch
from PIL import Image
from rapidata import RapidataClient

# Define storage directories
IMAGE_DIR = "images_compare"
METADATA_DIR = "metadata_compare"
ORDER_RESULTS_DIR = "order_results"

os.makedirs(ORDER_RESULTS_DIR, exist_ok=True)

# Path for metadata files
MAPPING_FILE = os.path.join(METADATA_DIR, "image_mapping.json")
PROMPTS_CSV = os.path.join(METADATA_DIR, "prompts.csv")


def load_image_mapping():
    """Loads image mapping from JSON file."""
    if not os.path.exists(MAPPING_FILE):
        raise FileNotFoundError(f"❌ Mapping file not found: {MAPPING_FILE}")

    with open(MAPPING_FILE, "r") as f:
        image_mapping = json.load(f)

    print(f"🟢 Loaded image mapping from {MAPPING_FILE}")
    return image_mapping


def create_rapidata_order(rapi: RapidataClient):
    """Reads the JSON mapping and submits a batch order to Rapidata."""
    image_mapping = load_image_mapping()

    prompts = [entry["prompt"] for entry in image_mapping["data"]]
    datapoints = [[os.path.join(IMAGE_DIR, img) for img in entry["images"]] for entry in image_mapping["data"]]
    
    # validation_set = rapi.validation.find_validation_sets(name="Example Compare Validation Set")
    # print(validation_set)

    order = rapi.order.create_compare_order(
        name="3000 Prompts offline comparison",
        instruction="Which logo fits the description better?",
        contexts=prompts,  # Pass prompts as context
        datapoints=datapoints,
        responses_per_datapoint=11,
        validation_set_id="67f002c7075c672e0bfe36e6"

    ).run()

    return order


def save_order_results(order):
    """Saves Rapidata order results in the order_results folder."""
    results = order.get_results()

    order_filename = f"order_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    results_path = os.path.join(ORDER_RESULTS_DIR, order_filename)

    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"🟢 Order results saved at {results_path}")
    return results_path


if __name__ == "__main__":
    rapi = RapidataClient()
    # Step 2: Submit order to Rapidata
    order = create_rapidata_order(rapi)

    # Step 3: Track order progress
    order.display_progress_bar()

    # Step 4: Save order results
    save_order_results(order)

    print("✅ Batch processing complete!")
 