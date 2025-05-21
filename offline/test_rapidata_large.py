import os
import json
from datetime import datetime
from diffusers import StableDiffusionXLPipeline
import torch
from PIL import Image
from rapidata import RapidataClient

# Initialize the SDXL pipeline
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", 
    torch_dtype=torch.float16, 
    use_safetensors=True, 
    variant="fp16"
).to("cuda")

# Directory to store generated images
BASE_DIR = "images"

# Define prompts for batch processing
PROMPTS = [
    "A logo for a pizza place",
    "A logo for new AI startup",
    "A logo for my new basketball team called the Stars",
    "A logo for my hotdog stand",
    "A logo for my book club"
]

def generate_image_pairs():
    """Generates two SDXL images for each prompt and saves them in separate folders."""
    image_pairs = []
    
    for prompt in PROMPTS:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_path = os.path.abspath(os.path.join(BASE_DIR, f"pair_{timestamp}"))
        os.makedirs(folder_path, exist_ok=True)

        image_paths = []
        for i in range(2):  # Generate two images per prompt
            img = pipe(prompt).images[0]
            img_path = os.path.abspath(os.path.join(folder_path, f"image_{i}.png"))
            img.save(img_path)
            image_paths.append(img_path)

        print(f"✅ Images saved for prompt '{prompt}' in {folder_path}")
        image_pairs.append((prompt, image_paths, folder_path))

    return image_pairs


def create_rapidata_order(rapi: RapidataClient, image_pairs):
    """Submits a batch order to Rapidata using locally stored images."""
    prompts = [pair[0] for pair in image_pairs]
    datapoints = [pair[1] for pair in image_pairs]  # List of image path pairs
    
    order = rapi.order.create_compare_order(
        name="SDXL Image Comparison Order",
        instruction="Which logo is better?",
        contexts=prompts,  # Pass prompts as context
        datapoints=datapoints,
        responses_per_datapoint=25
    ).run()

    return order


def save_results(order, image_pairs):
    """Saves Rapidata results in each respective folder."""
    results = order.get_results()

    for i, (_, _, folder_path) in enumerate(image_pairs):
        results_path = os.path.join(folder_path, "data.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=4)

        print(f"🟢 Results saved in: {results_path}")

    return results


if __name__ == "__main__":
    rapi = RapidataClient()

    # Generate images for batch processing
    image_pairs = generate_image_pairs()

    # Submit images to Rapidata
    order = create_rapidata_order(rapi, image_pairs)

    # Track order progress
    order.display_progress_bar()

    # Save results in respective folders
    results = save_results(order, image_pairs)

    print("✅ Batch processing complete!")
