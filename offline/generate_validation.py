import os
import json
import csv
from datetime import datetime
from diffusers import StableDiffusionXLPipeline
import torch
from PIL import Image

# Define storage directories
IMAGE_DIR = "images_validation"
METADATA_DIR = "metadata_validation"

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(METADATA_DIR, exist_ok=True)

# Path for metadata files
MAPPING_FILE = os.path.join(METADATA_DIR, "image_mapping_validation.json")
PROMPTS_CSV = os.path.join(METADATA_DIR, "prompts_validation.csv")
BETTER_IMAGES_JSON = os.path.join(METADATA_DIR, "better_images_validation.json")

# Initialize SDXL pipeline
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", 
    torch_dtype=torch.float16, 
    use_safetensors=True, 
    variant="fp16"
).to("cuda")

def modify_prompt(prompt):
    """Modifies the prompt to create a slightly different version."""
    if "pizza restaurant" in prompt:
        return prompt.replace("pizza restaurant", "thai restaurant")
    elif "basketball team" in prompt:
        return prompt.replace("basketball team", "soccer team")
    elif "book club" in prompt:
        return prompt.replace("book club", "movie club")
    else:
        return prompt + " (variation)" # default if no specific replacement

def generate_images_and_metadata():
    """Generates one image per original and modified prompt, stores them, and creates metadata."""
    prompts = [
        "A logo for my new pizza restaurant",
        "A logo for a new basketball team called the Stars",
        "A logo for a book club"
    ]

    image_mapping = {"data": []}
    better_images = {"data": []} # store which image is better
    
    with open(PROMPTS_CSV, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Original Prompt", "Modified Prompt"])  # CSV header

        for i, original_prompt in enumerate(prompts):
            modified_prompt = modify_prompt(original_prompt)

            image_ids = [f"image_{i*2+1:04d}.png", f"image_{i*2+2:04d}.png"]
            image_paths = [os.path.join(IMAGE_DIR, img_id) for img_id in image_ids]

            # Generate images
            img1 = pipe(original_prompt).images[0]
            img1.save(image_paths[0])
            img2 = pipe(modified_prompt).images[0]
            img2.save(image_paths[1])

            # Store mapping info
            image_mapping["data"].append({
                "original_prompt": original_prompt,
                "modified_prompt": modified_prompt,
                "images": image_ids  # Only store filenames, not full paths
            })
            
            #store better image info
            better_images["data"].append({
                "original_prompt":original_prompt,
                "better_image":image_ids[0]
            })

            # Write to CSV
            writer.writerow([original_prompt, modified_prompt])

            print(f"✅ Images saved: {image_paths}")

    # Save mapping JSON
    with open(MAPPING_FILE, "w") as f:
        json.dump(image_mapping, f, indent=4)

    #save better images json
    with open(BETTER_IMAGES_JSON, "w") as f:
        json.dump(better_images, f, indent=4)

    print(f"🟢 Image mapping saved at {MAPPING_FILE}")
    print(f"🟢 Prompts saved at {PROMPTS_CSV}")
    print(f"🟢 Better images saved at {BETTER_IMAGES_JSON}")

if __name__ == "__main__":
    generate_images_and_metadata()