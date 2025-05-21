import os
import csv
import json



# Constants
IMAGE_DIR = "images_finetuned"
METADATA_DIR = "metadata_finetuned"
PROMPTS_FILE = "prompts.txt"
MAPPING_FILE = os.path.join(METADATA_DIR, "image_mapping.json")

def recreate_metadata():
    image_mapping = {"data": []}
    image_files = sorted(f for f in os.listdir(IMAGE_DIR) if f.endswith(".png"))
    print(f"Found {len(image_files)} image files in '{IMAGE_DIR}'.")

    # Read prompts.txt, one prompt per non-empty line
    with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]

    index = 0
    for i, prompt in enumerate(prompts, start=1):
        if index + 1 >= len(image_files):
            print(f"⚠️ Not enough images left to pair with prompt #{i}: '{prompt}'")
            break

        img1 = image_files[index]
        img2 = image_files[index + 1]

        image_mapping["data"].append({
            "prompt": prompt,
            "images": [img1, img2]
        })

        index += 2

    with open(MAPPING_FILE, "w", encoding="utf-8") as f:
        json.dump(image_mapping, f, indent=4, ensure_ascii=False)

    print(f"✅ Recreated metadata and saved to {MAPPING_FILE}")


if __name__ == "__main__":
    recreate_metadata()
