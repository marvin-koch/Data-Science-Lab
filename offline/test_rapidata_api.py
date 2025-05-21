from diffusers import StableDiffusionXLPipeline
import torch
from PIL import Image
from datetime import datetime
import os

# Initialize the SDXL pipeline
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", 
    torch_dtype=torch.float16, 
    use_safetensors=True, 
    variant="fp16"
).to("cuda")

# Prompt
prompt = "A logo for a new italian restaurant"

# Create a timestamped folder for this pair inside /images
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
folder_path = os.path.join("images", f"pair_{timestamp}")
os.makedirs(folder_path, exist_ok=True)

# Generate two images from the same prompt
image_paths = []
for i in range(2):
    img = pipe(prompt).images[0]
    img_path = os.path.abspath(os.path.join(folder_path, f"image_{i}.png"))
    img.save(img_path)
    image_paths.append(str(img_path))

print("Images generated and saved in:", folder_path)


from rapidata import RapidataClient

rapi = RapidataClient()

order = rapi.order.create_compare_order(
    name="Example Alignment Order",
    instruction="Which logo do you prefer?",
    contexts=[prompt],
    datapoints=[image_paths],
).run()

order.display_progress_bar()

results = order.get_results()
print(results)


results_path = os.path.join(folder_path, "data.json")

import json
with open('data.json', 'w') as f:
    json.dump(results, f)