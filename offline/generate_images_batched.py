import os
import json
import csv
import time # Import time for basic benchmarking
from datetime import datetime
from diffusers import StableDiffusionXLPipeline, AutoencoderKL
import torch
from PIL import Image
# from rapidata import RapidataClient # Removed as it wasn't used in the provided snippet

# Define storage directories
IMAGE_DIR = "images"
METADATA_DIR = "metadata"
# ORDER_RESULTS_DIR = "order_results" # Removed as it wasn't used

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(METADATA_DIR, exist_ok=True)

# Path for metadata files
MAPPING_FILE = os.path.join(METADATA_DIR, "image_mapping.json")
PROMPTS_CSV = os.path.join(METADATA_DIR, "prompts.csv")
PROMPTS_TXT = os.path.join("prompts.txt") # Assuming prompts.txt is in the root directory

# --- Configuration ---
# Adjust batch_size based on your GPU memory (VRAM). Start small (e.g., 4 or 8) and increase if possible.
# Higher batch size generally means faster processing, up to the memory limit.
BATCH_SIZE = 8 # <--- TUNE THIS VALUE
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32 # Use float32 for CPU

# Initialize SDXL pipeline
print("Loading SDXL Pipeline...")

vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix",
    torch_dtype=DTYPE # <<< Use the determined dtype (should be float16 on CUDA)
)
 
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=DTYPE,
    use_safetensors=True,
    variant="fp16" if DTYPE == torch.float16 else None,# Only use fp16 variant if dtype is float16
    vae=vae
)

try:
    pipe.vae.enable_slicing()
    print("VAE Slicing enabled.")
except Exception as e:
    print(f"Could not enable VAE slicing: {e} (Might not be applicable to this pipeline/version)")
    
pipe.load_lora_weights("artificialguybr/LogoRedmond-LogoLoraForSDXL-V2")
    
pipe.enable_xformers_memory_efficient_attention()

# Optimization: Use torch.compile if available (PyTorch 2.0+) for potential speedup
# try:
#     pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
#     print("Applied torch.compile to unet.")
# except Exception as e:
#     print(f"torch.compile not available or failed: {e}")

pipe.to(DEVICE)
print(f"Pipeline loaded on {DEVICE} with dtype {DTYPE}.")


def generate_images_batched(batch_size=4):
    """Generates two images per prompt using batching, stores them, and creates metadata."""
    start_time = time.time()

    print(f"Reading prompts from {PROMPTS_TXT}...")
    with open(PROMPTS_TXT, "r") as f:
        prompts = [line.strip() for line in f.readlines() if line.strip()] # Skip empty lines
    print(f"Found {len(prompts)} prompts.")
    if not prompts:
        print("No prompts found. Exiting.")
        return

    image_mapping = {"data": []}
    global_image_counter = 1 # Start image numbering from 1

    print(f"Starting image generation with batch size {batch_size}...")
    with open(PROMPTS_CSV, "w", newline="", encoding='utf-8') as csvfile: # Use utf-8 encoding
        writer = csv.writer(csvfile)
        writer.writerow(["Prompt"])  # CSV header

        # Process prompts in batches
        for i in range(0, len(prompts), batch_size):
            batch_start_time = time.time()
            batch_prompts = prompts[i : i + batch_size]
            print(batch_prompts[0])
            num_prompts_in_batch = len(batch_prompts)
            
            print(f"\n--- Processing batch {i//batch_size + 1}/{ (len(prompts) + batch_size - 1)//batch_size } (Prompts {i+1}-{i+num_prompts_in_batch}) ---")
            print(f"Prompts in batch: {batch_prompts}")

            try:
                # Generate images: 2 images for each prompt in the batch
                # The result 'images' will be a flat list: [prompt1_img1, prompt1_img2, prompt2_img1, prompt2_img2, ...]
                generated_images = pipe(
                    prompt=batch_prompts,
                    num_images_per_prompt=2,
                    # Add other parameters like guidance_scale, num_inference_steps if needed
                    # guidance_scale=7.5,
                    num_inference_steps=30,
                ).images
                
                if DEVICE == 'cuda':
                    print("  Clearing CUDA cache immediately after pipe()...")
                    torch.cuda.empty_cache()
                    print("  CUDA cache cleared.")
                # ---
                
                print(f"Generated {len(generated_images)} images for {num_prompts_in_batch} prompts.")

                # Process and save images for the current batch
                for j, prompt in enumerate(batch_prompts):
                    # Get the two images for the current prompt
                    # Index calculation: 2 images per prompt, j is the index within the batch
                    img1_index = j * 2
                    img2_index = j * 2 + 1

                    if img1_index >= len(generated_images) or img2_index >= len(generated_images):
                         print(f"Warning: Expected 2 images for prompt '{prompt}' but got fewer. Skipping.")
                         continue # Or handle error appropriately

                    img1 = generated_images[img1_index]
                    img2 = generated_images[img2_index]

                    # Define image filenames using the global counter
                    image_id1 = f"image_{global_image_counter:04d}.png"
                    image_id2 = f"image_{global_image_counter + 1:04d}.png"
                    image_path1 = os.path.join(IMAGE_DIR, image_id1)
                    image_path2 = os.path.join(IMAGE_DIR, image_id2)

                    # Save images
                    img1.save(image_path1)
                    img2.save(image_path2)
                    
                    # Store mapping info
                    image_mapping["data"].append({
                        "prompt": prompt,
                        "images": [image_id1, image_id2]
                    })

                    # Write prompt to CSV
                    writer.writerow([prompt])

                    print(f"  ✅ Saved: {image_id1}, {image_id2} for prompt: \"{prompt[:50]}...\"")

                    # Increment global counter for the next pair
                    global_image_counter += 2
                    
                    # ----> Clear Cache Here <----
                    # After processing the batch and *before* the next pipe() call
                    del img1 # Delete image objects
                    del img2
                    
                if DEVICE == 'cuda':
                    print("  Clearing CUDA cache...")
                    torch.cuda.empty_cache()
                    print("  CUDA cache cleared.")
                        
                del generated_images # Explicitly delete large tensor list

            except Exception as e:
                 print(f"!!!!!!!!!!!!!!!!! ERROR processing batch starting with prompt: '{batch_prompts[0]}' !!!!!!!!!!!!!!!!!")
                 print(f"Error details: {e}")
                 print("Skipping this batch and continuing...")
                 # Optionally: add prompts from failed batch to a retry list
                 # Increment counter anyway to avoid filename collisions if you decide to skip
                 global_image_counter += len(batch_prompts) * 2
                 
                 if DEVICE == 'cuda':
                     print("  Clearing CUDA cache after error...")
                     torch.cuda.empty_cache()
                     print("  CUDA cache cleared after error.")
          

            batch_end_time = time.time()
            print(f"--- Batch finished in {batch_end_time - batch_start_time:.2f} seconds ---")

            # Optional: Clear CUDA cache between batches if memory is extremely tight
            # if DEVICE == 'cuda':
            #     torch.cuda.empty_cache()

    # --- Save Metadata ---
    print("\nSaving final metadata...")
    try:
        with open(MAPPING_FILE, "w", encoding='utf-8') as f: # Use utf-8 encoding
            json.dump(image_mapping, f, indent=4, ensure_ascii=False) # ensure_ascii=False for non-latin chars
        print(f"🟢 Image mapping saved at {MAPPING_FILE}")
    except Exception as e:
        print(f"Error saving JSON mapping: {e}")

    print(f"🟢 Prompts CSV saved at {PROMPTS_CSV}")

    end_time = time.time()
    total_time = end_time - start_time
    print(f"\n🏁 Finished generating {global_image_counter - 1} images for {len(prompts)} prompts in {total_time:.2f} seconds.")
    if len(prompts) > 0:
        print(f"Average time per prompt (pair): {total_time / len(prompts):.2f} seconds")


if __name__ == "__main__":
    # Step 1: Generate images & save metadata (Always run the generation function in this version)
    # The check for existing files is removed, assuming you want to regenerate if script is run.
    # If you want to keep the check, uncomment the lines below and indent the call.
    # if not os.path.exists(MAPPING_FILE) or not os.path.exists(PROMPTS_CSV):
    #     print("Metadata files not found. Generating images...")
    #     generate_images_batched(batch_size=BATCH_SIZE)
    # else:
    #     print("Metadata files already exist. Skipping image generation.")
    #     print(f"Mapping file: {MAPPING_FILE}")
    #     print(f"Prompts CSV: {PROMPTS_CSV}")
    #     print(f"Images should be in: {IMAGE_DIR}")

    generate_images_batched(batch_size=BATCH_SIZE)