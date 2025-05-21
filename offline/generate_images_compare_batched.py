import os
import json
import csv
import time
from datetime import datetime
from diffusers import StableDiffusionXLPipeline, AutoencoderKL
import torch
from PIL import Image

# -------------------------------
#      Directory Setup
# -------------------------------
IMAGE_DIR = "images_compare"
METADATA_DIR = "metadata_compare"
LORA_DIR = "LORA"

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(METADATA_DIR, exist_ok=True)

MAPPING_FILE = os.path.join(METADATA_DIR, "image_mapping.json")
PROMPTS_CSV = os.path.join(METADATA_DIR, "prompts.csv")
PROMPTS_TXT = os.path.join("prompts_compare.txt")
LORA_WEIGHTS = os.path.join(LORA_DIR, "diffusion-sdxl-dpo/checkpoint-10000")

# -------------------------------
#        Configuration
# -------------------------------
BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# -------------------------------
#         Load VAE
# -------------------------------
print("Loading shared VAE...")
vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix",
    torch_dtype=DTYPE
)

# -------------------------------
#     Load Base SDXL Pipeline
# -------------------------------
print("Loading base SDXL pipeline...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=DTYPE,
    use_safetensors=True,
    variant="fp16" if DTYPE == torch.float16 else None,
    vae=vae
)

pipe.to(DEVICE)
pipe.enable_xformers_memory_efficient_attention()
try:
    pipe.vae.enable_slicing()
except Exception as e:
    print(f"Could not enable VAE slicing on base model: {e}")

print("✅ Base model loaded.")

# -------------------------------
#     Load LoRA-Augmented Pipeline
# -------------------------------
print("Loading LoRA-augmented pipeline...")

pipe_lora = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",  # Same base
    torch_dtype=DTYPE,
    use_safetensors=True,
    variant="fp16" if DTYPE == torch.float16 else None,
    vae=vae
)

# ✳️ Apply LoRA weights here
# Example if using built-in `load_lora_weights` from diffusers:
# pipe_lora.load_lora_weights("path/to/lora_weights", adapter_name="lora")

# If you want to merge the LoRA into UNet permanently:
# pipe_lora.fuse_lora()
pipe_lora.load_lora_weights(pretrained_model_name_or_path_or_dict=LORA_WEIGHTS, weight_name="pytorch_lora_weights.safetensors")
pipe_lora.to(DEVICE)
pipe_lora.enable_xformers_memory_efficient_attention()
try:
    pipe_lora.vae.enable_slicing()
except Exception as e:
    print(f"Could not enable VAE slicing on LoRA model: {e}")

print("✅ LoRA model loaded.")


def generate_images_batched(batch_size=4):
    start_time = time.time()

    print(f"Reading prompts from {PROMPTS_TXT}...")
    with open(PROMPTS_TXT, "r") as f:
        prompts = [line.strip() for line in f.readlines() if line.strip()]
    print(f"Found {len(prompts)} prompts.")
    if not prompts:
        print("No prompts found. Exiting.")
        return

    image_mapping = {"data": []}
    global_image_counter = 1

    print(f"Starting image generation with batch size {batch_size}...")
    with open(PROMPTS_CSV, "w", newline="", encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Prompt"])  

        for i in range(0, len(prompts), batch_size):
            batch_start_time = time.time()
            batch_prompts = prompts[i : i + batch_size]

            print(f"\n--- Batch {i//batch_size + 1} ---")
            print(f"Prompts: {batch_prompts}")

            try:
                # Generate with base model
                images_base = pipe(
                    prompt=batch_prompts,
                    num_images_per_prompt=1,
                    num_inference_steps=30,
                ).images

                # Generate with LoRA model
                images_lora = pipe_lora(
                    prompt=batch_prompts,
                    num_images_per_prompt=1,
                    num_inference_steps=30,
                ).images

                if DEVICE == 'cuda':
                    torch.cuda.empty_cache()

                for j, prompt in enumerate(batch_prompts):
                    img_base = images_base[j]
                    img_lora = images_lora[j]

                    image_id1 = f"image_{global_image_counter:04d}.png"
                    image_id2 = f"image_{global_image_counter + 1:04d}.png"
                    image_path1 = os.path.join(IMAGE_DIR, image_id1)
                    image_path2 = os.path.join(IMAGE_DIR, image_id2)

                    img_base.save(image_path1)
                    img_lora.save(image_path2)

                    image_mapping["data"].append({
                        "prompt": prompt,
                        "images": [image_id1, image_id2]
                    })

                    writer.writerow([prompt])
                    print(f"  ✅ Saved: {image_id1} (base), {image_id2} (lora)")

                    global_image_counter += 2
                    del img_base, img_lora

                del images_base, images_lora
                if DEVICE == 'cuda':
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error in batch: {e}")
                global_image_counter += len(batch_prompts) * 2
                if DEVICE == 'cuda':
                    torch.cuda.empty_cache()

            batch_end_time = time.time()
            print(f"Batch finished in {batch_end_time - batch_start_time:.2f} seconds")

    print("\nSaving final metadata...")
    with open(MAPPING_FILE, "w", encoding='utf-8') as f:
        json.dump(image_mapping, f, indent=4, ensure_ascii=False)

    print(f"🟢 Metadata saved to {MAPPING_FILE}")
    print(f"🟢 Prompts CSV saved to {PROMPTS_CSV}")

    end_time = time.time()
    total_time = end_time - start_time
    print(f"\n🏁 Finished in {total_time:.2f} seconds.")
    if len(prompts) > 0:
        print(f"Average time per prompt: {total_time / len(prompts):.2f} seconds")
        
if __name__ == "__main__":
    generate_images_batched(batch_size=BATCH_SIZE)