import os
import json
import csv
from datetime import datetime
from diffusers import StableDiffusionXLPipeline
import torch
from PIL import Image
# Keep rapidata import if needed elsewhere, but not directly used here
# from rapidata import RapidataClient




import os
import json
import csv
import time # Added for timing batches
from typing import Optional # Added for refined type hint

from diffusers import StableDiffusionXLPipeline, AutoencoderKL # Added VAE import
import torch
from PIL import Image
# Keep rapidata import if needed elsewhere, but not directly used here
# from rapidata import RapidataClient

def generate_images_for_iteration(
    model_path_or_name: str,
    lora_path: str = None,
    prompts: list = None,
    iteration_num: int = 0,
    base_image_dir: str = "images",
    base_metadata_dir: str = "metadata",
    batch_size: int = 32,                   # Added: Batch size for generation
    num_inference_steps: int = 30,       # Added: Inference steps parameter
    guidance_scale: float = 7.5,         # Added: Guidance scale parameter
    num_images_per_prompt: int = 2,      # Added: Number of images per prompt
    vae_path: Optional[str] = "madebyollin/sdxl-vae-fp16-fix", # Added: Optional separate VAE path
    device: str = "cuda" if torch.cuda.is_available() else "cpu", # Added: Device selection
    dtype: torch.dtype = torch.float16 if torch.cuda.is_available() else torch.float32 # Added: Dtype selection
) -> Optional[str]: # Refined return type hint
    """
    Generates images for a list of prompts using a specified base model,
    optionally applying LoRA weights, saves them in iteration-specific folders,
    and creates metadata. Uses batched generation for efficiency.

    Args:
        model_path_or_name: HF model name/path of the BASE diffusion model.
        lora_path: Optional path to the directory containing LoRA weights
                   (e.g., 'models_lora/sdxl_dpo_lora_iter_0').
        prompts: A list of strings (prompts).
        iteration_num: The current training loop iteration number.
        base_image_dir: Base directory to store images.
        base_metadata_dir: Base directory to store metadata.
        batch_size: Number of prompts to process simultaneously in one batch.
        num_inference_steps: Number of denoising steps.
        guidance_scale: Classifier-free guidance scale.
        num_images_per_prompt: How many images to generate for each prompt.
        vae_path: Optional path/name for a separate VAE (e.g., "madebyollin/sdxl-vae-fp16-fix").
                  Set to None to use the VAE included with the base model.
        device: The device to run generation on ('cuda' or 'cpu').
        dtype: The torch dtype to use (torch.float16 or torch.float32).

    Returns:
        Path to the generated image mapping JSON file for this iteration.
        Returns None if generation fails or no prompts provided.
    """
    if not prompts:
        print("❌ No prompts provided to generate_images_for_iteration.")
        return None

    print(f"\n--- Generating Images: Iteration {iteration_num} ---")
    print(f"Using base model: {model_path_or_name}")
    if lora_path:
        print(f"Attempting to apply LoRA weights from: {lora_path}")
    else:
        print("No LoRA weights specified, using base model only.")
    print(f"Using device: {device}, dtype: {dtype}, batch size: {batch_size}")
    if vae_path:
        print(f"Using separate VAE: {vae_path}")

    start_time_total = time.time()

    # Create iteration-specific directories
    iter_image_dir = os.path.join(base_image_dir, f"iter_{iteration_num}")
    iter_metadata_dir = os.path.join(base_metadata_dir, f"iter_{iteration_num}")
    os.makedirs(iter_image_dir, exist_ok=True)
    os.makedirs(iter_metadata_dir, exist_ok=True)

    mapping_file = os.path.join(iter_metadata_dir, "image_mapping.json")
    prompts_csv = os.path.join(iter_metadata_dir, "prompts.csv")


    
    pipe = None
    vae = None
    try:
        # --- Load VAE (Optional but recommended for consistency/performance) ---
        if vae_path:
            try:
                print(f"Loading VAE: {vae_path}")
                vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=dtype)
                print("VAE loaded.")
            except Exception as e_vae:
                print(f"⚠️ Warning: Could not load VAE from {vae_path}. Using default VAE. Error: {e_vae}")
                vae = None # Fallback to default

        # --- Load Base Pipeline ---
        print(f"Loading base model pipeline: {model_path_or_name}")
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_path_or_name,
            vae=vae, # Pass loaded VAE or None
            torch_dtype=dtype,
            use_safetensors=True,
            variant="fp16" if dtype == torch.float16 else None
        )
        pipe.to(device)
        try: # Optional optimizations
            pipe.enable_xformers_memory_efficient_attention()
            print("xFormers enabled.")
        except Exception:
            print("xFormers not available or failed to enable.")
        if vae: # Only enable slicing if we loaded a separate VAE we control
             try:
                 pipe.vae.enable_slicing()
                 print("VAE slicing enabled.")
             except Exception:
                 print("VAE slicing not available or failed to enable.")
        print("Base pipeline loaded.")

        # --- Apply LoRA weights if provided ---
        if lora_path and os.path.isdir(lora_path):
            lora_weights_file = os.path.join(lora_path, "pytorch_lora_weights.safetensors")
            # Add check for adapter_model.safetensors as well if needed:
            # lora_adapter_file = os.path.join(lora_path, "adapter_model.safetensors")

            weight_file_to_load = None
            if os.path.exists(lora_weights_file):
                 weight_file_to_load = lora_weights_file
            # elif os.path.exists(lora_adapter_file): # Example for alternative name
            #      weight_file_to_load = lora_adapter_file

            if weight_file_to_load:
                 print(f"  Found LoRA weights file: {os.path.basename(weight_file_to_load)}")
                 print(f"  Applying LoRA weights to the pipeline...")
                 # load_lora_weights expects the *directory* or specific file path
                 pipe.load_lora_weights(lora_path, weight_name=os.path.basename(weight_file_to_load))
                 # Optional: Fuse LoRA for potentially faster inference (uses more memory)
                 # pipe.fuse_lora()
                 # print("  Fused LoRA weights.")
                 print("  LoRA weights applied successfully.")
            else:
                 print(f"  ⚠️ Warning: LoRA directory '{lora_path}' exists, but no compatible weights file "
                       f"(e.g., 'pytorch_lora_weights.safetensors') found inside. Using base model only.")
        elif lora_path == "artificialguybr/LogoRedmond-LogoLoraForSDXL-V2":
            pipe.load_lora_weights("artificialguybr/LogoRedmond-LogoLoraForSDXL-V2")

        elif lora_path:
            print(f"  ⚠️ Warning: Specified LoRA path '{lora_path}' is not a valid directory. Using base model only.")
        # --- End Apply LoRA ---

        image_mapping = {"data": []}
        global_image_counter = 0
        num_prompts = len(prompts)

        with open(prompts_csv, "w", newline="", encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Prompt"])  # CSV header

            print(f"\nStarting batched image generation ({num_prompts} prompts)...")
            # --- Batch Loop ---
            for i in range(0, num_prompts, batch_size):
                batch_start_time = time.time()
                # Determine actual batch size (handles the last partial batch)
                current_batch_prompts = prompts[i : min(i + batch_size, num_prompts)]
                current_batch_size = len(current_batch_prompts)

                if not current_batch_prompts: continue # Skip if batch is empty for some reason

                print(f"\n--- Processing Batch {i//batch_size + 1}/{ (num_prompts + batch_size - 1) // batch_size } "
                      f"(Prompts {i+1}-{i+current_batch_size}) ---")
                # print(f"Prompts in batch: {current_batch_prompts}") # Optional: print all prompts

                try:
                    # --- Generate Images for the Batch ---
                    # The pipeline call handles the list of prompts.
                    # It returns a flat list of images: [prompt1_img1, prompt1_img2, ..., promptN_img1, promptN_img2]
                    generated_images = pipe(
                        prompt=current_batch_prompts,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        num_images_per_prompt=num_images_per_prompt,
                    ).images

                    if len(generated_images) != current_batch_size * num_images_per_prompt:
                        print(f"  ⚠️ Warning: Expected {current_batch_size * num_images_per_prompt} images, "
                              f"but received {len(generated_images)}. Skipping batch.")
                        # Increment counter cautiously, assuming failure for the whole batch
                        global_image_counter += current_batch_size * num_images_per_prompt
                        if device == 'cuda': torch.cuda.empty_cache()
                        continue

                    # --- Process and Save Images from the Batch ---
                    image_index_in_batch = 0 # Index into the flat generated_images list
                    for j, prompt in enumerate(current_batch_prompts):
                        prompt_image_filenames_relative = [] # Store relative paths for JSON

                        # Loop to save the `num_images_per_prompt` images for this prompt
                        for k in range(num_images_per_prompt):
                            # Construct a unique filename
                            img_filename_base = f"iter{iteration_num}_promptidx{i+j}_img{k}_{global_image_counter:04d}.png"
                            img_path_full = os.path.join(iter_image_dir, img_filename_base)

                            try:
                                img = generated_images[image_index_in_batch]
                                img.save(img_path_full)

                                # Store filename relative to base_image_dir for JSON
                                relative_filename = os.path.join(f"iter_{iteration_num}", img_filename_base)
                                prompt_image_filenames_relative.append(relative_filename)

                            except Exception as e_save:
                                print(f"    ❌ Error saving image {image_index_in_batch} for prompt '{prompt[:50]}...': {e_save}")
                                # Decide if we should skip this image or the whole prompt
                                # For now, we just won't add its filename to the list

                            image_index_in_batch += 1
                            global_image_counter += 1
                            if 'img' in locals(): del img # Delete PIL image obj promptly

                        # Store mapping info for this prompt (only includes successfully saved images)
                        if prompt_image_filenames_relative:
                            image_mapping["data"].append({
                                "prompt": prompt,
                                "images": prompt_image_filenames_relative
                            })
                            # Write prompt to CSV only if images were successfully generated/saved
                            writer.writerow([prompt])
                            print(f"    ✅ Saved {len(prompt_image_filenames_relative)} image(s) for prompt {i+j+1}: ...{prompt_image_filenames_relative[-1][-25:]}")
                        else:
                             print(f"    ⚠️ No images successfully saved for prompt {i+j+1}: '{prompt[:50]}...'")


                    # Clear generated images list for the batch and cache
                    del generated_images
                    if device == 'cuda':
                        torch.cuda.empty_cache()

                except Exception as e_batch:
                    print(f"    ❌ Critical Error processing batch starting with prompt {i+1}: {e_batch}")
                    # Increment counter to avoid filename collisions if we continue, assuming all images in batch failed
                    global_image_counter += current_batch_size * num_images_per_prompt
                    if device == 'cuda':
                        torch.cuda.empty_cache() # Clear cache on error too
                    # Consider whether to `continue` to next batch or `return None` on critical failure
                    continue # Skip adding this batch to the mapping

                batch_end_time = time.time()
                print(f"--- Batch finished in {batch_end_time - batch_start_time:.2f} seconds ---")

        # --- Save mapping JSON ---
        with open(mapping_file, "w", encoding='utf-8') as f:
            json.dump(image_mapping, f, indent=4, ensure_ascii=False)

        total_time = time.time() - start_time_total
        print(f"\n🟢 Image generation complete for iteration {iteration_num}.")
        print(f"🟢 Saved {len(image_mapping['data'])} prompt entries to {mapping_file}")
        print(f"🟢 Prompts reference saved at {prompts_csv}")
        print(f"🕒 Total generation time: {total_time:.2f} seconds.")
        if num_prompts > 0:
            print(f"🕒 Average time per prompt: {total_time / num_prompts:.2f} seconds")
        return mapping_file

    except Exception as e_outer:
         print(f"❌ An critical error occurred during pipeline setup or overall generation for iteration {iteration_num}: {e_outer}")
         # Attempt cleanup even if setup failed partially
         if device == 'cuda': torch.cuda.empty_cache()
         return None # Indicate failure
    finally:
        # --- Cleanup ---
        print("Cleaning up resources...")
        del pipe
        del vae # Delete VAE object if it was loaded
        if device == 'cuda':
            torch.cuda.empty_cache()
            print("Cleared GPU cache.")

# --- Example Usage (similar to the original __main__ block) ---
# You would call this function from your main training loop.
# Here's how a standalone test might look:

if __name__ == "__main__":
    # Example prompts (replace with your actual prompts list or loading mechanism)
    TEST_PROMPTS = [
        "A photorealistic astronaut riding a horse on the moon",
        "A vibrant oil painting of a futuristic city skyline at sunset",
        "Steampunk cat operating a complex machine, detailed illustration",
        "An abstract sculpture made of swirling light trails in a dark void",
        "A tranquil Japanese garden with a koi pond and cherry blossoms, watercolor style",
        "Macro photograph of a dewdrop on a spider web, sharp focus",
        "A medieval knight fighting a dragon made of clouds, epic fantasy art",
        "A cozy bookstore cafe on a rainy night, warm lighting, digital painting",
    ]

    # --- Test Case 1: Base Model Only ---
    print("\n" + "="*20 + " TEST: Base Model Only " + "="*20)
    try:
        generate_images_for_iteration(
            model_path_or_name="stabilityai/stable-diffusion-xl-base-1.0",
            lora_path=None, # Explicitly None
            prompts=TEST_PROMPTS,
            iteration_num=0, # Test as iteration 0
            base_image_dir="images_batch_test",
            base_metadata_dir="metadata_batch_test",
            batch_size=4, # Adjust batch size based on your GPU VRAM
            num_images_per_prompt=2,
            num_inference_steps=25 # Slightly fewer steps for faster testing
        )
    except Exception as e:
        print(f"An error occurred during base model test: {e}")

    # --- Test Case 2: With LoRA (if available) ---
    # Replace with the actual path to a LoRA trained for SDXL 1.0 base
    test_lora_dir = "models_lora/sdxl_dpo_lora_iter_0" # Example path

    if os.path.isdir(test_lora_dir):
        print("\n" + "="*20 + f" TEST: With LoRA ({test_lora_dir}) " + "="*20)
        try:
            generate_images_for_iteration(
                model_path_or_name="stabilityai/stable-diffusion-xl-base-1.0",
                lora_path=test_lora_dir,
                prompts=TEST_PROMPTS[:4], # Use fewer prompts for LoRA test
                iteration_num=999, # Use a distinct iteration number for test output
                base_image_dir="images_batch_test", # Can reuse or use different dirs
                base_metadata_dir="metadata_batch_test",
                batch_size=2, # May need smaller batch size with LoRA
                num_images_per_prompt=1, # Generate only 1 image per prompt for LoRA test
                num_inference_steps=25
            )
        except Exception as e:
            print(f"An error occurred during LoRA test: {e}")
    else:
        print(f"\nSkipping LoRA test: Directory '{test_lora_dir}' not found.")
# --- Rest of the file (e.g., standalone test block) remains the same ---

# commented out for now
# if __name__ == "__main__":
#     try:
#         from prompts import PROMPT_LIST
#         # Generate using the base model for testing
#         generate_images_for_iteration(
#             model_path_or_name="stabilityai/stable-diffusion-xl-base-1.0",
#             lora_path=None, # Explicitly None for base model test
#             prompts=PROMPT_LIST,
#             iteration_num=0, # Standalone test as iteration 0
#             base_image_dir="images",
#             base_metadata_dir="metadata"
#         )
#         # Example of testing with a LoRA path (if one exists from a previous run)
#         # test_lora = "models_lora/sdxl_dpo_lora_iter_0"
#         # if os.path.isdir(test_lora):
#         #      print("\n--- Testing with LoRA ---")
#         #      generate_images_for_iteration(
#         #         model_path_or_name="stabilityai/stable-diffusion-xl-base-1.0",
#         #         lora_path=test_lora,
#         #         prompts=["A futuristic cityscape with flying cars, LoRA style test"],
#         #         iteration_num=999, # Use a distinct iteration number for test output
#         #         base_image_dir="images_test_lora",
#         #         base_metadata_dir="metadata_test_lora"
#         #      )

#     except ImportError:
#         print("Could not import PROMPT_LIST from prompts.py. Please create prompts.py.")
#     except Exception as e:
#         print(f"An error occurred during standalone test: {e}")
