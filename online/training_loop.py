import os
import torch
from rapidata import RapidataClient

# Import functions from our refactored modules
from prompts import PROMPT_LIST
from generate_images import generate_images_for_iteration
from run_orders import create_rapidata_order_for_iteration, save_order_results_for_iteration
from transform_data import transform_and_upload_for_iteration
from types import SimpleNamespace
from LORA.train_diffusion_dpo_sdxl import main
import shutil

# --- Configuration ---
NUM_ITERATIONS = 5  # Total number of feedback loops to run
BASE_MODEL_NAME = "stabilityai/stable-diffusion-xl-base-1.0" # Initial model
FINETUNED_MODEL_DIR = "models" # Directory to save fine-tuned models
BASE_IMAGE_DIR = "images"
BASE_METADATA_DIR = "metadata"
BASE_ORDER_RESULTS_DIR = "order_results"
# IMPORTANT: Set your HF Token and Repo Name
# Load from environment variable if possible
HF_TOKEN = os.getenv("HF_TOKEN")
HF_REPO_BASE_NAME = "username/onlinetest" # CHANGE THIS default or set env var
RAPIDATA_VALIDATION_SET_ID = "67f002c7075c672e0bfe36e6" # Optional: Name of your validation set on Rapidata


# Create base directories if they don't exist
os.makedirs(FINETUNED_MODEL_DIR, exist_ok=True)
os.makedirs(BASE_IMAGE_DIR, exist_ok=True)
os.makedirs(BASE_METADATA_DIR, exist_ok=True)
os.makedirs(BASE_ORDER_RESULTS_DIR, exist_ok=True)


# --- Placeholder Fine-tuning Function ---
def fine_tune_model(base_model_path: str, feedback_dataset_repo_id: str, iteration_num: int, output_dir_base: str, lora_model_path: str,) -> str:
    """
    Placeholder function for fine-tuning the model.
    Replace this with your actual fine-tuning script call (e.g., using diffusers trainer, trl, etc.).

    Args:
        base_model_path: Path to the model to be fine-tuned.
        feedback_dataset_repo_id: Hugging Face repo ID of the preference dataset.
        iteration_num: Current loop iteration number.
        output_dir_base: Base directory to save the new model.

    Returns:
        Path to the newly saved fine-tuned model directory.
    """
    print(f"\n--- Fine-tuning Model: Iteration {iteration_num} ---")
    print(f"Using base model: {base_model_path}")
    print(f"Using feedback dataset: {feedback_dataset_repo_id}")

    # Define where the new model will be saved
    new_model_output_path = os.path.join(output_dir_base, f"sdxl_LORA_finetuned_iter_{iteration_num}")
    os.makedirs(new_model_output_path, exist_ok=True)

    print(f"Simulating fine-tuning process...")
    args = SimpleNamespace(
    # --- Values explicitly assigned in the original namespace snippet ---
    pretrained_model_name_or_path=base_model_path,  # Required, but was set
    pretrained_vae_model_name_or_path="madebyollin/sdxl-vae-fp16-fix", # Default was None
    output_dir=new_model_output_path, # Default was "diffusion-dpo-lora"
    mixed_precision="fp16", # Default was None
    dataset_name=feedback_dataset_repo_id, # Default was None, but check raises ValueError if None
    cache_dir="/home/jupyter/marvin/DS_Lab/DiffusionDPO/datasets/rapidata", # Default was None
    train_batch_size=16, # Default was 4
    dataloader_num_workers=2, # Default was 0
    gradient_accumulation_steps=1, # Default was 1
    gradient_checkpointing=True, # Default was False
    use_8bit_adam=True, # Default was False
    rank=8, # Default was 4
    learning_rate=1e-5, # Default was 5e-4
    lr_scheduler="constant", # Default was "constant" (no change here, but listed for completeness)
    lr_warmup_steps=0, # Default was 500
    max_train_steps=1000, # Default was None
    checkpointing_steps=400, # Default was 500
    run_validation=True, # Default was False
    validation_steps=400, # Default was 200
    seed=0, # Default was None
    report_to="wandb", # Default was "tensorboard"
    dataset_split_name="train", # Default was "validation"
    reference_lora_model_path = lora_model_path, # path to lora weights, if None, just use base sdxl
    # --- Arguments using default values from the parser ---
    revision=None,
    variant=None,
    max_train_samples=None,
    resolution=1024,
    vae_encode_batch_size=8,
    no_hflip=False, # action="store_true" defaults to False
    random_crop=False, # action="store_true" defaults to False
    num_train_epochs=10,
    checkpoints_total_limit=None,
    resume_from_checkpoint=None,
    beta_dpo=5000,
    scale_lr=False, # action="store_true" defaults to False
    lr_num_cycles=1,
    lr_power=1.0,
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_weight_decay=1e-2,
    adam_epsilon=1e-08,
    max_grad_norm=1.0,
    push_to_hub=False, # action="store_true" defaults to False
    hub_token=None,
    hub_model_id=None,
    logging_dir="logs",
    allow_tf32=False, # action="store_true" defaults to False
    prior_generation_precision=None,
    local_rank=-1, # Note: original code might update this from env var, SimpleNamespace won't do that automatically
    enable_xformers_memory_efficient_attention=True, # action="store_true" defaults to False
    is_turbo=False, # action="store_true" defaults to False
    tracker_name="diffusion-dpo-lora-sdxl", # Default was "diffusion-dpo-lora-sdxl"
    iter_num=iteration_num
)
    main(args)

    print(f"Fine-tuning complete. 'Model' saved to: {new_model_output_path}")


    # Clean up GPU memory after potential training
    torch.cuda.empty_cache()

    return new_model_output_path


# --- Main Training Loop ---
def run_training_loop():
    """Executes the full RLHF-style training loop."""

    if not HF_TOKEN:
        print("❌ Error: HUGGINGFACE_TOKEN environment variable not set.")
        print("Please set your Hugging Face write token.")
        return
    if HF_REPO_BASE_NAME == "YourUsername/SDXL-RLHF-Loop":
        print("⚠️ Warning: HF_REPO_BASE_NAME is set to the default.")
        print("Please set it to your desired Hugging Face repository base name (e.g., 'MyUser/MyProject').")
        # Decide if you want to stop or continue with the default name
        # return

    print("🚀 Starting Training Loop...")
    print(f"Number of iterations: {NUM_ITERATIONS}")
    print(f"Initial model: {BASE_MODEL_NAME}")
    print(f"Feedback dataset base repo: {HF_REPO_BASE_NAME}")
    print(f"Fine-tuned models will be saved in: {FINETUNED_MODEL_DIR}")

    # Initialize Rapidata client (ensure API key is configured)
    try:
        rapi = RapidataClient()
        print("🟢 Rapidata client initialized.")
    except Exception as e:
        print(f"❌ Failed to initialize Rapidata client: {e}")
        print("Ensure your Rapidata API key is configured correctly (e.g., RAPIDATA_API_KEY env var).")
        return

    lora_model_path = None  # Start with the base model

    for i in range(0, NUM_ITERATIONS):
        print(f"\n================ Iteration {i+1}/{NUM_ITERATIONS} ================")
        iteration_start_time = datetime.now()

        try:
            # 1. Generate Images
            # Pass the *path* to the current model (local path after first iteration)
            image_mapping_file = generate_images_for_iteration(
                model_path_or_name=BASE_MODEL_NAME,
                prompts=PROMPT_LIST,
                iteration_num=i,
                base_image_dir=BASE_IMAGE_DIR,
                base_metadata_dir=BASE_METADATA_DIR,
                lora_path=lora_model_path
            )
            torch.cuda.empty_cache()
            # 2. Create Rapidata Order
            rapidata_order = create_rapidata_order_for_iteration(
                rapi=rapi,
                mapping_file_path=image_mapping_file,
                iteration_num=i,
                base_image_dir=BASE_IMAGE_DIR, # Pass base dir, paths in mapping are relative to this
                validation_set_id=RAPIDATA_VALIDATION_SET_ID # Pass optional validation set name
            )
            torch.cuda.empty_cache()
            if not rapidata_order:
                print(f"❌ Failed to create Rapidata order for iteration {i}. Stopping loop.")
                break

            # 3. Wait for Order and Save Results
            order_results_file = save_order_results_for_iteration(
                order=rapidata_order,
                iteration_num=i,
                base_order_results_dir=BASE_ORDER_RESULTS_DIR
            )
            if not order_results_file:
                print(f"❌ Failed to get/save Rapidata results for iteration {i}. Stopping loop.")
                break
            torch.cuda.empty_cache()
            # 4. Transform Data and Upload to Hugging Face Hub
            feedback_repo_id = transform_and_upload_for_iteration(
                rapidata_results_path=order_results_file,
                image_mapping_path=image_mapping_file,
                iteration_num=i,
                hf_repo_base_name=HF_REPO_BASE_NAME,
                hf_token=HF_TOKEN,
                base_image_dir=BASE_IMAGE_DIR
            )
            if os.path.isdir(BASE_IMAGE_DIR):
                shutil.rmtree(BASE_IMAGE_DIR)

                      
            if not feedback_repo_id:
                 print(f"❌ Failed to transform or upload data for iteration {i}. Stopping loop.")
                 break
            torch.cuda.empty_cache()
            # 5. Fine-tune the Model
            new_model_path = fine_tune_model(
                base_model_path=BASE_MODEL_NAME,
                feedback_dataset_repo_id=feedback_repo_id,
                iteration_num=i,
                output_dir_base=FINETUNED_MODEL_DIR,
                lora_model_path = lora_model_path,
            )
            torch.cuda.empty_cache()
            # Update the model path for the next iteration
            lora_model_path = new_model_path #+ "/pytorch_lora_weights.safetensors"
            
            print(f"✅ Iteration {i} complete. Next iteration will use model {BASE_MODEL_NAME} with LoRA weights: {lora_model_path}")

        except Exception as e:
            print(f"❌❌❌ An error occurred during iteration {i}: {e} ❌❌❌")
            import traceback
            traceback.print_exc()
            print(f"Stopping the loop due to error.")
            break # Stop the loop if any step fails critically

        iteration_end_time = datetime.now()
        print(f"Iteration {i} duration: {iteration_end_time - iteration_start_time}")


    print("\n🏁 Training loop finished.")
    print(f"The final model is {BASE_MODEL_NAME} with LoRA weights path: {lora_model_path}")

if __name__ == "__main__":
    from datetime import datetime
    start_time = datetime.now()
    run_training_loop()
    
    end_time = datetime.now()
    print(f"\nTotal execution time: {end_time - start_time}")
