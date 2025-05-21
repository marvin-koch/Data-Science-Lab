# Make sure VALIDATION_PROMPTS is defined globally in your script
# VALIDATION_PROMPTS = ["Prompt 1", "Prompt 2", ...] # Length should match base images

@torch.no_grad()
def log_validation(
    args: argparse.Namespace,
    unet: UNet2DConditionModel,
    vae: AutoencoderKL,
    accelerator: Accelerator,
    weight_dtype: torch.dtype,
    global_step: int | str,
    rapi_client=None,
):
    """
    Runs validation: generates LoRA images for VALIDATION_PROMPTS,
    pairs them with pre-existing base images from args.base_image_dir,
    and optionally submits the pairs to Rapidata.
    """
    if not accelerator.is_main_process:
        return

    logger.info(f"Running validation comparison for step: {global_step} using pre-existing base images...")

    # --- Base Image Directory Check ---
    base_image_dir = Path(args.base_image_dir)
    if not base_image_dir.is_dir():
        logger.error(f"Base image directory not found: {base_image_dir}. Skipping validation.")
        return

    num_prompts = len(VALIDATION_PROMPTS)
    logger.info(f"Using {num_prompts} prompts from VALIDATION_PROMPTS list.")

    # --- Configuration & Paths for this step ---
    validation_output_dir = Path(args.output_dir) / f"validation_{global_step}"
    lora_image_dir = validation_output_dir / "images_compare_lora" # Specific dir for LoRA images
    # metadata_dir = validation_output_dir / "metadata_compare" # Optional: If saving metadata
    os.makedirs(lora_image_dir, exist_ok=True)
    # os.makedirs(metadata_dir, exist_ok=True) # Optional

    # --- Load LoRA Pipeline ---
    logger.info("Loading LoRA pipeline with current UNet weights...")
    try:
        vae.to(accelerator.device)
        unet_lora = accelerator.unwrap_model(unet)
        pipe_lora = StableDiffusionXLPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            unet=unet_lora,
            vae=vae,
            revision=args.revision,
            variant=args.variant,
            torch_dtype=weight_dtype,
            use_safetensors=True,
        )
        pipe_lora = pipe_lora.to(accelerator.device)
        pipe_lora.set_progress_bar_config(disable=True)
        if args.enable_xformers_memory_efficient_attention:
            pipe_lora.enable_xformers_memory_efficient_attention()
        logger.info("LoRA pipeline loaded.")
    except Exception as e:
        logger.error(f"Failed to load LoRA pipeline: {e}")
        return

    # --- Generation (LoRA only) & Pair Preparation ---
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None
    context = torch.cuda.amp.autocast(enabled=accelerator.native_amp)
    guidance_scale = 0.0 if args.is_turbo else 5.0
    num_inference_steps = 4 if args.is_turbo else 25
    # Use a consistent batch size for validation generation
    validation_batch_size = args.train_batch_size # Or a fixed value like 4 or 8

    logger.info(f"Generating LoRA images for validation with guidance: {guidance_scale}, steps: {num_inference_steps}, batch_size: {validation_batch_size}")

    # Lists to hold data specifically for Rapidata submission
    prompts_for_rapidata = []
    datapoints_for_rapidata = [] # List of [absolute_base_path, absolute_lora_path]

    try:
        current_prompt_idx = 0
        for i in tqdm(range(0, num_prompts, validation_batch_size), desc=f"Validation LoRA Batch (step {global_step})"):
            batch_prompts = VALIDATION_PROMPTS[i : i + validation_batch_size]
            batch_indices = list(range(i, i + len(batch_prompts))) # Global indices (0 to num_prompts-1)

            if not batch_prompts: continue # Should not happen if num_prompts > 0

            with context:
                images_lora = pipe_lora(
                    prompt=batch_prompts,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    num_images_per_prompt=1,
                ).images

            # Process generated images in the batch
            for j, prompt in enumerate(batch_prompts):
                original_index = batch_indices[j] # The index within VALIDATION_PROMPTS (0 to num_prompts-1)
                lora_image = images_lora[j]

                # 1. Determine pre-existing base image path
                # Assuming base images are image0001.png, image0002.png, ...
                base_filename = f"image{original_index + 1:04d}.png"
                base_image_abs_path = (base_image_dir / base_filename).resolve()

                # 2. Define and save the new LoRA image
                lora_filename = f"lora_step{global_step}_idx{original_index + 1:04d}.png"
                lora_image_abs_path = (lora_image_dir / lora_filename).resolve()
                lora_image.save(lora_image_abs_path)

                # 3. Check if base image exists before adding pair
                if not base_image_abs_path.exists():
                    logger.warning(f"Base image missing for prompt index {original_index}: {base_image_abs_path}. Skipping pair.")
                    continue

                # 4. Add valid pair data for Rapidata
                prompts_for_rapidata.append(prompt)
                datapoints_for_rapidata.append([str(base_image_abs_path), str(lora_image_abs_path)])

                del lora_image

            del images_lora
            if accelerator.device.type == 'cuda':
                torch.cuda.empty_cache()

        logger.info(f"Generated {len(datapoints_for_rapidata)} LoRA images and found corresponding base images.")
        logger.info(f"🟢 Step {global_step} LoRA validation images saved to: {lora_image_dir}")

        # --- Submit to Rapidata (If enabled and pairs exist) ---
        if args.run_rapidata_comparison and rapi_client:
            if not datapoints_for_rapidata:
                logger.warning("No valid image pairs were prepared. Skipping Rapidata submission.")
            else:
                logger.info(f"Attempting to submit {len(datapoints_for_rapidata)} pairs to Rapidata for step {global_step}...")
                try:
                    order_name = f"{args.tracker_project_name} - Step {global_step}"
                    instruction = args.rapidata_instruction
                    responses_per_datapoint = args.rapidata_responses_per_dp
                    validation_set_id = args.rapidata_validation_set_id

                    # Submit the order (ensure non-blocking)
                    order = rapi_client.order.create_compare_order(
                        name=order_name,
                        instruction=instruction,
                        contexts=prompts_for_rapidata, # Use the prompts aligned with valid pairs
                        datapoints=datapoints_for_rapidata, # Use the list of [abs_base, abs_lora] paths
                        responses_per_datapoint=responses_per_datapoint,
                        validation_set_id=validation_set_id
                    )

                    order_id = getattr(order, 'id', None)
                    if order_id:
                        logger.info(f"✅ Successfully submitted Rapidata order. Order ID: {order_id}")
                        try:
                            # Log ID to accelerator trackers if step is an integer
                            current_step_int = int(global_step) if str(global_step).isdigit() else None
                            if current_step_int is not None:
                                accelerator.log({"rapidata_order_id": str(order_id)}, step=current_step_int)
                        except Exception as log_err:
                             logger.warning(f"Could not log rapidata order ID to accelerator: {log_err}")
                    else:
                        logger.warning("Rapidata order submitted, but could not retrieve Order ID.")

                except Exception as e:
                    logger.error(f"Failed to submit Rapidata order for step {global_step}: {e}", exc_info=True)

        # --- No step-specific metadata saving needed unless required downstream ---
        # If you needed image_mapping.json: build it here from the datapoints_for_rapidata list

    except Exception as e:
        logger.error(f"Error during LoRA validation image generation: {e}", exc_info=True)
    finally:
        # --- Cleanup LoRA Pipeline ---
        logger.info("Cleaning up LoRA validation pipeline...")
        del pipe_lora
        # unet_lora is just a reference from accelerator, no need to del
        if accelerator.device.type == 'cuda':
            torch.cuda.empty_cache()
        logger.info(f"Validation comparison for step {global_step} finished.")