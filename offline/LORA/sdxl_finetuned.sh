accelerate launch train_diffusion_dpo_sdxl_finetuned.py \
  --pretrained_model_name_or_path=stabilityai/stable-diffusion-xl-base-1.0  \
  --pretrained_vae_model_name_or_path=madebyollin/sdxl-vae-fp16-fix \
  --output_dir="diffusion-sdxl-dpo_v2" \
  --mixed_precision="fp16" \
  --dataset_name=MarvinKoch/offlinedata3000promptsfinetuned \
  --cache_dir="/home/jupyter/marvin/DS_Lab/DiffusionDPO/datasets/rapidata" \
  --train_batch_size=2 \
  --dataloader_num_workers=2 \
  --gradient_accumulation_steps=2 \
  --gradient_checkpointing \
  --use_8bit_adam \
  --rank=8 \
  --learning_rate=5e-6 \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=500 \
  --max_train_steps=5000 \
  --checkpointing_steps=400 \
  --seed="0" \
  --report_to="wandb" \
  --dataset_split_name="train" \
  --enable_xformers_memory_efficient_attention \
  --run_validation \
  --validation_steps=400 \
  --reference_lora_model_path="artificialguybr/LogoRedmond-LogoLoraForSDXL-V2" \
  --beta_dpo=40 \
  --max_grad_norm=0.5 \
  --adam_weight_decay=0 \


 
  