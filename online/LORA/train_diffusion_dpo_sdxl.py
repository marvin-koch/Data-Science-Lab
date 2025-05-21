#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 bram-w, The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import contextlib
import io
import logging
import math
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
import wandb
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig
from .validation_step_rlhf import run_order

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.loaders import StableDiffusionXLLoraLoaderMixin
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, convert_state_dict_to_diffusers
from diffusers.utils.import_utils import is_xformers_available
from math import ceil


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.25.0.dev0")

logger = get_logger(__name__)

global_step = 0

VALIDATION_IMG_SAVE_PATH = os.path.join(os.getcwd(), "valid_images_finetuned/")

VALIDATION_PROMPTS_SHORT = [
    "futuristic logo for a AI lab, using red and black colors, icon inside shape, with a spiral icon, in a embossed style.",
    "playful logo for a music festival, using lavender and gray colors, icon beside, with a rabbit icon, in a deep shadows style.",
    "cartoon logo for a AI lab, using green and yellow colors, sharp edges, with a drone icon, in a retro vibeneon glow style.",
    "abstract logo for a eco company, using yellow and black colors, circular frame, with a rocket icon, in a polished style.",
    "playful logo for a music festival, using blue and white colors, icon inside shape, with a moon icon, in a neon lights style.",
    "cartoon logo for a gaming company, using red and black colors, icon above, with a tiger icon, in a vintage texture style.",
    "mascot logo for a construction, using orange and teal colors, clean lines, with a antenna icon, in a silhouette style.",
    "rustic logo for a coffee shop, using orange and teal colors, thin outline, with a rabbit icon, in a silhouette style.",
    "chic logo for a music festival, using black and gold colors, circular frame, with a spiral icon, in a metallic style.",
    "artistic logo for a bookstore, using maroon and beige colors, clean lines, with a antenna icon, in a matte finish style.",
    "smooth logo for a education platform, using black and red colors, symmetrical layout, with a guitar icon, in a flat style.",
    "hand-drawn logo for a cybersecurity firm, using green and yellow colors, icon above, with a sun icon, in a retro vibeneon glow style.",
    "handmade logo for a coffee shop, using white and silver colors, thin outline, with a boat icon, in a soft glow style.",
    "bright logo for a skincare brand, using white and black colors, symmetrical layout, with a flame icon, in a colorful style.",
    "elegant logo for a finance app, using pink and purple colors, sharp edges, with a satellite icon, in a dynamic lighting style.",
    "mascot logo for a comic book store, using peach and mint colors, clean lines, with a bear icon, in a glossy style.",
    "edgy logo for a education platform, using yellow and black colors, asymmetrical layout, with a yin yang icon, in a watercolor style.",
    "bold logo for a finance app, using pink and purple colors, stacked layout, with a drone icon, in a sketchy style.",
    "flat logo for a medical clinic, using black and gold colors, thick outline, with a paintbrush icon, in a vintage texture style.",
    "cartoon logo for a gaming company, using sky blue and cream colors, thick outline, with a cat icon, in a dynamic lighting style.",
    "vintage logo for a fitness brand, using blue and white colors, clean lines, with a lion icon, in a soft glow style.",
    "bold logo for a fitness brand, using white and black colors, thick outline, with a eagle icon, in a retro vibeneon glow style.",
    "abstract logo for a car brand, using turquoise and white colors, circular frame, with a raindrop icon, in a shadowed style.",
    "cartoon logo for a travel agency, using mint and coral colors, simple shape, with a drone icon, in a glossy style.",
    "playful logo for a yoga studio, using lime green and charcoal colors, simple shape, with a rocket icon, in a neon lights style."
]

VALIDATION_PROMPTS = [
    "futuristic logo for a AI lab, using red and black colors, icon inside shape, with a spiral icon, in a embossed style. Single centered logo on a plain background. No duplicates or mockups.",
    "playful logo for a music festival, using lavender and gray colors, icon beside, with a rabbit icon, in a deep shadows style. Single centered logo on a plain background. No duplicates or mockups.",
    "cartoon logo for a AI lab, using green and yellow colors, sharp edges, with a drone icon, in a retro vibeneon glow style. Single centered logo on a plain background. No duplicates or mockups.",
    "abstract logo for a eco company, using yellow and black colors, circular frame, with a rocket icon, in a polished style. Single centered logo on a plain background. No duplicates or mockups.",
    "playful logo for a music festival, using blue and white colors, icon inside shape, with a moon icon, in a neon lights style. Single centered logo on a plain background. No duplicates or mockups.",
    "cartoon logo for a gaming company, using red and black colors, icon above, with a tiger icon, in a vintage texture style. Single centered logo on a plain background. No duplicates or mockups.",
    "mascot logo for a construction, using orange and teal colors, clean lines, with a antenna icon, in a silhouette style. Single centered logo on a plain background. No duplicates or mockups.",
    "rustic logo for a coffee shop, using orange and teal colors, thin outline, with a rabbit icon, in a silhouette style. Single centered logo on a plain background. No duplicates or mockups.",
    "chic logo for a music festival, using black and gold colors, circular frame, with a spiral icon, in a metallic style. Single centered logo on a plain background. No duplicates or mockups.",
    "artistic logo for a bookstore, using maroon and beige colors, clean lines, with a antenna icon, in a matte finish style. Single centered logo on a plain background. No duplicates or mockups.",
    "smooth logo for a education platform, using black and red colors, symmetrical layout, with a guitar icon, in a flat style. Single centered logo on a plain background. No duplicates or mockups.",
    "hand-drawn logo for a cybersecurity firm, using green and yellow colors, icon above, with a sun icon, in a retro vibeneon glow style. Single centered logo on a plain background. No duplicates or mockups.",
    "handmade logo for a coffee shop, using white and silver colors, thin outline, with a boat icon, in a soft glow style. Single centered logo on a plain background. No duplicates or mockups.",
    "bright logo for a skincare brand, using white and black colors, symmetrical layout, with a flame icon, in a colorful style. Single centered logo on a plain background. No duplicates or mockups.",
    "elegant logo for a finance app, using pink and purple colors, sharp edges, with a satellite icon, in a dynamic lighting style. Single centered logo on a plain background. No duplicates or mockups.",
    "mascot logo for a comic book store, using peach and mint colors, clean lines, with a bear icon, in a glossy style. Single centered logo on a plain background. No duplicates or mockups.",
    "edgy logo for a education platform, using yellow and black colors, asymmetrical layout, with a yin yang icon, in a watercolor style. Single centered logo on a plain background. No duplicates or mockups.",
    "bold logo for a finance app, using pink and purple colors, stacked layout, with a drone icon, in a sketchy style. Single centered logo on a plain background. No duplicates or mockups.",
    "flat logo for a medical clinic, using black and gold colors, thick outline, with a paintbrush icon, in a vintage texture style. Single centered logo on a plain background. No duplicates or mockups.",
    "cartoon logo for a gaming company, using sky blue and cream colors, thick outline, with a cat icon, in a dynamic lighting style. Single centered logo on a plain background. No duplicates or mockups.",
    "vintage logo for a fitness brand, using blue and white colors, clean lines, with a lion icon, in a soft glow style. Single centered logo on a plain background. No duplicates or mockups.",
    "bold logo for a fitness brand, using white and black colors, thick outline, with a eagle icon, in a retro vibeneon glow style. Single centered logo on a plain background. No duplicates or mockups.",
    "abstract logo for a car brand, using turquoise and white colors, circular frame, with a raindrop icon, in a shadowed style. Single centered logo on a plain background. No duplicates or mockups.",
    "cartoon logo for a travel agency, using mint and coral colors, simple shape, with a drone icon, in a glossy style. Single centered logo on a plain background. No duplicates or mockups.",
    "playful logo for a yoga studio, using lime green and charcoal colors, simple shape, with a rocket icon, in a neon lights style. Single centered logo on a plain background. No duplicates or mockups."
]

VALIDATION_BATCH_SIZE = 4  # Or whatever your batch limit is


TRAINING_ADAPTER = "training adapter"
REFERENCE_ADAPTER = "reference adapter"
VALIDATION_IMG_SAVE_PATH = os.path.join(os.getcwd(), "valid_images_finetuned/")

def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")


def log_validation(args, unet, vae, accelerator, weight_dtype, epoch, global_step, is_final_validation=False):
    logger.info(f"Running validation for global step:{global_step}  \n Generating images with prompts:\n" f" {VALIDATION_PROMPTS}.")

    # if is_final_validation:
    #     if args.mixed_precision == "fp16":
    #         vae.to(weight_dtype)

    # create pipeline
    pipeline = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )
    if not is_final_validation:
        pipeline.unet = accelerator.unwrap_model(unet)
    else:
        pipeline.load_lora_weights(args.output_dir, weight_name="pytorch_lora_weights.safetensors")

    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    # run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None
    images = []
    # context = contextlib.nullcontext() if is_final_validation else torch.cuda.amp.autocast()
    context = torch.cuda.amp.autocast()

    guidance_scale = 7.5
    num_inference_steps = 30
    # if args.is_turbo:
    #     guidance_scale = 0.0
    #     num_inference_steps = 4
    # for i,prompt in enumerate(VALIDATION_PROMPTS):
    #     with context:
    #         image = pipeline(
    #             prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, generator=generator
    #         ).images[0]
    #         images.append(image)
    #         step_save_dir = os.path.join(VALIDATION_IMG_SAVE_PATH, f"step_{global_step}/") # Define step-specific dir
    #         print(f"Saving images to {step_save_dir}") # Log the correct directory
    #         os.makedirs(step_save_dir, exist_ok=True) # Ensure the directory exists
    #         filename = f"{global_step}_image_{i:04d}.png"
    #         full_image_path = os.path.join(step_save_dir, filename) # Construct full path *inside* step dir
    #         image.save(full_image_path) # Save the image to the correct path
    
    
    
    
 
    num_batches = ceil(len(VALIDATION_PROMPTS) / VALIDATION_BATCH_SIZE)

    step_save_dir = os.path.join(VALIDATION_IMG_SAVE_PATH, f"iter_{args.iter_num}", f"step_{global_step}/")
    os.makedirs(step_save_dir, exist_ok=True)
    print(f"Saving images to {step_save_dir}")

    image_idx = 0  # Keeps track of image index globally across batches

    for batch_num in range(num_batches):
        batch_start = batch_num * VALIDATION_BATCH_SIZE
        batch_end = batch_start + VALIDATION_BATCH_SIZE
        prompt_batch = VALIDATION_PROMPTS[batch_start:batch_end]

        with context:
            print(f"Generating batch {batch_num+1}/{num_batches}...")
            results = pipeline(
                prompt_batch,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator
            )
            batch_images = results.images

        for image in batch_images:
            filename = f"{global_step}_image_{image_idx:04d}.png"
            full_image_path = os.path.join(step_save_dir, filename)
            image.save(full_image_path)
            images.append(image)
            image_idx += 1

    metrics = run_order(os.path.join(VALIDATION_IMG_SAVE_PATH, f"iter_{args.iter_num}", f"step_{global_step}/"),VALIDATION_IMG_SAVE_PATH,VALIDATION_PROMPTS_SHORT,global_step)
    accelerator.log(metrics, step=global_step)
    # accelerator.log(logs, step=global_step)
    tracker_key = "test" if is_final_validation else "validation"
    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images(tracker_key, np_images, epoch, dataformats="NHWC")
        if tracker.name == "wandb":
            tracker.log(
                {
                    tracker_key: [
                        wandb.Image(image, caption=f"{i}: {VALIDATION_PROMPTS[i]}") for i, image in enumerate(images)
                    ]
                }
            )


    # Also log images without the LoRA params for comparison.
    # if is_final_validation:
    #     pipeline.disable_lora()
    #     no_lora_images = [
    #         pipeline(
    #             prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, generator=generator
    #         ).images[0]
    #         for prompt in VALIDATION_PROMPTS
    #     ]

    #     for tracker in accelerator.trackers:
    #         if tracker.name == "tensorboard":
    #             np_images = np.stack([np.asarray(img) for img in no_lora_images])
    #             tracker.writer.add_images("test_without_lora", np_images, epoch, dataformats="NHWC")
    #         if tracker.name == "wandb":
    #             tracker.log(
    #                 {
    #                     "test_without_lora": [
    #                         wandb.Image(image, caption=f"{i}: {VALIDATION_PROMPTS[i]}")
    #                         for i, image in enumerate(no_lora_images)
    #                     ]
    #                 }
    #             )


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_vae_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained VAE model with better numerical stability. More details: https://github.com/huggingface/diffusers/pull/4038.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_split_name",
        type=str,
        default="validation",
        help="Dataset split to be used during training. Helpful to specify for conducting experimental runs.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--run_validation",
        default=False,
        action="store_true",
        help="Whether to run validation inference in between training and also after training. Helps to track progress.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=200,
        help="Run validation every X steps.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="diffusion-dpo-lora",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--iter_num", type=int, default=0)

    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--vae_encode_batch_size",
        type=int,
        default=8,
        help="Batch size to use for VAE encoding of the images for efficient processing.",
    )
    parser.add_argument(
        "--no_hflip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--random_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to random crop the input images to the resolution. If not set, the images will be center-cropped."
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--beta_dpo",
        type=int,
        default=5000,
        help="DPO KL Divergence penalty.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--prior_generation_precision",
        type=str,
        default=None,
        choices=["no", "fp32", "fp16", "bf16"],
        help=(
            "Choose prior generation precision between fp32, fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to  fp16 if a GPU is available else fp32."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--is_turbo",
        action="store_true",
        help=("Use if tuning SDXL Turbo instead of SDXL"),
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--tracker_name",
        type=str,
        default="diffusion-dpo-lora-sdxl",
        help=("The name of the tracker to report results to."),
    )
    parser.add_argument(
        "--reference_lora_model_path",
        type=str,
        default=None,
        help=(
            "Path to pretrained LoRA weights (directory containing 'pytorch_lora_weights.safetensors') "
            "to be loaded into **both** the training model (as initialization) and the reference model. " # <-- Updated help string
            "If None, the training model starts with fresh LoRA layers, and the base model (without LoRA) is used as reference."
        ),
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.dataset_name is None:
        raise ValueError("Must provide a `dataset_name`.")

    if args.is_turbo:
        assert "turbo" in args.pretrained_model_name_or_path

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def tokenize_captions(tokenizers, examples):
    captions = []
    for prompt in examples["prompt"]:
        captions.append(prompt)

    tokens_one = tokenizers[0](
        captions, truncation=True, padding="max_length", max_length=tokenizers[0].model_max_length, return_tensors="pt"
    ).input_ids
    tokens_two = tokenizers[1](
        captions, truncation=True, padding="max_length", max_length=tokenizers[1].model_max_length, return_tensors="pt"
    ).input_ids

    return tokens_one, tokens_two


@torch.no_grad()
def encode_prompt(text_encoders, text_input_ids_list):
    prompt_embeds_list = []

    for i, text_encoder in enumerate(text_encoders):
        text_input_ids = text_input_ids_list[i]

        prompt_embeds = text_encoder(
            text_input_ids.to(text_encoder.device),
            output_hidden_states=True,
        )

        # We are only ALWAYS interested in the pooled output of the final text encoder
        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds.hidden_states[-2]
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
        prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return prompt_embeds, pooled_prompt_embeds


def main(args):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load the tokenizers
    tokenizer_one = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        use_fast=False,
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
        use_fast=False,
    )

    # import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    def enforce_zero_terminal_snr(scheduler):
        # Modified from https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_ddpm.py#L93
        # Original implementation https://arxiv.org/pdf/2305.08891.pdf
        # Turbo needs zero terminal SNR
        # Turbo: https://static1.squarespace.com/static/6213c340453c3f502425776e/t/65663480a92fba51d0e1023f/1701197769659/adversarial_diffusion_distillation.pdf
        # Convert betas to alphas_bar_sqrt
        alphas = 1 - scheduler.betas
        alphas_bar = alphas.cumprod(0)
        alphas_bar_sqrt = alphas_bar.sqrt()

        # Store old values.
        alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
        alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()
        # Shift so last timestep is zero.
        alphas_bar_sqrt -= alphas_bar_sqrt_T
        # Scale so first timestep is back to old value.
        alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

        alphas_bar = alphas_bar_sqrt**2
        alphas = alphas_bar[1:] / alphas_bar[:-1]
        alphas = torch.cat([alphas_bar[0:1], alphas])

        alphas_cumprod = torch.cumprod(alphas, dim=0)
        scheduler.alphas_cumprod = alphas_cumprod
        return

    if args.is_turbo:
        enforce_zero_terminal_snr(noise_scheduler)

    text_encoder_one = text_encoder_cls_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    text_encoder_two = text_encoder_cls_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant
    )
    vae_path = (
        args.pretrained_model_name_or_path
        if args.pretrained_vae_model_name_or_path is None
        else args.pretrained_vae_model_name_or_path
    )
    vae = AutoencoderKL.from_pretrained(
        vae_path,
        subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
        revision=args.revision,
        variant=args.variant,
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
    )

    # We only train the additional adapter LoRA layers
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    unet.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move unet and text_encoders to device and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)

    # The VAE is always in float32 to avoid NaN losses.
    vae.to(accelerator.device, dtype=torch.float32)

    # Set up LoRA.
    #reference_unet = None
    
    unet_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    """
    if args.reference_lora_model_path:
        logger.info(f"Loading Reference LoRA UNet from: {args.reference_lora_model_path}")
        # Create a new UNet instance for the reference model
        reference_unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
        )
        
        logger.info("Adding adapter configuration to the reference UNet...")
        reference_unet.add_adapter(unet_lora_config)
        logger.info("Adapter configuration added.")
        try:
            # Load the LoRA weights into the reference UNet
            #reference_unet.load_lora_weights(args.reference_lora_model_path, weight_name="pytorch_lora_weights.safetensors")
            lora_state_dict, network_alphas = StableDiffusionXLLoraLoaderMixin.lora_state_dict(
                args.reference_lora_model_path, weight_name="pytorch_lora_weights.safetensors"
            )
            StableDiffusionXLLoraLoaderMixin.load_lora_into_unet(
                lora_state_dict, network_alphas=network_alphas, unet=reference_unet # Pass the reference UNet object
            )
            logger.info(f"Successfully loaded reference LoRA weights from {args.reference_lora_model_path}")
        except Exception as e:
            logger.error(f"Failed to load reference LoRA weights from {args.reference_lora_model_path}: {e}")
            raise e

        # Move reference unet to device and set dtype
        reference_unet.to(accelerator.device, dtype=weight_dtype)
        # Ensure reference model is in eval mode and requires no gradients
        reference_unet.eval()
        reference_unet.requires_grad_(False)
    else:
        logger.info("Using the base model as the reference model (no reference LoRA specified).")
    
    """
  

    # --- Start Added LoRA Loading for Training UNet ---
    if args.reference_lora_model_path:
        # if "base_training" in unet.peft_config:
        #     unet.delete_adapter("base_training")
            
        # if "reference" in unet.peft_config:
        #     unet.delete_adapter("reference")

        # unet.add_adapter(unet_lora_config, "base_training")
        # logger.info("Added LoRA adapter 'train'.")
        
        # unet.add_adapter(unet_lora_config, adapter_name="reference")
        # logger.info("Added LoRA adapter 'reference'.")
        # logger.info(f"Loading initial LoRA weights for training UNet from: {args.reference_lora_model_path}")
        try:
            # Load the LoRA weights into the training UNet
            #unet.load_lora_weights(args.reference_lora_model_path, weight_name="pytorch_lora_weights.safetensors")
            # lora_state_dict, network_alphas = StableDiffusionXLLoraLoaderMixin.lora_state_dict(
            #     args.reference_lora_model_path, weight_name="pytorch_lora_weights.safetensors"
            # )
            # StableDiffusionXLLoraLoaderMixin.load_lora_into_unet(
            #     lora_state_dict, network_alphas=network_alphas, unet=unet # Pass the reference UNet object
            # )
            
            # unet.load_lora_adapter(
            #     args.reference_lora_model_path,
            #     adapter_name="base_training",
            #     weight_name="pytorch_lora_weights.safetensors", # Adjust if filename differs
            # )
            # unet.load_lora_adapter(
            #     args.reference_lora_model_path,
            #     adapter_name="reference",
            #     weight_name="pytorch_lora_weights.safetensors", # Adjust if filename differs
            # )
            weight_name="pytorch_lora_weights.safetensors"
            if args.reference_lora_model_path == "artificialguybr/LogoRedmond-LogoLoraForSDXL-V2":
                weight_name="LogoRedmondV2-Logo-LogoRedmAF.safetensors"
                
            lora_state_dict, network_alphas = StableDiffusionXLLoraLoaderMixin.lora_state_dict(
                args.reference_lora_model_path, weight_name=weight_name,  unet_config=unet.config 
            )
            unet.add_adapter(unet_lora_config, "def")

            StableDiffusionXLLoraLoaderMixin.load_lora_into_unet(
                lora_state_dict,
                network_alphas,
                unet, # Pass the reference UNet object as the first argument
                adapter_name=TRAINING_ADAPTER # Use 'adapter_name'
            )
            StableDiffusionXLLoraLoaderMixin.load_lora_into_unet(
                lora_state_dict,
                network_alphas,
                unet, # Pass the reference UNet object again
                adapter_name=REFERENCE_ADAPTER # Use 'adapter_name'
            )
           
            # --- Debugging: Check registered adapters ---
            if hasattr(unet, 'peft_config'):
                print(f"Registered adapters after loading: {list(unet.peft_config.keys())}")
                if not unet.peft_config:
                    print("WARNING: peft_config is empty after loading attempts!")
            else:
                print("WARNING: UNet does not have peft_config attribute.")
            # --- End Debugging ---
            
            unet.set_adapter(TRAINING_ADAPTER)
            logger.info("Adapter 'train' set as active for training.")

            for name, param in unet.named_parameters():
                # Unfreeze only the parameters belonging to the 'train' adapter
                # PEFT parameter names often include 'lora_A.train', 'lora_B.train' etc.
                if ".lora_" in name and ".def." in name:
                    param.requires_grad_(False)
                # Ensure reference adapter parameters remain frozen (they should be by default now)
                if ".lora_" in name and "reference" in name:
                    param.requires_grad_(False)
                if ".lora_" in name and "training" in name:
                    param.requires_grad_(True)
            
            logger.info(f"Successfully loaded initial LoRA weights for training UNet from {args.reference_lora_model_path}")
        except Exception as e:

            logger.error(f"Failed to load initial LoRA weights for training UNet from {args.reference_lora_model_path}: {e}")
            raise e
    else:
        # Add adapter and make sure the trainable params are in float32.
        unet.add_adapter(unet_lora_config)
        logger.info("Initializing training UNet with new LoRA layers (no initial weights specified).")
    # --- End Added LoRA Loading for Training UNet ---

    if args.mixed_precision == "fp16":
        for param in unet.parameters():
            # only upcast trainable parameters (LoRA) into fp32
            if param.requires_grad:
                param.data = param.to(torch.float32)
        
    

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
            


        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()



    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            # there are only two options here. Either are just the unet attn processor layers
            # or there are the unet and text encoder atten layers
            unet_lora_layers_to_save = None

            for model in models:
                if isinstance(model, type(accelerator.unwrap_model(unet))):
                    if args.reference_lora_model_path:
                        unet_lora_layers_to_save = convert_state_dict_to_diffusers(get_peft_model_state_dict(model, adapter_name=TRAINING_ADAPTER))
                    else:
                        unet_lora_layers_to_save = convert_state_dict_to_diffusers(get_peft_model_state_dict(model))
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()
          
            StableDiffusionXLLoraLoaderMixin.save_lora_weights(output_dir, unet_lora_layers=unet_lora_layers_to_save)

    def load_model_hook(models, input_dir):
        unet_ = None

        while len(models) > 0:
            model = models.pop()

            if isinstance(model, type(accelerator.unwrap_model(unet))):
                unet_ = model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        lora_state_dict, network_alphas = StableDiffusionXLLoraLoaderMixin.lora_state_dict(input_dir)
        if args.reference_lora_model_path:
            StableDiffusionXLLoraLoaderMixin.load_lora_into_unet(
                        lora_state_dict, network_alphas=network_alphas, unet=unet_, adapter_name=TRAINING_ADAPTER
                    )
        else:
            StableDiffusionXLLoraLoaderMixin.load_lora_into_unet(
                lora_state_dict, network_alphas=network_alphas, unet=unet_
            )

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation
    params_to_optimize = list(filter(lambda p: p.requires_grad, unet.parameters()))
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Dataset and DataLoaders creation:
    train_dataset = load_dataset(
        args.dataset_name,
        cache_dir=args.cache_dir,
        split=args.dataset_split_name,
    )
    def determine_label(row):
        if row['weighted_results_image1_preference'] > row['weighted_results_image2_preference']:
            return 1  # Label 0 for image1
        elif row['weighted_results_image1_preference'] < row['weighted_results_image2_preference']:
            return 0  # Label 1 for image2
        else:
            return 1  # Label -1 for a tie (optional)
        
    # train_dataset = train_dataset.map(lambda x: {'label_0': determine_label(x), 'caption': x['prompt'],  # Rename "prompt" -> "caption"
    #      'jpg_0': x['image1'], 'jpg_1': x['image2']}, remove_columns=['votes_image1', 'votes_image2', 'prompt', 'image1', 'image2'], num_proc=2)
   
    train_dataset = train_dataset.map(lambda x: {'label_0': determine_label(x)}, remove_columns=['weighted_results_image1_preference', 'weighted_results_image2_preference'], num_proc=2)


    
    # Preprocessing the datasets.
    train_resize = transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR)
    train_crop = transforms.RandomCrop(args.resolution) if args.random_crop else transforms.CenterCrop(args.resolution)
    train_flip = transforms.RandomHorizontalFlip(p=1.0)
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize([0.5], [0.5])

    def preprocess_train(examples):
        all_pixel_values = []
        images = [(im_bytes).convert("RGB") for im_bytes in examples["image1"]]
        original_sizes = [(image.height, image.width) for image in images]
        crop_top_lefts = []

        for col_name in ["image1", "image2"]:
            images = [(im_bytes).convert("RGB") for im_bytes in examples[col_name]]
            if col_name == "image2":
                # Need to bring down the image to the same resolution.
                # This seems like the simplest reasonable approach.
                # "::-1" because PIL resize takes (width, height).
                images = [image.resize(original_sizes[i][::-1]) for i, image in enumerate(images)]
            pixel_values = [to_tensor(image) for image in images]
            all_pixel_values.append(pixel_values)

        # Double on channel dim, jpg_y then jpg_w
        im_tup_iterator = zip(*all_pixel_values)
        combined_pixel_values = []
        for im_tup, label_0 in zip(im_tup_iterator, examples["label_0"]):
            if label_0 == 0:
                im_tup = im_tup[::-1]

            combined_im = torch.cat(im_tup, dim=0)  # no batch dim

            # Resize.
            combined_im = train_resize(combined_im)

            # Flipping.
            if not args.no_hflip and random.random() < 0.5:
                combined_im = train_flip(combined_im)

            # Cropping.
            if not args.random_crop:
                y1 = max(0, int(round((combined_im.shape[1] - args.resolution) / 2.0)))
                x1 = max(0, int(round((combined_im.shape[2] - args.resolution) / 2.0)))
                combined_im = train_crop(combined_im)
            else:
                y1, x1, h, w = train_crop.get_params(combined_im, (args.resolution, args.resolution))
                combined_im = crop(combined_im, y1, x1, h, w)

            crop_top_left = (y1, x1)
            crop_top_lefts.append(crop_top_left)
            combined_im = normalize(combined_im)
            combined_pixel_values.append(combined_im)

        examples["pixel_values"] = combined_pixel_values
        examples["original_sizes"] = original_sizes
        examples["crop_top_lefts"] = crop_top_lefts
        tokens_one, tokens_two = tokenize_captions([tokenizer_one, tokenizer_two], examples)
        examples["input_ids_one"] = tokens_one
        examples["input_ids_two"] = tokens_two
        return examples

    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            train_dataset = train_dataset.shuffle(seed=args.seed).select(range(args.max_train_samples))
        # Set the training transforms
        train_dataset = train_dataset.with_transform(preprocess_train)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        original_sizes = [example["original_sizes"] for example in examples]
        crop_top_lefts = [example["crop_top_lefts"] for example in examples]
        input_ids_one = torch.stack([example["input_ids_one"] for example in examples])
        input_ids_two = torch.stack([example["input_ids_two"] for example in examples])

        return {
            "pixel_values": pixel_values,
            "input_ids_one": input_ids_one,
            "input_ids_two": input_ids_two,
            "original_sizes": original_sizes,
            "crop_top_lefts": crop_top_lefts,
        }

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers(args.tracker_name, config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    unet.train()
    vae.eval()
    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # (batch_size, 2*channels, h, w) -> (2*batch_size, channels, h, w)
                pixel_values = batch["pixel_values"].to(dtype=vae.dtype)
                feed_pixel_values = torch.cat(pixel_values.chunk(2, dim=1))

                latents = []
                for i in range(0, feed_pixel_values.shape[0], args.vae_encode_batch_size):
                    with torch.no_grad():
                        latents.append(
                            vae.encode(feed_pixel_values[i : i + args.vae_encode_batch_size]).latent_dist.sample()
                        )
                latents = torch.cat(latents, dim=0)
                latents = latents * vae.config.scaling_factor
                if args.pretrained_vae_model_name_or_path is None:
                    latents = latents.to(weight_dtype)

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents).chunk(2)[0].repeat(2, 1, 1, 1)

                # Sample a random timestep for each image
                bsz = latents.shape[0] // 2
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device, dtype=torch.long
                ).repeat(2)
                if args.is_turbo:
                    # Learn a 4 timestep schedule
                    timesteps_0_to_3 = timesteps % 4
                    timesteps = 250 * timesteps_0_to_3 + 249

                # Add noise to the model input according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_model_input = noise_scheduler.add_noise(latents, noise, timesteps)

                # time ids
                def compute_time_ids(original_size, crops_coords_top_left):
                    # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
                    target_size = (args.resolution, args.resolution)
                    add_time_ids = list(original_size + crops_coords_top_left + target_size)
                    add_time_ids = torch.tensor([add_time_ids])
                    add_time_ids = add_time_ids.to(accelerator.device, dtype=weight_dtype)
                    return add_time_ids

                add_time_ids = torch.cat(
                    [compute_time_ids(s, c) for s, c in zip(batch["original_sizes"], batch["crop_top_lefts"])]
                ).repeat(2, 1)

                # Get the text embedding for conditioning
                prompt_embeds, pooled_prompt_embeds = encode_prompt(
                    [text_encoder_one, text_encoder_two], [batch["input_ids_one"], batch["input_ids_two"]]
                )
                prompt_embeds = prompt_embeds.repeat(2, 1, 1)
                pooled_prompt_embeds = pooled_prompt_embeds.repeat(2, 1)

                # Predict the noise residual
                model_pred = unet(
                    noisy_model_input,
                    timesteps,
                    prompt_embeds,
                    added_cond_kwargs={"time_ids": add_time_ids, "text_embeds": pooled_prompt_embeds},
                ).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # Compute losses.
                model_losses = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                model_losses = model_losses.mean(dim=list(range(1, len(model_losses.shape))))
                model_losses_w, model_losses_l = model_losses.chunk(2)

                # For logging
                raw_model_loss = 0.5 * (model_losses_w.mean() + model_losses_l.mean())
                model_diff = model_losses_w - model_losses_l  # These are both LBS (as is t)

                # Reference model predictions.
                #accelerator.unwrap_model(unet).disable_adapters()
                # --- Start Modified Reference Prediction ---
                # Reference model predictions.
                with torch.no_grad():
                    if args.reference_lora_model_path:
                        unet.eval() # Switch to evaluation mode for reference pass (disables dropout etc.)

                        unet.set_adapter(REFERENCE_ADAPTER)

                        # Use the dedicated reference UNet (already has LoRA loaded)
                        ref_preds = unet(
                            noisy_model_input,
                            timesteps,
                            prompt_embeds,
                            added_cond_kwargs={"time_ids": add_time_ids, "text_embeds": pooled_prompt_embeds},
                        ).sample
                        
                        unet.set_adapter(TRAINING_ADAPTER)
                        
                        unet.train()

                    else:
                        # Original behavior: Use the base model by disabling adapters on the *training* UNet
                        accelerator.unwrap_model(unet).disable_adapters()
                        ref_preds = unet(
                            noisy_model_input,
                            timesteps,
                            prompt_embeds,
                            added_cond_kwargs={"time_ids": add_time_ids, "text_embeds": pooled_prompt_embeds},
                        ).sample
                        # Re-enable adapters only if we disabled them
                        accelerator.unwrap_model(unet).enable_adapters()

                    # Calculate reference loss (remains the same)
                    ref_loss = F.mse_loss(ref_preds.float(), target.float(), reduction="none")
                    ref_loss = ref_loss.mean(dim=list(range(1, len(ref_loss.shape))))

                    ref_losses_w, ref_losses_l = ref_loss.chunk(2)
                    ref_diff = ref_losses_w - ref_losses_l
                    raw_ref_loss = ref_loss.mean()

                # Only re-enable adapters if the reference_unet was None (handled inside the else block now)
                # accelerator.unwrap_model(unet).enable_adapters() # This line is removed/integrated above
                # --- End Modified Reference Prediction ---

                # Final loss.
                scale_term = -0.5 * args.beta_dpo
                inside_term = scale_term * (model_diff - ref_diff)
                loss = -1 * F.logsigmoid(inside_term).mean()

                implicit_acc = (inside_term > 0).sum().float() / inside_term.size(0)
                implicit_acc += 0.5 * (inside_term == 0).sum().float() / inside_term.size(0)

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_optimize, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                    if args.run_validation and global_step % args.validation_steps == 0:
                        log_validation(
                            args, unet=unet, vae=vae, accelerator=accelerator, weight_dtype=weight_dtype, epoch=epoch, global_step=global_step
                        )

            logs = {
                "loss": loss.detach().item(),
                "raw_model_loss": raw_model_loss.detach().item(),
                "ref_loss": raw_ref_loss.detach().item(),
                "implicit_acc": implicit_acc.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        unet = unet.to(torch.float32)
        if args.reference_lora_model_path:
            unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet, adapter_name=TRAINING_ADAPTER))

        else:
            unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))

        StableDiffusionXLLoraLoaderMixin.save_lora_weights(
            save_directory=args.output_dir,
            unet_lora_layers=unet_lora_state_dict,
            text_encoder_lora_layers=None,
            text_encoder_2_lora_layers=None,
        )

        # Final validation?
        if args.run_validation:
            log_validation(
                args,
                unet=None,
                vae=vae,
                accelerator=accelerator,
                weight_dtype=weight_dtype,
                epoch=epoch,
                global_step=global_step,
                is_final_validation=True,
            )

        if args.push_to_hub:
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


# if __name__ == "__main__":
#     args = parse_args()
#     main(args)


