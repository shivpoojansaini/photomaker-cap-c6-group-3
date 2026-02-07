#!/usr/bin/env python3
"""
PhotoMaker V2 CLI - Generate images without Gradio
Just edit the configuration below and run: python photomaker_cli.py
This version:
1. supports unlimited prompts per face
2. validates trigger word per prompt
3. extracts left/right face embeddings
4. defines all variables in correct order
5. removes all leftover PROMPT references
6. avoids NameErrors
7. keeps your adapter/sketch logic
8. returns the correct structure
"""

import torch
import torchvision.transforms.functional as TF
import numpy as np
import random
import os
import sys
from pathlib import Path

from diffusers.utils import load_image
from diffusers import EulerDiscreteScheduler, T2IAdapter
from huggingface_hub import hf_hub_download

from photomaker import PhotoMakerStableDiffusionXLAdapterPipeline
from photomaker import FaceAnalysis2, analyze_faces

from style_template import styles
from aspect_ratio_template import aspect_ratios

from PIL import Image, ImageDraw, ImageFont

# ============================================================
# CONFIGURATION - Edit these values directly
# ============================================================

# Input image(s) - provide path(s) to face image(s)
INPUT_IMAGES = [
    "/teamspace/studios/this_studio/PhotoMaker/Data/Input/Rafa&Fed_3.jpg",
    # "./input/face2.jpg",  # Add more images for better ID fidelity
]

# Prompt - must include 'img' trigger word
PROMPTS_FACE_LEFT = [
    "a photo of a man img wearing a hat",
]

PROMPTS_FACE_RIGHT = [
    "a photo of a man img wearing a hat and sunglasses",
]


# Output settings
OUTPUT_DIR = "/teamspace/studios/this_studio/PhotoMaker/Data/Output"
NUM_OUTPUTS = 1

# Style (check style_template.py for options)
STYLE_NAME = "Photographic (Default)"

# Negative prompt
NEGATIVE_PROMPT = "nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"

# Output dimensions
OUTPUT_WIDTH = 1024
OUTPUT_HEIGHT = 1024

# Generation parameters
NUM_STEPS = 50
GUIDANCE_SCALE = 5.0
STYLE_STRENGTH_RATIO = 20
SEED = None  # Set to None for random seed, or specify a number

# Sketch/Doodle settings (optional)
USE_SKETCH = False
SKETCH_IMAGE_PATH = None  # e.g., "./sketch.png"
ADAPTER_CONDITIONING_SCALE = 0.7
ADAPTER_CONDITIONING_FACTOR = 0.8

# ============================================================
# END OF CONFIGURATION
# ============================================================

MAX_SEED = np.iinfo(np.int32).max

from PIL import Image, ImageDraw, ImageFont

def add_watermark(image, text="© AI-Generated image by CAP-C6-Group_3", opacity=160):
    """
    Adds a semi-transparent text watermark to the bottom-right corner.
    Compatible with Pillow >=10 (no textsize()).
    """
    if image.mode != "RGBA":
        image = image.convert("RGBA")

    watermark_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(watermark_layer)

    # Auto-scale font size
    font_size = max(24, image.size[0] // 30)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()

    # Use textbbox instead of textsize
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    padding = font_size // 2
    position = (image.size[0] - text_w - padding, image.size[1] - text_h - padding)

    draw.text(position, text, fill=(255, 255, 255, opacity), font=font)

    return Image.alpha_composite(image, watermark_layer).convert("RGB")


def validate_trigger_word(pipe, prompt_text):
    image_token_id = pipe.tokenizer.convert_tokens_to_ids(pipe.trigger_word)
    input_ids = pipe.tokenizer.encode(prompt_text)

    if image_token_id not in input_ids:
        raise ValueError(
            f"Trigger word '{pipe.trigger_word}' missing in prompt: {prompt_text}"
        )

    if input_ids.count(image_token_id) > 1:
        raise ValueError(
            f"Multiple trigger words '{pipe.trigger_word}' found in prompt: {prompt_text}"
        )


def get_device():
    try:
        if torch.cuda.is_available():
            return "cuda"
        elif sys.platform == "darwin" and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    except:
        return "cpu"


def apply_style(style_name, positive, negative=""):
    default_style = "Photographic (Default)"
    p, n = styles.get(style_name, styles[default_style])
    return p.replace("{prompt}", positive), n + ' ' + negative


def load_pipeline(device):
    print("Loading pipeline...")
    
    base_model_path = 'SG161222/RealVisXL_V4.0'
    
    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device == "mps":
        torch_dtype = torch.float16
    
    print("Loading T2I adapter...")
    adapter = T2IAdapter.from_pretrained(
        "TencentARC/t2i-adapter-sketch-sdxl-1.0", 
        torch_dtype=torch_dtype, 
        variant="fp16"
    ).to(device)
    
    print("Loading main pipeline...")
    pipe = PhotoMakerStableDiffusionXLAdapterPipeline.from_pretrained(
        base_model_path, 
        adapter=adapter, 
        torch_dtype=torch_dtype,
        use_safetensors=True, 
        variant="fp16",
    ).to(device)
    
    print("Loading PhotoMaker adapter...")
    photomaker_ckpt = hf_hub_download(
        repo_id="TencentARC/PhotoMaker-V2", 
        filename="photomaker-v2.bin", 
        repo_type="model"
    )
    
    pipe.load_photomaker_adapter(
        os.path.dirname(photomaker_ckpt),
        subfolder="",
        weight_name=os.path.basename(photomaker_ckpt),
        trigger_word="img",
        pm_version="v2",
    )
    pipe.id_encoder.to(device)
    
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.fuse_lora()
    pipe.to(device)
    
    print("Pipeline loaded successfully!")
    return pipe


def load_face_detector(device):
    print("Loading face detector...")
    providers = ['CUDAExecutionProvider'] if device == "cuda" else ['CPUExecutionProvider']
    face_detector = FaceAnalysis2(
        providers=providers, 
        allowed_modules=['detection', 'recognition']
    )
    face_detector.prepare(ctx_id=0, det_size=(640, 640))
    return face_detector


def generate_image(pipe, face_detector, device):
    # -----------------------------
    # Sketch / Adapter handling
    # -----------------------------
    sketch_image = None
    adapter_scale = 0.
    adapter_factor = 0.

    if USE_SKETCH and SKETCH_IMAGE_PATH:
        sketch_image = Image.open(SKETCH_IMAGE_PATH).convert("RGBA")
        r, g, b, a = sketch_image.split()
        sketch_image = a.convert("RGB")
        sketch_image = TF.to_tensor(sketch_image) > 0.5
        sketch_image = TF.to_pil_image(sketch_image.to(torch.float32))
        adapter_scale = ADAPTER_CONDITIONING_SCALE
        adapter_factor = ADAPTER_CONDITIONING_FACTOR

    # -----------------------------
    # Load input images
    # -----------------------------
    if not INPUT_IMAGES:
        raise ValueError("No input images provided.")

    input_id_images = []
    for img_path in INPUT_IMAGES:
        if not os.path.exists(img_path):
            raise ValueError(f"Image not found: {img_path}")
        input_id_images.append(load_image(img_path))

    # -----------------------------
    # Detect faces & extract embeddings
    # -----------------------------
    img = input_id_images[0]
    img_array = np.array(img)[:, :, ::-1]

    faces = analyze_faces(face_detector, img_array)
    if len(faces) < 2:
        raise ValueError("Need at least two faces in the input image.")

    # Sort left → right
    faces_sorted = sorted(faces, key=lambda f: f['bbox'][0])
    left_face = faces_sorted[0]
    right_face = faces_sorted[1]

    emb_left = torch.from_numpy(left_face['embedding']).float()
    emb_right = torch.from_numpy(right_face['embedding']).float()

    id_embeds_face_left = torch.stack([emb_left])
    id_embeds_face_right = torch.stack([emb_right])

    # -----------------------------
    # Output dimensions
    # -----------------------------
    output_w = OUTPUT_WIDTH or 1024
    output_h = OUTPUT_HEIGHT or 1024

    # -----------------------------
    # Seed & generator
    # -----------------------------
    seed = SEED if SEED is not None else random.randint(0, MAX_SEED)
    generator = torch.Generator(device=device).manual_seed(seed)

    # -----------------------------
    # Merge step
    # -----------------------------
    start_merge_step = int(float(STYLE_STRENGTH_RATIO) / 100 * NUM_STEPS)
    if start_merge_step > 30:
        start_merge_step = 30

    # -----------------------------
    # Trigger word validator
    # -----------------------------
    def validate_trigger_word(prompt_text):
        image_token_id = pipe.tokenizer.convert_tokens_to_ids(pipe.trigger_word)
        input_ids = pipe.tokenizer.encode(prompt_text)

        if image_token_id not in input_ids:
            raise ValueError(f"Trigger word '{pipe.trigger_word}' missing in prompt: {prompt_text}")

        if input_ids.count(image_token_id) > 1:
            raise ValueError(f"Multiple trigger words '{pipe.trigger_word}' found in prompt: {prompt_text}")

    # -----------------------------
    # Multi‑prompt generation
    # -----------------------------
    images_face_left_all = []
    images_face_right_all = []

    # LEFT FACE
    for prompt_text in PROMPTS_FACE_LEFT:
        validate_trigger_word(prompt_text)
        prompt, negative_prompt = apply_style(STYLE_NAME, prompt_text, NEGATIVE_PROMPT)

        imgs = pipe(
            prompt=prompt,
            width=output_w,
            height=output_h,
            input_id_images=input_id_images,
            negative_prompt=negative_prompt,
            num_images_per_prompt=NUM_OUTPUTS,
            num_inference_steps=NUM_STEPS,
            start_merge_step=start_merge_step,
            generator=generator,
            guidance_scale=GUIDANCE_SCALE,
            id_embeds=id_embeds_face_left,
            image=sketch_image,
            adapter_conditioning_scale=adapter_scale,
            adapter_conditioning_factor=adapter_factor,
        ).images

        images_face_left_all.append((prompt_text, imgs))

    # RIGHT FACE
    for prompt_text in PROMPTS_FACE_RIGHT:
        validate_trigger_word(prompt_text)
        prompt, negative_prompt = apply_style(STYLE_NAME, prompt_text, NEGATIVE_PROMPT)

        imgs = pipe(
            prompt=prompt,
            width=output_w,
            height=output_h,
            input_id_images=input_id_images,
            negative_prompt=negative_prompt,
            num_images_per_prompt=NUM_OUTPUTS,
            num_inference_steps=NUM_STEPS,
            start_merge_step=start_merge_step,
            generator=generator,
            guidance_scale=GUIDANCE_SCALE,
            id_embeds=id_embeds_face_right,
            image=sketch_image,
            adapter_conditioning_scale=adapter_scale,
            adapter_conditioning_factor=adapter_factor,
        ).images

        images_face_right_all.append((prompt_text, imgs))

    return images_face_left_all, images_face_right_all, seed



def main():
    print("=" * 50)
    print("PhotoMaker V2 CLI")
    print("=" * 50)
    
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = get_device()
    print(f"Using device: {device}")
    
    pipe = load_pipeline(device)
    face_detector = load_face_detector(device)
    
    try:
        # IMPORTANT: generate_image must now return:
        # images_face0, images_face1, used_seed

        images_face_left, images_face_right, used_seed = generate_image(pipe, face_detector, device)
        print(f"\nSaving outputs to {output_dir}/")

        # Save LEFT FACE outputs
        for prompt_text, imgs in images_face_left:
            safe_prompt = prompt_text.replace(" ", "_").replace(",", "")
            for i, img in enumerate(imgs):
                img = add_watermark(img)
                filename = f"left_{safe_prompt}_seed{used_seed}_{i+1}.png"
                img.save(output_dir / filename)
                print(f"Saved: {filename}")

        # Save RIGHT FACE outputs
        for prompt_text, imgs in images_face_right:
            safe_prompt = prompt_text.replace(" ", "_").replace(",", "")
            for i, img in enumerate(imgs):
                img = add_watermark(img)
                filename = f"right_{safe_prompt}_seed{used_seed}_{i+1}.png"
                img.save(output_dir / filename)
                print(f"Saved: {filename}")

        
        print(f"\nDone! Generated {len(images_face_left)} image(s) for face_left and {len(images_face_right)} image(s) for face_right with seed {used_seed}")
        
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
