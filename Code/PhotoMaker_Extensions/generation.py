# generation.py

import random
import numpy as np
import torch
from diffusers.utils import load_image
from .style_template import styles
from .face_utils import extract_left_right_embeddings

MAX_SEED = np.iinfo(np.int32).max


def validate_trigger_word(pipe, prompt):
    token_id = pipe.tokenizer.convert_tokens_to_ids(pipe.trigger_word)
    ids = pipe.tokenizer.encode(prompt)

    if token_id not in ids:
        raise ValueError(f"Trigger word '{pipe.trigger_word}' missing in prompt: {prompt}")

    if ids.count(token_id) > 1:
        raise ValueError(f"Multiple trigger words '{pipe.trigger_word}' found in prompt: {prompt}")


def apply_style(style_name, positive, negative):
    default = "Photographic (Default)"
    p, n = styles.get(style_name, styles[default])
    return p.replace("{prompt}", positive), n + " " + negative


def generate_images(
    pipe,
    face_detector,
    input_image_path,
    left_prompt,
    right_prompt,
    seed,
    style_name,
    negative_prompt,
    width,
    height,
    num_outputs,
    num_steps,
    style_strength_ratio,
    guidance_scale
):
    # Load input image
    input_image = load_image(input_image_path)

    # Extract identity embeddings
    id_left, id_right = extract_left_right_embeddings(face_detector, input_image)

    # Seed
    seed = seed if seed is not None else random.randint(0, MAX_SEED)
    generator = torch.Generator(device=pipe.device).manual_seed(seed)

    # Merge step
    start_merge_step = int(float(style_strength_ratio) / 100 * num_steps)
    start_merge_step = min(start_merge_step, 30)

    left_results = []
    right_results = []

    # LEFT FACE
    validate_trigger_word(pipe, left_prompt)
    prompt_left, neg_left = apply_style(style_name, left_prompt, negative_prompt)

    imgs_left = pipe(
        prompt=prompt_left,
        width=width,
        height=height,
        input_id_images=[input_image],
        negative_prompt=neg_left,
        num_images_per_prompt=num_outputs,
        num_inference_steps=num_steps,
        start_merge_step=start_merge_step,
        generator=generator,
        guidance_scale=guidance_scale,
        id_embeds=id_left,
    ).images

    left_results.append((left_prompt, imgs_left))

    # RIGHT FACE
    validate_trigger_word(pipe, right_prompt)
    prompt_right, neg_right = apply_style(style_name, right_prompt, negative_prompt)

    imgs_right = pipe(
        prompt=prompt_right,
        width=width,
        height=height,
        input_id_images=[input_image],
        negative_prompt=neg_right,
        num_images_per_prompt=num_outputs,
        num_inference_steps=num_steps,
        start_merge_step=start_merge_step,
        generator=generator,
        guidance_scale=guidance_scale,
        id_embeds=id_right,
    ).images

    right_results.append((right_prompt, imgs_right))

    return left_results, right_results, seed
