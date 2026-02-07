from pathlib import Path
from .pipeline_loader import load_pipeline, get_device
from .face_utils import load_face_detector
from .generation import generate_images
from .watermark import add_watermark
from .config import (
    OUTPUT_DIR,
    STYLE_NAME,
    NEGATIVE_PROMPT,
    OUTPUT_WIDTH,
    OUTPUT_HEIGHT,
    NUM_OUTPUTS,
    NUM_STEPS,
    STYLE_STRENGTH_RATIO,
    GUIDANCE_SCALE,
)
from .invisible_watermark.utils import encode_watermark


# 64-bit watermark payload (example)
bitstring = [1,0,1,1,0,1,0,1] * 8


def main(input_image, left_prompt, right_prompt, seed=None, watermark=True):
    print("=" * 50)
    print("PhotoMaker V2 CLI")
    print("=" * 50)

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Using device: {device}")

    pipe = load_pipeline(device)
    face_detector = load_face_detector(device)

    left_imgs, right_imgs, seed = generate_images(
        pipe=pipe,
        face_detector=face_detector,
        input_image_path=input_image,
        left_prompt=left_prompt,
        right_prompt=right_prompt,
        seed=seed,
        style_name=STYLE_NAME,
        negative_prompt=NEGATIVE_PROMPT,
        width=OUTPUT_WIDTH,
        height=OUTPUT_HEIGHT,
        num_outputs=NUM_OUTPUTS,
        num_steps=NUM_STEPS,
        style_strength_ratio=STYLE_STRENGTH_RATIO,
        guidance_scale=GUIDANCE_SCALE,
    )

    print(f"\nSaving outputs to {output_dir}/")

    # LEFT FACE
    for prompt_text, imgs in left_imgs:
        safe = prompt_text.replace(" ", "_").replace(",", "")
        for i, img in enumerate(imgs):

            if watermark:
                # Invisible watermark
                img = encode_watermark(img, bitstring)
                # Visible watermark
                img = add_watermark(img)
            else:
                print("Skipping watermark for LEFT image")

            filename = f"left_{safe}_seed{seed}_{i+1}.png"
            img.save(output_dir / filename)
            print(f"Saved: {filename}")

    # RIGHT FACE
    for prompt_text, imgs in right_imgs:
        safe = prompt_text.replace(" ", "_").replace(",", "")
        for i, img in enumerate(imgs):

            if watermark:
                # Invisible watermark
                img = encode_watermark(img, bitstring)
                # Visible watermark
                img = add_watermark(img)
            else:
                print("Skipping watermark for RIGHT image")

            filename = f"right_{safe}_seed{seed}_{i+1}.png"
            img.save(output_dir / filename)
            print(f"Saved: {filename}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--input_image", type=str, required=True)
    parser.add_argument("--left_prompt", type=str, required=True)
    parser.add_argument("--right_prompt", type=str, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--watermark", action="store_true")

    args = parser.parse_args()

    main(
        input_image=args.input_image,
        left_prompt=args.left_prompt,
        right_prompt=args.right_prompt,
        seed=args.seed,
        watermark=args.watermark,
    )
