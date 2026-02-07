#!/usr/bin/env python3
"""
Test script for Multi-Identity PhotoMaker Pipeline

This script demonstrates and tests the multi-identity generation capability.
Run with: python test_multi_identity.py

Requirements:
- PhotoMaker model weights
- SDXL base model
- Test images with faces
"""

import argparse
import torch
from PIL import Image
from pathlib import Path

# Test imports first
def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")

    try:
        from photomaker import (
            PhotoMakerMultiIdentityPipeline,
            MultiIdentityPromptParser,
            parse_multi_identity_prompt,
            SpatialMaskGenerator,
            RegionalMasks,
            RegionalAttentionConfig,
            RegionalAttnProcessor2_0,
        )
        print("  [OK] All imports successful")
        return True
    except ImportError as e:
        print(f"  [FAIL] Import error: {e}")
        return False


def test_prompt_parser():
    """Test the prompt parser with various input formats"""
    print("\nTesting prompt parser...")

    from photomaker.prompt_parser import MultiIdentityPromptParser, parse_multi_identity_prompt

    test_cases = [
        # (input_prompt, expected_num_regions)
        ("left person as astronaut, right person as doctor", 2),
        ("the man on the left is a chef, the woman on the right is a scientist", 2),
        ("person1 as astronaut, person2 as doctor", 2),
        ("a man img wearing suit, a woman img in dress", 2),
    ]

    parser = MultiIdentityPromptParser()

    all_passed = True
    for prompt, expected_regions in test_cases:
        result = parser.parse(prompt)

        if len(result.regions) == expected_regions:
            print(f"  [OK] '{prompt[:40]}...' -> {len(result.regions)} regions")
            for r in result.regions:
                print(f"       - {r.region_id}: '{r.description[:30]}...'")
        else:
            print(f"  [FAIL] '{prompt[:40]}...' -> expected {expected_regions}, got {len(result.regions)}")
            all_passed = False

    return all_passed


def test_spatial_masks():
    """Test spatial mask generation"""
    print("\nTesting spatial mask generator...")

    from photomaker.spatial_masks import SpatialMaskGenerator, RegionalMasks

    device = torch.device("cpu")
    dtype = torch.float32

    # Test position-based masks
    generator = SpatialMaskGenerator(
        latent_height=128,
        latent_width=128,
        blur_sigma=8.0
    )

    masks = generator.generate_position_based_masks(
        positions=["left", "right"],
        device=device,
        dtype=dtype
    )

    if masks.masks.shape == (2, 128, 128):
        print(f"  [OK] Position-based masks shape: {masks.masks.shape}")
    else:
        print(f"  [FAIL] Expected (2, 128, 128), got {masks.masks.shape}")
        return False

    # Test face bbox-based masks
    face_bboxes = [
        (100, 100, 200, 250),  # Left face
        (400, 100, 500, 250),  # Right face
    ]

    masks = generator.generate_masks_from_faces(
        face_bboxes=face_bboxes,
        face_positions=["left", "right"],
        image_width=640,
        image_height=480,
        device=device,
        dtype=dtype
    )

    if masks.masks.shape == (2, 128, 128):
        print(f"  [OK] Face-based masks shape: {masks.masks.shape}")
    else:
        print(f"  [FAIL] Expected (2, 128, 128), got {masks.masks.shape}")
        return False

    # Check mask properties
    print(f"  [OK] Left mask range: [{masks.masks[0].min():.2f}, {masks.masks[0].max():.2f}]")
    print(f"  [OK] Right mask range: [{masks.masks[1].min():.2f}, {masks.masks[1].max():.2f}]")

    return True


def test_regional_attention():
    """Test regional attention processor configuration"""
    print("\nTesting regional attention processor...")

    from photomaker.regional_attention import (
        RegionalAttentionConfig,
        RegionalAttnProcessor2_0,
    )

    config = RegionalAttentionConfig(
        num_identities=2,
        tokens_per_identity=2,
        shared_token_count=77,
        enable_regional=True
    )

    processor = RegionalAttnProcessor2_0(config)

    if processor.regional_config == config:
        print(f"  [OK] Regional config set correctly")
        print(f"       - num_identities: {config.num_identities}")
        print(f"       - tokens_per_identity: {config.tokens_per_identity}")
        print(f"       - shared_token_count: {config.shared_token_count}")
        return True
    else:
        print(f"  [FAIL] Regional config not set correctly")
        return False


def test_full_pipeline(
    model_path: str = None,
    photomaker_path: str = None,
    input_image_path: str = None,
    left_image_path: str = None,
    right_image_path: str = None,
    output_path: str = "output_multi_identity.png",
):
    """
    Full pipeline test with actual model loading and generation.

    This test requires:
    - SDXL model weights
    - PhotoMaker adapter weights
    - Test images
    """
    print("\nTesting full pipeline...")

    if model_path is None:
        print("  [SKIP] No model path provided. Use --model_path to run full test.")
        return True

    try:
        from photomaker import PhotoMakerMultiIdentityPipeline
        from diffusers import EulerDiscreteScheduler

        print(f"  Loading model from {model_path}...")

        # Load pipeline
        pipe = PhotoMakerMultiIdentityPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            use_safetensors=True,
        ).to("cuda")

        # Load scheduler
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

        # Load PhotoMaker adapter
        if photomaker_path:
            print(f"  Loading PhotoMaker adapter from {photomaker_path}...")
            pipe.load_photomaker_adapter(
                photomaker_path,
                weight_name="photomaker-v2.bin",
                trigger_word="img",
            )

        # Setup face detector
        print("  Setting up face detector...")
        pipe.setup_face_detector(device="cuda")

        # Option 1: Single image with 2 people (SIMPLER)
        if input_image_path:
            print(f"  Using single image mode: {input_image_path}")
            input_image = Image.open(input_image_path).convert("RGB")

            result = pipe.generate_from_single_image(
                input_image=input_image,
                prompt="left person as astronaut in space suit, right person as doctor in white coat",
                num_inference_steps=30,
                guidance_scale=5.0,
                start_merge_step=10,
            )

            result.images[0].save(output_path)
            print(f"  [OK] Generated image saved to {output_path}")
            return True

        # Option 2: Two separate images
        if left_image_path and right_image_path:
            print(f"  Using two-image mode: {left_image_path}, {right_image_path}")
            left_images = [Image.open(left_image_path).convert("RGB")]
            right_images = [Image.open(right_image_path).convert("RGB")]

            result = pipe(
                prompt="left person as astronaut in space suit, right person as doctor in white coat",
                identity_images=[left_images, right_images],
                regional_positions=["left", "right"],
                num_inference_steps=30,
                guidance_scale=5.0,
                start_merge_step=10,
            )

            result.images[0].save(output_path)
            print(f"  [OK] Generated image saved to {output_path}")
            return True

        print("  [SKIP] No test images provided. Use --input_image OR --left_image and --right_image.")
        return True

    except Exception as e:
        print(f"  [FAIL] Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Test Multi-Identity PhotoMaker Pipeline")
    parser.add_argument("--model_path", type=str, help="Path to SDXL model")
    parser.add_argument("--photomaker_path", type=str, help="Path to PhotoMaker adapter")
    parser.add_argument("--input_image", type=str, help="Path to single image with 2 people (RECOMMENDED)")
    parser.add_argument("--left_image", type=str, help="Path to left person's image (alternative)")
    parser.add_argument("--right_image", type=str, help="Path to right person's image (alternative)")
    parser.add_argument("--output", type=str, default="output_multi_identity.png", help="Output path")
    parser.add_argument("--full_test", action="store_true", help="Run full pipeline test")

    args = parser.parse_args()

    print("=" * 60)
    print("Multi-Identity PhotoMaker Pipeline Tests")
    print("=" * 60)

    results = []

    # Run unit tests
    results.append(("Imports", test_imports()))
    results.append(("Prompt Parser", test_prompt_parser()))
    results.append(("Spatial Masks", test_spatial_masks()))
    results.append(("Regional Attention", test_regional_attention()))

    # Run full pipeline test if requested
    if args.full_test or args.model_path:
        results.append(("Full Pipeline", test_full_pipeline(
            model_path=args.model_path,
            photomaker_path=args.photomaker_path,
            input_image_path=args.input_image,
            left_image_path=args.left_image,
            right_image_path=args.right_image,
            output_path=args.output,
        )))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {name}")
        if not passed:
            all_passed = False

    print("\n" + ("All tests passed!" if all_passed else "Some tests failed."))
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
