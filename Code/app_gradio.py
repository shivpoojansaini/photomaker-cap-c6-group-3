import gradio as gr
from pathlib import Path
import glob
import shutil

from PhotoMaker_Extensions.cli import main as run_photomaker
from PhotoMaker_Extensions import config
from PhotoMaker_Extensions.invisible_watermark.utils import decode_watermark
from PIL import Image


def get_or_save_input_image(uploaded_file):
    input_dir = Path("/teamspace/studios/this_studio/PhotoMaker-CAP-C6-Group_3/Data/Input")
    input_dir.mkdir(parents=True, exist_ok=True)

    saved_path = input_dir / "uploaded_input_image.png"

    if uploaded_file is not None:
        shutil.copy(uploaded_file, saved_path)
        return str(saved_path)

    existing = list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg"))
    return str(existing[0]) if existing else None


def verify_watermark(uploaded_image):
    if uploaded_image is None:
        return "Please upload an image first."

    try:
        # uploaded_image is a file path → load it
        img = Image.open(uploaded_image)

        bits = decode_watermark(img)
        bitstring = "".join(str(b) for b in bits)

        confidence = sum(bits) / len(bits)

        return (
            f"Recovered watermark bits:\n{bitstring}\n\n"
            f"Confidence: {confidence:.2f}\n"
            f"Status: {'Watermark FOUND' if confidence > 0.6 else 'Watermark NOT detected'}"
        )
    except Exception as e:
        return f"Error during verification: {e}"



def get_existing_input_image():
    input_dir = Path("/teamspace/studios/this_studio/PhotoMaker-CAP-C6-Group_3/Data/Input")
    existing = list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg"))
    return str(existing[0]) if existing else None


def generate_images(uploaded_image, left_prompt, right_prompt, seed_value, watermark_mode):
    image_path = get_or_save_input_image(uploaded_image)
    if image_path is None:
        return "No input image found. Please upload one.", [], []

    try:
        seed = int(seed_value) if seed_value else None
    except:
        seed = None

    run_photomaker(
        input_image=image_path,
        left_prompt=left_prompt,
        right_prompt=right_prompt,
        seed=seed,
        watermark=(watermark_mode == "With Watermark")
    )

    out_dir = Path(config.OUTPUT_DIR)
    left_imgs = sorted(glob.glob(str(out_dir / "left_*.png")))
    right_imgs = sorted(glob.glob(str(out_dir / "right_*.png")))

    return "Generation complete.", left_imgs, right_imgs



def build_ui():
    with gr.Blocks() as demo:
        gr.Markdown("## 🎨 PhotoMaker V2 — Gradio UI (Dynamic Prompts + Auto‑Load Input Image)")

        with gr.Row():
            with gr.Column(scale=1):
                uploaded_image = gr.Image(
                    label="Upload Input Image (optional)",
                    type="filepath",
                    value=get_existing_input_image()
                )

                left_prompt = gr.Textbox(label="Left Face Prompt")
                right_prompt = gr.Textbox(label="Right Face Prompt")
                seed = gr.Textbox(label="Seed (optional)")

                watermark_mode = gr.Radio(
                    choices=["With Watermark", "Without Watermark"],
                    value="With Watermark",
                    label="Watermark Mode"
                )

                generate_btn = gr.Button("Generate Images")
                verify_button = gr.Button("Verify Watermark")
                verify_output = gr.Textbox(label="Verification Result")


            with gr.Column(scale=2):
                status = gr.Textbox(label="Status")
                left_gallery = gr.Gallery(label="Left Face Results", columns=2)
                right_gallery = gr.Gallery(label="Right Face Results", columns=2)

        # Generate button wiring
        generate_btn.click(
            fn=generate_images,
            inputs=[uploaded_image, left_prompt, right_prompt, seed, watermark_mode],
            outputs=[status, left_gallery, right_gallery],
        )


        # Verify watermark button wiring
        verify_button.click(
            fn=verify_watermark,
            inputs=[uploaded_image],
            outputs=[verify_output]
        )

    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(
        share=True,
        allowed_paths=[
            "/teamspace/studios/this_studio/PhotoMaker-CAP-C6-Group_3/Data/Output",
            "/teamspace/studios/this_studio/PhotoMaker-CAP-C6-Group_3/Data/Input"
        ]
    )
