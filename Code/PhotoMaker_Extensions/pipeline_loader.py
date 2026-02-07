# pipeline_loader.py

import torch
import os
from diffusers import EulerDiscreteScheduler, T2IAdapter
from huggingface_hub import hf_hub_download
from photomaker import PhotoMakerStableDiffusionXLAdapterPipeline


def get_device():
    try:
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    except:
        return "cpu"


def load_pipeline(device):
    print("Loading pipeline...")

    base_model_path = "SG161222/RealVisXL_V4.0"

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
    ckpt = hf_hub_download(
        repo_id="TencentARC/PhotoMaker-V2",
        filename="photomaker-v2.bin",
        repo_type="model"
    )

    pipe.load_photomaker_adapter(
        os.path.dirname(ckpt),
        subfolder="",
        weight_name=os.path.basename(ckpt),
        trigger_word="img",
        pm_version="v2",
    )

    pipe.id_encoder.to(device)
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.fuse_lora()
    pipe.to(device)

    print("Pipeline loaded successfully!")
    return pipe
