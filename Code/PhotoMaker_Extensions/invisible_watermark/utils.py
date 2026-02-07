# PhotoMaker_Extensions/invisible_watermark/utils.py

import torch
from pathlib import Path
from .encoder import WatermarkEncoder
from .decoder import WatermarkDecoder


BIT_LENGTH = 64

encoder = WatermarkEncoder(bit_length=BIT_LENGTH)
decoder = WatermarkDecoder(bit_length=BIT_LENGTH)

# Try loading trained weights
try:
    encoder.load_state_dict(torch.load(
        Path(__file__).parent / "encoder_best.pth",
        map_location="cpu"
    ))
    decoder.load_state_dict(torch.load(
        Path(__file__).parent / "decoder_best.pth",
        map_location="cpu"
    ))
    print("Loaded trained invisible watermark models.")
except FileNotFoundError:
    print("Warning: Trained watermark models not found. Using untrained models.")

encoder.eval()
decoder.eval()



from PIL import Image
import torchvision.transforms as T

to_tensor = T.ToTensor()
to_pil = T.ToPILImage()

def encode_watermark(pil_img, bitstring):
    img = to_tensor(pil_img).unsqueeze(0)
    bits = torch.tensor(bitstring).float().unsqueeze(0)
    wm = encoder(img, bits)
    return to_pil(wm.squeeze(0))

def decode_watermark(pil_img):
    img = to_tensor(pil_img).unsqueeze(0)
    bits = decoder(img).detach().squeeze(0)
    return (bits > 0.5).int().tolist()

