import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .encoder import WatermarkEncoder
from .decoder import WatermarkDecoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BIT_LENGTH = 64

def apply_attacks(img):
    noise = 0.01 * torch.randn_like(img)
    img = torch.clamp(img + noise, 0, 1)
    return img

def main():
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    encoder = WatermarkEncoder(bit_length=BIT_LENGTH).to(DEVICE)
    decoder = WatermarkDecoder(bit_length=BIT_LENGTH).to(DEVICE)

    opt = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-4)

    lambda_img = 1.0
    lambda_wm = 5.0

    for epoch in range(5):
        for imgs, _ in loader:
            imgs = imgs.to(DEVICE)

            bits = torch.randint(0, 2, (imgs.size(0), BIT_LENGTH), device=DEVICE).float()

            watermarked = encoder(imgs, bits)
            attacked = apply_attacks(watermarked)
            pred_bits = decoder(attacked)

            img_loss = F.mse_loss(watermarked, imgs)
            wm_loss = F.binary_cross_entropy(pred_bits, bits)

            loss = lambda_img * img_loss + lambda_wm * wm_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

        print(f"Epoch {epoch}: img_loss={img_loss.item():.4f}, wm_loss={wm_loss.item():.4f}")

    torch.save(encoder.state_dict(), "encoder_trained.pth")
    torch.save(decoder.state_dict(), "decoder_trained.pth")

if __name__ == "__main__":
    main()
