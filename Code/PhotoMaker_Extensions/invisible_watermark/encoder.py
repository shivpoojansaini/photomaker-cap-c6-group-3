import torch
import torch.nn as nn
import torch.nn.functional as F

class WatermarkEncoder(nn.Module):
    def __init__(self, bit_length=64):
        super().__init__()
        self.bit_length = bit_length

        # 256 feature maps of size 16×16
        self.bit_fc = nn.Linear(bit_length, 256 * 16 * 16)

        self.conv = nn.Sequential(
            nn.Conv2d(3 + 256, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 1),
        )

    def forward(self, image, bits):
        B, C, H, W = image.shape

        # Correct reshape
        x = self.bit_fc(bits).view(B, 256, 16, 16)

        # Upsample to image size
        x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)

        # Concatenate bit features with image
        x = torch.cat([image, x], dim=1)

        # Predict residual
        residual = self.conv(x)

        return torch.clamp(image + 0.01 * residual, 0.0, 1.0)


