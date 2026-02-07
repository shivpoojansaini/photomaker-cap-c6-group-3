# PhotoMaker_Extensions/invisible_watermark/decoder.py

import torch
import torch.nn as nn

class WatermarkDecoder(nn.Module):
    def __init__(self, bit_length=64):
        super().__init__()
        self.bit_length = bit_length

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

        self.fc = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, bit_length),
            nn.Sigmoid()
        )

    def forward(self, image):
        x = self.conv(image)
        x = x.view(x.size(0), -1)
        return self.fc(x)
