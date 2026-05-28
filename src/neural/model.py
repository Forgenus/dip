import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.block(batch)


class PairClassifier(nn.Module):
    def __init__(self, input_channels: int = 2) -> None:
        super().__init__()
        if input_channels <= 0:
            raise ValueError("input_channels must be a positive integer")

        self.features = nn.Sequential(
            ConvBlock(input_channels, 32),
            ConvBlock(32, 32),
            nn.MaxPool2d(2),
            ConvBlock(32, 64),
            ConvBlock(64, 64),
            nn.MaxPool2d(2),
            ConvBlock(64, 128),
            ConvBlock(128, 128),
            nn.MaxPool2d(2),
            ConvBlock(128, 192),
            ConvBlock(192, 192),
            nn.MaxPool2d(2),
            ConvBlock(192, 256),
            ConvBlock(256, 256),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.SiLU(inplace=True),
            nn.Linear(256, 64),
            nn.SiLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        values = self.features(batch)
        avg_pool = values.mean(dim=(-2, -1))
        max_pool = values.amax(dim=(-2, -1))
        pooled = torch.cat((avg_pool, max_pool), dim=1)
        logits = self.classifier(pooled)
        return logits.squeeze(dim=1)
