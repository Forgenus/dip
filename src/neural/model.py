import torch
import torch.nn as nn


class PairClassifier(nn.Module):
    def __init__(self, input_channels: int = 3) -> None:
        super().__init__()
        if input_channels <= 0:
            raise ValueError("input_channels must be positive")

        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        values = self.features(batch)
        probabilities = self.classifier(values)
        return probabilities.squeeze(dim=1)
