from __future__ import annotations

import torch.nn as nn


# CNN compacta: menos parametros para reducir el riesgo de sobreajuste.
class CNN2DAdaptive(nn.Module):
    def __init__(
        self,
        num_classes: int,
        channels: tuple[int, ...] = (8, 16, 32, 32),
        dropout: float = 0.35,
        kernel_size: int | tuple[int, int] = 3,
        pool_size: tuple[int, int] = (2, 2),
        hidden_dim: int = 32,
    ):
        super().__init__()
        layers = []
        in_channels = 1
        for out_channels in channels:
            layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size, padding="same"),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                ]
            )
            in_channels = out_channels

        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(pool_size)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels[-1] * pool_size[0] * pool_size[1], hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


class MLPHuBERT(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.45, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.mean(dim=1)
        return self.net(x)



def crear_cnn_mel(num_classes: int, dropout: float = 0.35) -> CNN2DAdaptive:
    return CNN2DAdaptive(
        num_classes=num_classes,
        channels=(8, 16, 32, 32),
        dropout=dropout,
        kernel_size=3,
        pool_size=(2, 2),
        hidden_dim=32,
    )


def crear_cnn_wavelet(num_classes: int, dropout: float = 0.45) -> CNN2DAdaptive:
    return CNN2DAdaptive(
        num_classes=num_classes,
        channels=(16, 32, 64, 64),
        dropout=dropout,
        kernel_size=(5, 3),
        pool_size=(2, 2),
        hidden_dim=64,
    )