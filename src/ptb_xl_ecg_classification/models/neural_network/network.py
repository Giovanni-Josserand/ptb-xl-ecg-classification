import torch
from torch import nn

from ptb_xl_ecg_classification.models.neural_network.residual_block import ResidualBlock1D


class ResNet1D(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels=12,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
        )
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = nn.Sequential(
            ResidualBlock1D(64, 64),
            ResidualBlock1D(64, 64),
        )

        self.layer2 = nn.Sequential(
            ResidualBlock1D(64, 128, stride=2),
            ResidualBlock1D(128, 128),
        )

        self.layer3 = nn.Sequential(
            ResidualBlock1D(128, 256, stride=2),
            ResidualBlock1D(256, 256),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, 5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)

        return x