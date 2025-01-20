import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleIICNet(nn.Module):
    def __init__(self, input_channels=2, num_clusters=10):
        super(SimpleIICNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.cluster_head = nn.Conv2d(256, num_clusters, kernel_size=1)

    def forward(self, x):
        features = self.encoder(x)  # Output: (B, 256, H, W)
        clusters = self.cluster_head(features)  # Output: (B, num_clusters, H, W)
        return clusters


import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut layer
        self.shortcut = nn.Sequential() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += shortcut
        x = self.relu(x)
        return x


class ImprovedIICNet(nn.Module):
    def __init__(self, input_channels=2, num_clusters=10, use_gap=False):
        super(ImprovedIICNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2),
            ResidualBlock(128, 256),
            ResidualBlock(256, 256),
            nn.MaxPool2d(2, 2),
        )

        self.cluster_head = nn.Conv2d(256, num_clusters, kernel_size=1)

        # Optional global average pooling for reduced feature maps
        self.use_gap = use_gap
        if use_gap:
            self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        features = self.encoder(x)  # Output: (B, 256, H, W)
        if self.use_gap:
            features = self.gap(features)  # Output: (B, 256, 1, 1)
            features = features.view(features.size(0), -1)  # Flatten to (B, 256)
        clusters = self.cluster_head(features)  # Output: (B, num_clusters, H, W)
        return clusters
