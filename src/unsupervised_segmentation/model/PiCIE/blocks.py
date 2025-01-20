import torch
import torch.nn as nn
import torchvision.models as models
from sklearn.cluster import MiniBatchKMeans


class PiCIEBackbone(nn.Module):
    def __init__(self, pretrained=True, out_channels=256):
        super(PiCIEBackbone, self).__init__()
        # Load a ResNet model as the backbone
        resnet = models.resnet50(pretrained=pretrained)

        # Extract layers from ResNet
        self.layer1 = nn.Sequential(*list(resnet.children())[:5])  # Conv1 to Layer1
        self.layer2 = nn.Sequential(*list(resnet.children())[5:6])  # Layer2
        self.layer3 = nn.Sequential(*list(resnet.children())[6:7])  # Layer3
        self.layer4 = nn.Sequential(*list(resnet.children())[7:8])  # Layer4

        # Add 1x1 convolutions for feature reduction
        self.conv1 = nn.Conv2d(256, out_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(512, out_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(1024, out_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(2048, out_channels, kernel_size=1)

    def forward(self, x):
        # Forward pass through the backbone layers
        feat1 = self.layer1(x)
        feat2 = self.layer2(feat1)
        feat3 = self.layer3(feat2)
        feat4 = self.layer4(feat3)

        # Apply 1x1 convolutions to reduce feature dimensions
        feat1 = self.conv1(feat1)
        feat2 = self.conv2(feat2)
        feat3 = self.conv3(feat3)
        feat4 = self.conv4(feat4)

        return [feat1, feat2, feat3, feat4]


class ClusteringModule:
    def __init__(self, num_clusters=20, batch_size=256):
        self.num_clusters = num_clusters
        self.kmeans = MiniBatchKMeans(n_clusters=num_clusters, batch_size=batch_size, random_state=42)

    def fit(self, features):
        """
        Fit k-means to the provided features.
        Args:
            features (torch.Tensor): Extracted features of shape (N, C, H, W).
        """
        # Flatten features to (N*H*W, C)
        n, c, h, w = features.shape
        features = features.permute(0, 2, 3, 1).reshape(-1, c)

        # Fit k-means
        self.kmeans.partial_fit(features.cpu().numpy())

    def predict(self, features):
        """
        Predict cluster assignments for the given features.
        Args:
            features (torch.Tensor): Input features of shape (N, C, H, W).
        Returns:
            torch.Tensor: Cluster assignments of shape (N, H, W).
        """
        n, c, h, w = features.shape
        features = features.permute(0, 2, 3, 1).reshape(-1, c)
        labels = self.kmeans.predict(features.cpu().numpy())
        return torch.tensor(labels, dtype=torch.long).view(n, h, w)
