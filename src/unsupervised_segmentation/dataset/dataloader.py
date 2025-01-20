import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
from PIL import Image


# Import additional transforms
from torchvision.transforms import RandomApply, RandomRotation, GaussianBlur


from torchvision.transforms import RandomApply, RandomRotation, ColorJitter, GaussianBlur
import torchvision.transforms.functional as TF


class MultispectralDataset(Dataset):
    def __init__(self, images, transform=None, nr_channels: int = 2):
        self.images = images
        self.nr_channels = nr_channels

        # Base transform: Geometric augmentations applied once (same for both views)
        self.base_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ]
        )

        # Post transform: Appearance augmentations applied independently
        self.post_transform = transforms.Compose(
            [
                RandomApply([transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0))], p=0.5),
                RandomApply([AdditiveGaussianNoise(mean=0.0, std=0.01)], p=0.5),
                RandomApply([SpeckleNoise(mean=0.0, std=0.01)], p=0.5),
            ]
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx].clone().detach().float()

        # Apply base transform once (same for both views)
        transformed_img = self.base_transform(img)

        # Apply post transform independently to create two views
        img1 = self.post_transform(transformed_img)
        img2 = self.post_transform(transformed_img)

        return img1, img2


def log_and_normalize(img, max_value=65535.0):
    """
    Applies logarithmic scaling and normalizes the image globally based on a fixed range.
    Args:
        img (torch.Tensor): Input SAR image tensor.
        max_value (float): Maximum possible value in the SAR image (default: 65535 for 16-bit images).
    Returns:
        torch.Tensor: Log-scaled and normalized image.
    """
    log_img = torch.log1p(img)  # Apply log(1 + x)
    normalized_img = log_img / torch.log1p(torch.tensor(max_value))  # Normalize to [0, 1] globally
    return normalized_img


def add_random_band_dropout(img, p=0.2):
    """
    Randomly drops a band with probability `p`.
    Args:
        img (torch.Tensor): Input SAR image tensor of shape (C, H, W).
        p (float): Probability of dropping a band.
    Returns:
        torch.Tensor: Image with randomly dropped bands.
    """
    if torch.rand(1).item() < p:
        channels = img.shape[0]
        drop_idx = torch.randint(0, channels, (1,)).item()
        img[drop_idx, :, :] = 0.0  # Set the selected band to zero
    return img


class AdditiveGaussianNoise:
    """Applies additive Gaussian noise to the image."""

    def __init__(self, mean=0.0, std=0.01):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        noise = torch.randn_like(img) * self.std + self.mean
        return img + noise


class SpeckleNoise:
    """Applies multiplicative speckle noise to the image."""

    def __init__(self, mean=0.0, std=0.01):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        noise = torch.randn_like(img) * self.std + self.mean
        return img + img * noise


class MultispectralDataset2(Dataset):
    def __init__(self, images, transform=None, nr_channels: int = 2):
        self.images = images
        self.nr_channels = nr_channels

        # Base transform: Geometric augmentations applied once
        self.base_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ]
        )

        # Post transform: Appearance augmentations applied independently
        self.post_transform = transforms.Compose(
            [
                RandomApply([GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0))], p=0.5),
                # transforms.Lambda(lambda x: log_and_normalize(x * 65535)),
            ]
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        img = np.clip(img, a_min=0, a_max=1048)  # Clip pixel values
        img = img / 1048.0  # Normalize to [0, 1] range

        img = torch.tensor(img, dtype=torch.float32)  # Convert to PyTorch tensor

        # Apply base transform once
        transformed_img = self.base_transform(img)

        # Apply post transform independently to create two views
        img1 = self.post_transform(transformed_img)
        img2 = self.post_transform(transformed_img)

        return img1, img2


def log_and_normalize(img, max_value=65535.0):
    """
    Applies logarithmic scaling and normalizes the image globally based on a fixed range.
    Args:
        img (torch.Tensor): Input SAR image tensor.
        max_value (float): Maximum possible value in the SAR image (default: 65535 for 16-bit images).
    Returns:
        torch.Tensor: Log-scaled and normalized image.
    """
    log_img = torch.log1p(img)  # Apply log(1 + x)
    normalized_img = log_img / torch.log1p(torch.tensor(max_value))  # Normalize to [0, 1] globally
    return normalized_img
