import torch


def monitor_cluster_usage(pred):
    """
    Monitors the percentage of pixels assigned to each cluster.
    Args:
        pred: Softmax probabilities of shape (B, C, H, W)
    """
    cluster_assignments = torch.argmax(pred, dim=1).view(-1)  # Flatten spatial dimensions
    unique_clusters, counts = torch.unique(cluster_assignments, return_counts=True)
    total_pixels = pred.shape[0] * pred.shape[2] * pred.shape[3]

    usage = dict(zip(unique_clusters.cpu().numpy(), (counts.cpu().numpy() / total_pixels) * 100))
    print(f"Cluster usage: {usage}%")
