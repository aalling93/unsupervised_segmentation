import torch


def compute_blockwise_joint_distribution(pred1, pred2, block_size=64):
    """
    Computes the joint distribution between two predictions using block-wise processing.
    Args:
        pred1: Softmax probabilities of shape (B, C, H, W)
        pred2: Softmax probabilities of shape (B, C, H, W)
        block_size: Size of the block to process at a time
    Returns:
        Joint distribution matrix of shape (C, C)
    """
    B, C, H, W = pred1.shape
    joint = torch.zeros(C, C).cuda()  # Initialize joint distribution on GPU

    for h in range(0, H, block_size):
        for w in range(0, W, block_size):
            block_pred1 = pred1[:, :, h : h + block_size, w : w + block_size]  # (B, C, block_H, block_W)
            block_pred2 = pred2[:, :, h : h + block_size, w : w + block_size]  # (B, C, block_H, block_W)

            # Sum over spatial dimensions (block_H * block_W) to get (B, C)
            block_pred1_mean = block_pred1.sum(dim=(2, 3))  # (B, C)
            block_pred2_mean = block_pred2.sum(dim=(2, 3))  # (B, C)

            # Compute joint distribution by multiplying the means and summing over the batch
            joint_block = torch.einsum("bc,bd->cd", block_pred1_mean, block_pred2_mean)  # (C, C)
            joint += joint_block

    joint /= joint.sum()  # Normalize to get probability distribution
    return joint


def sinkhorn_knopp(joint, num_iters=3, epsilon=1e-10, reg=1e-3):
    """
    Applies Sinkhorn-Knopp normalization with added regularization for stability.
    Args:
        joint: Joint distribution matrix of shape (C, C)
        num_iters: Number of normalization iterations
        epsilon: Small constant to avoid division by zero
        reg: Regularization constant added to joint for numerical stability
    Returns:
        Normalized joint distribution matrix
    """
    joint = joint + reg  # Add regularization for numericsal stability
    joint /= joint.sum()  # Normalize to sum to 1

    for _ in range(num_iters):
        joint /= joint.sum(dim=1, keepdim=True) + epsilon  # Normalize rows
        joint /= joint.sum(dim=0, keepdim=True) + epsilon  # Normalize columns

    return joint


def iic_loss_with_sinkhorn(pred1, pred2, lambda_entropy=4, block_size=64):
    """
        Computes the IIC loss with Sinkhorn-Knopp normalization and entropy regularizati---------------------------------------------------------------------------
    RuntimeError                              Traceback (most recent call last)
    Cell In[23], line 1
    ----> 1 if torch.isnan(img1).any():
          2     print("Tensor contains NaNs")
          3 if torch.isinf(img1).any():

    RuntimeError: CUDA error: device-side assert triggered
    Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.on.
        Args:
            pred1: Softmax probabilities of shape (B, C, H, W)
            pred2: Softmax probabilities of shape (B, C, H, W)
            lambda_entropy: Weight for entropy regularization term
            block_size: Size of the block to process at a time
        Returns:
            Combined loss value
    """
    # Compute blockwise joint distribution
    joint = compute_blockwise_joint_distribution(pred1, pred2, block_size=block_size)

    # Apply Sinkhorn-Knopp normalization
    joint = sinkhorn_knopp(joint)

    # Marginal distributions
    marginal1 = joint.sum(dim=1, keepdim=True)
    marginal2 = joint.sum(dim=0, keepdim=True)

    # Mutual information loss
    mi = torch.sum(joint * torch.log((joint + 1e-10) / (marginal1 * marginal2 + 1e-10)))

    # Entropy regularization to encourage balanced cluster usage
    marginal = joint.sum(dim=0)
    entropy = -torch.sum(marginal * torch.log(marginal + 1e-10))

    return -mi + lambda_entropy * entropy
