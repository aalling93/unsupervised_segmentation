class PiCIELoss(nn.Module):
    import torch

    def __init__(self, lambda_entropy=0.1):
        super(PiCIELoss, self).__init__()
        self.lambda_entropy = lambda_entropy

    def forward(self, pred1, pred2):
        """
        Compute the cross-view contrastive loss.
        Args:
            pred1, pred2 (torch.Tensor): Predicted cluster assignments from two augmentations.
        Returns:
            torch.Tensor: Loss value.
        """
        # Compute cross-entropy loss
        loss = F.cross_entropy(pred1, pred2)

        # Entropy regularization (optional)
        entropy = -torch.mean(torch.sum(F.softmax(pred1, dim=1) * F.log_softmax(pred1, dim=1), dim=1))
        loss += self.lambda_entropy * entropy

        return loss
