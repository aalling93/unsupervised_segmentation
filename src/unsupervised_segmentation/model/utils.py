import torch
import datetime


def save_checkpoint(model, optimizer, scheduler, epoch, loss, path="checkpoint.pt"):
    """
    Saves the model checkpoint including weights, optimizer, scheduler, and training state.

    Args:
        model (torch.nn.Module): The PyTorch model to save.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        epoch (int): Current epoch number.
        loss (float): Current loss value.
        path (str): Path to save the checkpoint.
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "loss": loss,
        "model_architecture": model.__class__.__name__,  # Save model class name for reference
        "timestamp": str(datetime.datetime.now()),  # Optional: Save timestamp for tracking
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved at {path}")


def load_checkpoint(model, optimizer, scheduler, path="checkpoint.pt"):
    """
    Loads the model checkpoint including weights, optimizer, scheduler, and training state.

    Args:
        model (torch.nn.Module): The PyTorch model instance.
        optimizer (torch.optim.Optimizer): The optimizer instance.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler instance.
        path (str): Path to the checkpoint file.

    Returns:
        int: The epoch to resume from.
        float: The loss value at the last saved epoch.
    """
    checkpoint = torch.load(path)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler and checkpoint["scheduler_state_dict"]:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]

    print(f"Checkpoint loaded from {path}. Resuming from epoch {epoch}, loss {loss:.7f}")
    return epoch, loss
