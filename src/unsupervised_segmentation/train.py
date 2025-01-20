import wandb
import argparse
import os
import torch
import random
import numpy as np
import datetime
import sys
import multiprocessing as mp
import glob
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
import gc
import torch.optim as optim
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F


from unsupervised_segmentation.dataset.dataloader import MultispectralDataset
from unsupervised_segmentation.training.utils import monitor_cluster_usage
from unsupervised_segmentation.model.model import SimpleIICNet, ImprovedIICNet
from unsupervised_segmentation.model.losses import iic_loss_with_sinkhorn
from unsupervised_segmentation.model.utils import save_checkpoint, load_checkpoint


def main():
    """
    # s
    generally, in python, there is no reason to have a main() script... It is imporatn is other lang. as java.
    since it
    - Encapsulate Functionality: Keep the bulk of your code within a single function which improves readability, manageability, and makes the script modular and easier to debug.
    - Control Execution Flow: Only execute the high-level logic (like setting up data loaders, training models, etc.)
    when you explicitly call main(), giving you control over when and how your script executes when imported or run.
    but we use it often still..

    However, here it is important for the multiprocessing.
    - Multiprocessing: We are using PyTorchs DataLoader with num_workers > 0, which under the hood utilizes Pythons multiprocessing.
        To ensure that new processes start correctly without executing the script's initialization code multiple times, we wrap the entry point of your script in this if block.
    - Modularity and Safety: This setup makes our script safer to import in other Python scripts and interact with without triggering the execution of the training loop and other setup code unintentionally.


    Direct Execution: Without wrapping in main(), all code at the top level of a script executes as soon as the script is loaded into the Python interpreter.
    This happens regardless of whether the script is being run as an executable script or being imported as a module by another script.
    With main(), we control exactly when and how parts of your script run.
    By placing code within main() and calling main() only inside if __name__ == '__main__':, we ensure that this code only runs when the script is executed directly.
    This is crucial for scripts that might have side effects (like starting a training process or modifying files).

    WIHTOUT THE MAIN WE CAN NOT DO PROPER MULTI THREADING. More specifically:

    - Without main(): Code that initializes multiprocessing workers (such as the use of DataLoader with num_workers > 0 in PyTorch) runs immediately if placed at the top level.
    If this script is imported from another script that also uses multiprocessing, or even the same script is re-imported, it can lead to recursive process spawning or unexpected behavior,
    because each import or execution attempts to re-initialize all top-level code.
    - With main(): Initialization of multiprocessing components happens only under controlled conditions (when the script is explicitly executed, not when imported).
    This avoids issues in environments that do not use the fork method for starting worker processes, such as Windows or MacOS with Python 3.8 and above.
    """

    try:

        wandb.init(
            project="unsupervised_segmentation",
            config={"num_clusters": 10, "learning_rate": 1e-4, "batch_size": 32, "num_epochs": 70, "temperature": 1.0, "lambda_entropy": 1.0},
        )

        model_name = datetime.datetime.now().strftime("%d_%m_%YT%H_%M")
        path = f"../../data/model_training/{model_name}"
        os.makedirs("../../data/model_training", exist_ok=True)
        os.makedirs(path, exist_ok=True)

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        gc.collect()

        args = args_parser.parse_args()

        images = glob.glob(f"{args.path_to_data}/*.npy")

        imgs = []
        for pa in images:
            imgs.append(np.load(pa))
        imgs = np.array(imgs)
        img2 = np.transpose(imgs, axes=(0, 3, 1, 2))
        train_images = torch.tensor(img2[0:2500], dtype=torch.float32)
        test_images = torch.tensor(img2[2500:], dtype=torch.float32)

        # Create dataset and dataloader

        dataset = MultispectralDataset(train_images, transform=None, nr_channels=2)
        val_dataset = MultispectralDataset(test_images, transform=None, nr_channels=2)

        # dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
        # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=8, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=8, pin_memory=True)

        # Initialize model, optimizer, and hyperparameters
        num_clusters = 20  # Ensure this matches the number of clusters in your model

        # model = SimpleIICNet(input_channels=2, num_clusters=num_clusters).cuda()
        model = ImprovedIICNet(input_channels=2, num_clusters=10, use_gap=False).cuda()

        optimizer = optim.Adam(model.parameters(), lr=1e-2)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.01, patience=10, verbose=True)
        num_epochs = 7000
        temperature = 1.0
        lambda_entropy = 2.5

        min_val_loss = float("inf")
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0

            # Initialize cluster usage accumulator
            cluster_counts = torch.zeros(num_clusters).cuda()  # Accumulate counts across batches
            total_pixels = 0  # Track total number of pixels processed

            # Use tqdm to show progress bar for the dataloader
            with tqdm(dataloader, desc=f"Epoch [{epoch+1}/{num_epochs}]", unit="batch") as tepoch:
                for img1, img2 in tepoch:
                    img1, img2 = img1.cuda(), img2.cuda()

                    pred1 = model(img1)
                    pred2 = model(img2)

                    # Apply temperature scaling and softmax
                    pred1 = F.softmax(pred1 / temperature, dim=1)
                    pred2 = F.softmax(pred2 / temperature, dim=1)

                    # Check the shape of pred1 to ensure it matches the expected dimensions
                    # print(f"pred1 shape: {pred1.shape}")  # Debugging line

                    # Compute loss

                    loss = iic_loss_with_sinkhorn(pred1, pred2, lambda_entropy=lambda_entropy, block_size=64)

                    optimizer.zero_grad()
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    optimizer.step()

                    total_loss += loss.item()

                    # Update progress bar with batch loss
                    tepoch.set_postfix(loss=loss.item())

                    # Accumulate cluster counts for monitoring
                    cluster_assignments = torch.argmax(pred1, dim=1).view(-1)  # Flatten spatial dimensions

                    # Check if cluster assignments are within bounds
                    unique_clusters, counts = torch.unique(cluster_assignments, return_counts=True)
                    # print(f"Unique clusters: {unique_clusters}")  # Debugging line
                    if torch.any(unique_clusters >= num_clusters):
                        print(f"Error: cluster index exceeds num_clusters!")

                    # Safely update the cluster counts, ensuring valid indices
                    valid_clusters = unique_clusters[unique_clusters < num_clusters]
                    cluster_counts[valid_clusters] += counts.float()  # Accumulate counts
                    total_pixels += cluster_assignments.numel()  # Update total number of pixels

            # Compute mean cluster usage as a percentage
            mean_cluster_usage = (cluster_counts / total_pixels) * 100
            usage_dict = {i: mean_cluster_usage[i].item() for i in range(num_clusters) if mean_cluster_usage[i] > 0}
            this_loss = total_loss / len(dataloader)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {this_loss:.7f}")
            print(f"Mean Cluster Usage: {usage_dict}%")
            wandb.log({"epoch": epoch + 1, "train_loss": this_loss, "mean_cluster_usage": usage_dict}, commit=False)

            # Validation loop
            model.eval()
            val_loss = 0
            with torch.no_grad():
                with tqdm(val_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] Validation", unit="batch") as vepoch:
                    for img1, img2 in vepoch:
                        img1, img2 = img1.cuda(), img2.cuda()

                        pred1 = model(img1)
                        pred2 = model(img2)

                        pred1 = F.softmax(pred1 / temperature, dim=1)
                        pred2 = F.softmax(pred2 / temperature, dim=1)

                        loss = iic_loss_with_sinkhorn(pred1, pred2, lambda_entropy=lambda_entropy, block_size=64)
                        val_loss += loss.item()
                        vepoch.set_postfix(loss=loss.item())

            val_loss /= len(val_loader)
            print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.7f}")
            wandb.log({"epoch": epoch + 1, "val_loss": val_loss}, commit=True)

            scheduler.step(total_loss / len(dataloader))
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                save_checkpoint(model, optimizer, scheduler, epoch, val_loss, path=f"{path}/best_checkpoint.pt")
                torch.save(model, f"{path}/best_model_structure.pt")

    except Exception as e:
        print(e)


if __name__ == "__main__":

    args_parser = argparse.ArgumentParser(description="Training object detection model")
    ##### TRAINING HYPERPARAMETERS

    args_parser.add_argument(
        "-path_to_data", "--path_to_data", help="path_to_data notes", default="/mnt/hdd/Data/SAR/Sentinel1/Datasets/Archive/arctic/imgs", type=str
    )

    main()
