import os
import wandb
import subprocess
import logging

# export WANDB_DISABLE_SERVICE=true
os.environ["WANDB_MODE"] = "online"
wandb.login()


sweep_config = {
    "method": "random",
    "metric": {"name": "Validation loss", "goal": "minimize"},
    "parameters": {
        "model_weight": {"values": [0.1, 0.75, 5]},
        "epochs": {"value": 6},
    },
}


def train():
    with wandb.init() as run:
        # Convert sweep parameters to command-line arguments
        args = [
            "python3",
            "train.py",
            "--MODEL_WEIGHT_SIZE",
            str(run.config.model_weight),
            "--EPOCHS",
            str(run.config.epochs),
        ]
        # Call train.py with the specified arguments
        subprocess.run(args)


if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_config)  # , project="DetectionSweeps1"
    wandb.agent(sweep_id, function=train, count=3)
    logging.info("All runs finished. Clearing sweep and switching to online mode.")
