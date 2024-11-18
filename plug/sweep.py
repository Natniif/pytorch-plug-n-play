import os

import wandb

from plug.config import Config
from plug.run import main


def sweep(config_file: str):
    """
    Description:

    Args:
        config_file (str): Path to the configuration file

    """
    _config = Config.from_yaml(os.path.join("configs", config_file))

    wandb.login()

    sweep_config = {
        "method": "bayes",  # for example, grid, random, bayesian
        "metric": {"name": "val_loss", "goal": "minimize"},
        "parameters": {
            "model.batch_size": {"values": [32, 64, 128, 256]},
            # this will take a lot of training to get good results
            # maybe better way of testing schedulers
            "scheduler.scheduler": {
                "values": ["plateau", "step", "cosine", "cyclic", "none"]
            },
            "optimizer.optimizer": {"values": ["adam", "adamw", "sgd"]},
            "weight_decay": {"values": [0.01, 0.001, 0.0001]},
            "lr": {"values": [0.1, 0.01, 0.001, 0.0001]},
            "trainer.max_epochs": {"values": [1, 2, 3, 4, 5]},
        },
    }

    sweep_id = wandb.sweep(sweep_config, project=_config.wandb.project_name)

    wandb.agent(sweep_id, function=main, count=100)
