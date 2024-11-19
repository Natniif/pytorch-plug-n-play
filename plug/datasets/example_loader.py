import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from plug.config import Config


class ExampleDataset(torch.utils.data.Dataset):
    """
    Description:
        - Example dataset class
    """

    def __init__(self):
        """
        Description:
            - Constructor method for the ExampleDataset class
        """
        super().__init__()
        self.x = torch.randn(100, 1)
        self.y = torch.randn(100, 1)

    def __len__(self):
        """
        Description:
            - Method to get the length of the dataset
        """
        return len(self.x)

    def __getitem__(self, idx):
        """
        Description:
            - Method to get an item from the dataset
        """
        return self.x[idx], self.y[idx]


class ExampleDataloader(LightningDataModule):
    """
    Description:
        - Example dataloader class
    """

    def __init__(self, _config):
        """
        Description:
            - Constructor method for the ExampleDataloader class
        """
        super().__init__()

        _config = Config.from_yaml(_config)

        self.batch_size = _config.model.batch_size
        self.num_workers = _config.data.num_workers

    def prepare_data(self):
        """
        Description:
            - Method to prepare the data
        """
        pass

    def setup(self, stage=None):
        """
        Description:
            - Method to set up the data
        """

        self.train_dataset = ExampleDataset()
        self.val_dataset = ExampleDataset()

    def train_dataloader(self):
        """
        Description:
            - Method to get the training dataloader
        """
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        """
        Description:
            - Method to get the validation dataloader
        """
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
