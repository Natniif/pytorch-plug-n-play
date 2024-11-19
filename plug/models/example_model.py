import torch

from plug.models.basemodel import BaseModel


class ExampleModel(BaseModel):
    """
    Description:
        - Example model class
    """

    def __init__(self, _config):
        """
        Description:
            - Constructor method for the ExampleModel class
        """
        super().__init__(_config)
        self.model = torch.nn.Linear(1, 1)

    def forward(self, x):
        """
        Description:
            - Forward pass of the model
        """
        return self.model(x)
