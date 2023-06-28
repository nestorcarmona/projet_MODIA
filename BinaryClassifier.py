import torch
import torch.nn as nn

class BinaryClassifier(nn.Module):
    """A simple binary classifier."""
    def __init__(self, input_shape: int, dropout: float = 0.20):
        """A simple binary classifier.

        Parameters
        ----------
        input_shape : int
            The shape of the input data
        dropout : float, optional
            The dropout rate, by default 0.20
        """
        super(BinaryClassifier, self).__init__()
        self.layer1 = nn.Linear(input_shape, 64)
        self.layer2 = nn.Linear(64, 8)
        self.layer3 = nn.Linear(8, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Parameters
        ----------
        x : torch.Tensor
            The input data of shape (batch_size, input_shape)

        Returns
        -------
        torch.Tensor
            The output of the model of shape (batch_size, 1)
        """
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer3(x)
        x = self.sigmoid(x)
        return x
