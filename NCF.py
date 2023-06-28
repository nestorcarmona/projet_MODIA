from typing_extensions import override
import torch.nn as nn
import torch

class NCF(nn.Module):
    """Neural Collaborative Filtering (NCF)

    Reference: 
    ----------
    @article{he2017neural,
        title     = {Neural Collaborative Filtering},
        author    = {Xiangnan He and Lizi Liao and Hanwang Zhang and Liqiang Nie and Xia Hu and Tat-Seng Chua},
        journal   = {The Web Conference},
        year      = {2017},
        doi       = {10.1145/3038912.3052569},
        bibSource = {Semantic Scholar https://www.semanticscholar.org/paper/ad42c33c299ef1c53dfd4697e3f7f98ed0ca31dd}
    }
    """
    def __init__(self, n_users: int, n_items: int, n_factors: int = 8, dropout: float = 0.20) -> None:
        """Neural Collaborative Filtering (NCF)

        Parameters
        ----------
        n_users : int
            Number of users for the embeddings layer.
        n_items : int
            Number of items for the embeddings layer.
        n_factors : int, optional
            Embeddings layer size, by default 8
        dropout : float, optional
            Dropout rate, by default 0.20
        """
        super().__init__()
        # Embedding layers
        self.user_embeddings = torch.nn.Embedding(n_users, n_factors)
        self.item_embeddings = torch.nn.Embedding(n_items, n_factors)

        # MLP layers
        self.predictor = torch.nn.Sequential(
            nn.Linear(in_features=n_factors*2 , out_features=64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=32, out_features=1),
            nn.Sigmoid()
        )

    @override
    def forward(self, user: torch.tensor, item: torch.tensor) -> torch.Tensor:
        """Forward pass

        Parameters
        ----------
        user : torch.tensor
            User ids
        item : torch.tensor
            Item ids

        Returns
        -------
        torch.Tensor
            Predictions
        """ 
        # Pass through embedding layers
        user_emb = self.user_embeddings(user)
        item_emb = self.item_embeddings(item)

        # Concat the two embeddings
        z = torch.cat([user_emb, item_emb], dim=-1)

        # Pass through MLP
        y = self.predictor(z)
        return y