from typing_extensions import override
import torch.nn as nn
import torch

class NCF(nn.Module):
    def __init__(
            self,
            n_users: int, 
            n_items: int, 
            n_factors: int = 8, 
            dropout: float = 0.20
        ) -> None:
        super().__init__()
        # Embedding layers
        self.user_embeddings = nn.Embedding(n_users, n_factors)
        self.item_embeddings = nn.Embedding(n_items, n_factors)

        # MLP layers
        self.predictor = nn.Sequential(
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
        # Pass through embedding layers
        user_emb = self.user_embeddings(user)
        item_emb = self.item_embeddings(item)

        # Concat the two embeddings
        z = torch.cat([user_emb, item_emb], dim=-1)

        # Pass through MLP
        y = self.predictor(z)
        return y