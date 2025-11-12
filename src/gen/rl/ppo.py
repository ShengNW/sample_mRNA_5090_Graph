
import torch, torch.nn as nn, torch.nn.functional as F
VOCAB="AUGC"; V=4
class Policy(nn.Module):
    def __init__(
        self,
        L5,
        L3,
        hidden=512,
        num_organs=32,
        num_layers=6,
        num_heads=8,
        dropout=0.0,
    ):
        super().__init__()
        L=L5+L3; self.L=L
        self.emb = nn.Embedding(V, hidden)
        self.cond = nn.Embedding(num_organs, hidden)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.tr = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pi = nn.Linear(hidden, V)
        self.v = nn.Linear(hidden, 1)
    def forward(self, x_idx, organ_id):
        H = self.emb(x_idx) + self.cond(organ_id).unsqueeze(1)
        H = self.tr(H)
        logits = self.pi(H)
        value = self.v(H.mean(1)).squeeze(-1)
        return logits, value
