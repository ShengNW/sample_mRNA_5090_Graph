
import torch, torch.nn as nn, torch.nn.functional as F
VOCAB="AUGC"; V=4

def gumbel_softmax_sample(logits, tau: float):
    U = torch.rand_like(logits)
    g = -torch.log(-torch.log(U.clamp(min=1e-10, max=1-1e-10)))
    y = (logits + g) / tau
    return F.softmax(y, dim=-1)

class Generator(nn.Module):
    def __init__(self, L5, L3, hidden=512, num_layers=6, num_organs=32):
        super().__init__()
        L = L5+L3; self.L5=L5; self.L3=L3
        self.cond = nn.Embedding(num_organs, hidden)
        self.fc = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU())
        self.tr = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden, nhead=8, dropout=0.0, batch_first=True),
            num_layers=num_layers
        )
        self.proj = nn.Linear(hidden, V)
        self.pos = nn.Parameter(torch.randn(1, L, hidden)*0.02)
    def forward(self, z, organ_id, tau=1.0):
        h = self.cond(organ_id) + self.fc(z)
        B = z.size(0); L = self.pos.size(1)
        H = h.unsqueeze(1).expand(B, L, -1) + self.pos
        H = self.tr(H)
        logits = self.proj(H)
        y = gumbel_softmax_sample(logits, tau)  # (B, L, V)
        return y

class Discriminator(nn.Module):
    def __init__(self, L5, L3, hidden=512, num_layers=4, num_organs=32):
        super().__init__()
        L = L5+L3; self.L5=L5; self.L3=L3
        self.embed = nn.Linear(V, hidden)
        self.cond = nn.Embedding(num_organs, hidden)
        self.tr = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden, nhead=8, dropout=0.0, batch_first=True),
            num_layers=num_layers
        )
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.pos = nn.Parameter(torch.randn(1, L, hidden)*0.02)
    def forward(self, xOH, organ_id):
        H = self.embed(xOH) + self.cond(organ_id).unsqueeze(1) + self.pos
        H = self.tr(H)
        h = H.mean(dim=1)
        return self.head(h).squeeze(-1)
