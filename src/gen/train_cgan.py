
import argparse, contextlib, yaml
from pathlib import Path
import pandas as pd, torch
try:
    from torch.nn.attention import sdpa_kernel as sdp_kernel_ctx
except ImportError:  # pragma: no cover - fallback for older torch
    sdp_kernel_ctx = None
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from src.gen.models.cgan import Generator, Discriminator, VOCAB

TOK2ID = {c:i for i,c in enumerate(VOCAB)}

def one_hot_batch(seqs, L):
    B = len(seqs)
    x = torch.zeros(B, len(VOCAB), L, dtype=torch.float32)
    for i,s in enumerate(seqs):
        s = s.upper().replace("T","U"); s = s[:L].ljust(L,"U")
        for j,ch in enumerate(s):
            x[i, TOK2ID.get(ch,1), j] = 1.0
    return x

class SeedPairs(Dataset):
    def __init__(self, df, L5, L3):
        self.df=df; self.L5=L5; self.L3=L3
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        x5 = one_hot_batch([r.utr5], self.L5)
        x3 = one_hot_batch([r.utr3], self.L3)
        x = torch.cat([x5, x3], dim=-1).squeeze(0).transpose(0,1)  # (L,V)
        return x, int(r.organ_id)

def gradient_penalty(D, real, fake, organ_id, device, sdpa_context):
    """Compute the WGAN-GP gradient penalty in float32 for stability."""
    autocast_ctx = torch.amp.autocast("cuda", enabled=False) if torch.cuda.is_available() else contextlib.nullcontext()
    with sdpa_context(), autocast_ctx:
        real32 = real.detach().to(device=device, dtype=torch.float32)
        fake32 = fake.detach().to(device=device, dtype=torch.float32)
        alpha = torch.rand(real32.size(0), 1, 1, device=device, dtype=torch.float32)
        interp = (alpha*real32 + (1-alpha)*fake32).requires_grad_(True)
        d_interp = D(interp, organ_id)
        grad = torch.autograd.grad(outputs=d_interp, inputs=interp,
                                   grad_outputs=torch.ones_like(d_interp),
                                   create_graph=True, retain_graph=True, only_inputs=True)[0]
        grad = grad.view(grad.size(0), -1)
        gp = ((grad.norm(2, dim=1) - 1) ** 2).mean()
    return gp

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/m2_cgan.yaml")
    ap.add_argument("--seed_csv", default="outputs/phase2/m1/m1_topk.csv")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, "r"))
    out = Path(cfg["io"]["out_dir"]); out.mkdir(parents=True, exist_ok=True)

    L5=cfg["arch"]["max_len_utr5"]; L3=cfg["arch"]["max_len_utr3"]
    df = pd.read_csv(args.seed_csv)
    num_organs = int(df["organ_id"].max())+1 if "organ_id" in df.columns else 32
    ds = SeedPairs(df, L5, L3)
    dl = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G = Generator(L5,L3, hidden=cfg["arch"]["hidden"], num_layers=cfg["arch"]["num_layers"], num_organs=num_organs).to(device)
    D = Discriminator(L5,L3, hidden=cfg["arch"]["hidden"], num_layers=4, num_organs=num_organs).to(device)
    optG = torch.optim.AdamW(G.parameters(), lr=cfg["train"]["g_lr"], betas=(0.5,0.9))
    optD = torch.optim.AdamW(D.parameters(), lr=cfg["train"]["d_lr"], betas=(0.5,0.9))

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

        if sdp_kernel_ctx is not None:
            def sdpa_context():
                return sdp_kernel_ctx(
                    enable_flash=True,
                    enable_mem_efficient=False,
                    enable_math=True,
                )
        else:
            def sdpa_context():
                return torch.backends.cuda.sdp_kernel(
                    enable_flash=True,
                    enable_mem_efficient=False,
                    enable_math=True,
                )

        def amp_context():
            return torch.amp.autocast("cuda", dtype=torch.bfloat16)
    else:
        def sdpa_context():
            return contextlib.nullcontext()

        def amp_context():
            return contextlib.nullcontext()

    tau = float(cfg["train"]["gumbel_tau_start"]); tau_end = float(cfg["train"]["gumbel_tau_end"])

    for epoch in range(cfg["train"]["epochs"]):
        for xb, org in tqdm(dl, desc=f"CGAN epoch {epoch+1}"):
            xb = xb.to(device=device, dtype=torch.float32, non_blocking=True)      # (B, L, V)
            org = org.to(device, non_blocking=True)

            # D
            z = torch.randn(xb.size(0), D.head[0].in_features, device=device)
            with sdpa_context(), amp_context():
                y = G(z, org, tau=tau)          # (B, L, V)
                d_real = D(xb, org)
                d_fake = D(y.detach(), org)
                lossD_core = -(d_real.mean() - d_fake.mean())
            gp = gradient_penalty(D, xb, y.detach(), org, device, sdpa_context)
            lossD = lossD_core.float() + cfg["train"]["gp_lambda"]*gp
            optD.zero_grad()
            lossD.backward()
            optD.step()

            # G
            with sdpa_context(), amp_context():
                d_fake = D(y, org)
                lossG = -d_fake.mean()
            lossG = lossG.float()
            optG.zero_grad()
            lossG.backward()
            optG.step()
        tau = max(tau_end, tau*0.95)
        torch.save(G.state_dict(), out/f"cganG_epoch{epoch+1}.pt")
    print(f"[CGAN] Done. Weights under {out}")

if __name__ == "__main__":
    main()
