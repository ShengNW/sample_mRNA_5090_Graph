import argparse
from pathlib import Path
from typing import List, Sequence

import random
import torch
import yaml
from tqdm import tqdm

from src.gen.rl.ppo import Policy, VOCAB
from src.side.predict import (
    _build_alphabet,
    _default_extra_channels,
    _one_hot_encode,
    load_phase1_model,
)

TOK2ID = {c: i for i, c in enumerate(VOCAB)}
ID2TOK = {i: c for c, i in TOK2ID.items()}


class SidePredictor:
    """In-memory wrapper around the SIDE predictor used during RL training."""

    def __init__(self, predict_cfg: str) -> None:
        cfg = yaml.safe_load(open(predict_cfg, "r"))

        manifest_path = Path(cfg["dataset_dir"]) / "manifest.json"
        device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

        self.model, self.manifest = load_phase1_model(str(manifest_path), cfg["phase1_checkpoint"], device)
        self.model.eval()
        self.device = device

        self.batch_size = int(cfg.get("batch_size", 512))
        self.alphabet = _build_alphabet(self.manifest)
        self.mapping = {ch: idx for idx, ch in enumerate(self.alphabet)}
        self.seq_channels = len(self.alphabet)

        shapes = self.manifest["shapes"]
        self.L5 = int(shapes["utr5"][1])
        self.L3 = int(shapes["utr3"][1])
        self.total_c5 = int(shapes["utr5"][0])
        self.total_c3 = int(shapes["utr3"][0])

        extra5 = self.total_c5 - self.seq_channels
        extra3 = self.total_c3 - self.seq_channels
        extra5_arr = _default_extra_channels(extra5, self.L5)
        extra3_arr = _default_extra_channels(extra3, self.L3)
        self.extra5_default = torch.from_numpy(extra5_arr) if extra5_arr.size else None
        self.extra3_default = torch.from_numpy(extra3_arr) if extra3_arr.size else None

    def _encode_batch(
        self, seqs: Sequence[str], length: int, total_channels: int, extra_default: torch.Tensor | None
    ) -> torch.Tensor:
        batch_size = len(seqs)
        arr = torch.zeros((batch_size, total_channels, length), dtype=torch.float32)

        for i, seq in enumerate(seqs):
            enc = _one_hot_encode(seq, length, self.alphabet, self.mapping)
            arr[i, : self.seq_channels] = torch.from_numpy(enc)

        if extra_default is not None:
            arr[:, self.seq_channels :, :] = extra_default.unsqueeze(0)

        return arr

    def score(self, seq5: Sequence[str], seq3: Sequence[str], organ_ids: Sequence[int]) -> List[float]:
        assert len(seq5) == len(seq3) == len(organ_ids)
        preds: List[float] = []

        for start in range(0, len(seq5), self.batch_size):
            end = min(start + self.batch_size, len(seq5))
            cur_slice = slice(start, end)
            batch_seq5 = seq5[cur_slice]
            batch_seq3 = seq3[cur_slice]

            x5 = self._encode_batch(batch_seq5, self.L5, self.total_c5, self.extra5_default)
            x3 = self._encode_batch(batch_seq3, self.L3, self.total_c3, self.extra3_default)

            utr5 = x5.to(self.device, non_blocking=True)
            utr3 = x3.to(self.device, non_blocking=True)
            organs = torch.tensor(organ_ids[cur_slice], dtype=torch.long, device=self.device)

            with torch.no_grad():
                y = self.model(utr5, utr3, organs)
            preds.extend(y.detach().cpu().tolist())

        return preds


def sample_sequences(policy: Policy, organ_ids: torch.Tensor, L: int) -> torch.Tensor:
    device = next(policy.parameters()).device
    batch_size = int(organ_ids.shape[0])
    x = torch.full((batch_size, L), fill_value=TOK2ID["U"], dtype=torch.long, device=device)
    logits, _ = policy(x, organ_ids)
    probs = torch.softmax(logits, dim=-1)
    dist = torch.distributions.Categorical(probs=probs)
    samples = dist.sample()
    return samples


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/m3_rl.yaml")
    ap.add_argument("--predict-config", default="configs/gen_predict.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))
    out_dir = Path(cfg["io"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True

    L5 = int(cfg["env"]["max_len_utr5"])
    L3 = int(cfg["env"]["max_len_utr3"])
    L = L5 + L3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = Policy(L5, L3, hidden=512, num_organs=128).to(device)
    opt = torch.optim.AdamW(policy.parameters(), lr=float(cfg["train"]["lr"]))

    batch_size = max(1, int(cfg["train"].get("batch_size", 1)))
    total_steps = int(cfg["train"]["steps"])
    target = int(cfg["env"]["target_organ"])

    scorer = SidePredictor(args.predict_config)

    steps_done = 0
    pbar = tqdm(total=total_steps, desc="RL")
    organ_full = torch.full((batch_size,), target, dtype=torch.long, device=device)

    while steps_done < total_steps:
        cur_bs = min(batch_size, total_steps - steps_done)
        organ_slice = organ_full[:cur_bs]
        samples = sample_sequences(policy, organ_slice, L)

        samples_cpu = samples.detach().cpu().tolist()
        seqs = ["".join(ID2TOK[int(tok)] for tok in row) for row in samples_cpu]
        seq5 = [s[:L5] for s in seqs]
        seq3 = [s[L5:] for s in seqs]
        rewards = torch.tensor(scorer.score(seq5, seq3, [target] * cur_bs), dtype=torch.float32, device=device)

        logits, value = policy(samples, organ_slice)
        logp = torch.log_softmax(logits, dim=-1)
        chosen = torch.gather(logp, 2, samples.unsqueeze(-1)).squeeze(-1)

        policy_loss = -(chosen.mean(dim=1) * rewards).mean()
        value_loss = torch.mean((value - rewards) ** 2)
        loss = policy_loss + value_loss

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        for i in range(cur_bs):
            steps_done += 1
            pbar.update(1)
            if steps_done % 1000 == 0:
                with open(out_dir / "samples.csv", "a") as fh:
                    fh.write(f"{seq5[i]},{seq3[i]},{target},{rewards[i].item()}\n")

    torch.save(policy.state_dict(), out_dir / "ppo_final.pt")
    print(f"[RL] Done. Saved policy and samples in {out_dir}")


if __name__ == "__main__":
    main()
