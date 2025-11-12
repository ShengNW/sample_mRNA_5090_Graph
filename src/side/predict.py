import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from src.side.model import DualBranchCNNFiLM


def _build_alphabet(manifest: Dict) -> Sequence[str]:
    alphabet = manifest.get("alphabet")
    if isinstance(alphabet, list) and alphabet:
        return [str(ch).upper().replace("U", "T") for ch in alphabet]
    return list("ACGTN")


def _one_hot_encode(seq: str, length: int, alphabet: Sequence[str], mapping: Dict[str, int]) -> np.ndarray:
    seq = (seq or "").upper().replace("U", "T")
    arr = np.zeros((len(alphabet), length), dtype=np.float32)
    idx_default = mapping.get("N", len(alphabet) - 1)

    if length <= 0:
        return arr

    if len(seq) > length:
        seq = seq[-length:]
    offset = max(0, length - len(seq))
    if offset > 0:
        arr[idx_default, :offset] = 1.0
    for pos, ch in enumerate(seq):
        arr[mapping.get(ch, idx_default), offset + pos] = 1.0
    if offset + len(seq) < length:
        arr[idx_default, offset + len(seq) :] = 1.0
    return arr


def _default_extra_channels(extra: int, length: int) -> np.ndarray:
    if extra <= 0:
        return np.zeros((0, length), dtype=np.float32)
    extras = np.zeros((extra, length), dtype=np.float32)
    if extra >= 2:
        extras[1] = 1.0  # tRNA distance channel defaults to the "far" value
    return extras


class PairDataset(Dataset):
    def __init__(
        self,
        rows: List[Tuple[str, str, int]],
        L5: int,
        L3: int,
        manifest: Dict,
    ) -> None:
        self.rows = rows
        self.L5 = L5
        self.L3 = L3
        self.alphabet = _build_alphabet(manifest)
        self.mapping = {ch: idx for idx, ch in enumerate(self.alphabet)}
        self.seq_channels = len(self.alphabet)
        utr5_shape = manifest["shapes"]["utr5"]
        utr3_shape = manifest["shapes"]["utr3"]
        self.total_c5 = int(utr5_shape[0])
        self.total_c3 = int(utr3_shape[0])
        extra5 = self.total_c5 - self.seq_channels
        extra3 = self.total_c3 - self.seq_channels
        self.extra5_default = _default_extra_channels(extra5, L5)
        self.extra3_default = _default_extra_channels(extra3, L3)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s5, s3, organ = self.rows[idx]
        enc5 = _one_hot_encode(s5, self.L5, self.alphabet, self.mapping)
        enc3 = _one_hot_encode(s3, self.L3, self.alphabet, self.mapping)

        x5 = np.zeros((self.total_c5, self.L5), dtype=np.float32)
        x5[: self.seq_channels] = enc5
        if self.extra5_default.size:
            x5[self.seq_channels :, :] = self.extra5_default

        x3 = np.zeros((self.total_c3, self.L3), dtype=np.float32)
        x3[: self.seq_channels] = enc3
        if self.extra3_default.size:
            x3[self.seq_channels :, :] = self.extra3_default

        return {
            "utr5": torch.from_numpy(x5),
            "utr3": torch.from_numpy(x3),
            "organ_id": torch.tensor(int(organ), dtype=torch.long),
        }

def load_phase1_model(manifest_path: str, ckpt_path: str, device: torch.device):
    man = json.load(open(manifest_path, "r"))
    in_channels = int(man["shapes"]["utr5"][0])
    num_organs = int(man.get("num_organs", 0) or len(man.get("organ_vocab", {})))
    model = DualBranchCNNFiLM(
        in_channels=in_channels, num_organs=num_organs,
        conv_channels=[64,128,256], stem_channels=32, film_dim=32,
        hidden_dim=256, dropout=0.2,
    )
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    return model.eval().to(device), man

def infer(model, loader, device):
    preds = []
    with torch.no_grad():
        for batch in loader:
            utr5 = batch["utr5"].to(device)
            utr3 = batch["utr3"].to(device)
            organ = batch["organ_id"].to(device)
            y = model(utr5, utr3, organ)
            preds.append(y.detach().cpu().numpy())
    return np.concatenate(preds, axis=0)

def main():
    import yaml
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/gen_predict.yaml")
    ap.add_argument("--input", required=True, help="CSV/TSV columns: utr5,utr3,organ_id")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))
    manifest = Path(cfg["dataset_dir"]) / "manifest.json"
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    bs = int(cfg.get("batch_size", 512)); nw = int(cfg.get("num_workers", 0))
    model, man = load_phase1_model(str(manifest), cfg["phase1_checkpoint"], device)
    L5 = int(man["shapes"]["utr5"][1]); L3 = int(man["shapes"]["utr3"][1])

    sep = "," if args.input.endswith(".csv") else "\t"
    df = pd.read_csv(args.input, sep=sep)
    rows = list(zip(df["utr5"].tolist(), df["utr3"].tolist(), df["organ_id"].tolist()))
    ds = PairDataset(rows, L5, L3, man)
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)
    pred = infer(model, loader, device)
    df["pred"] = pred
    df.to_csv(args.out, index=False)
    print(f"Wrote {args.out} ({len(df)} rows)")

if __name__ == "__main__":
    main()
