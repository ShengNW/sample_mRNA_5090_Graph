#!/usr/bin/env python
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml


def _subset_organ(
    df: pd.DataFrame,
    organ_id: int | None,
    organ_name: str | None,
    name: str,
) -> pd.DataFrame:
    if organ_id is None and organ_name is None:
        return df

    for col in ["organ_id_target", "organ", "organ_id"]:
        if col not in df.columns:
            continue
        series = df[col]
        mask = None
        if organ_id is not None and pd.api.types.is_numeric_dtype(series):
            mask = series == organ_id
        elif organ_name is not None:
            mask = series.astype(str) == str(organ_name)
        elif organ_id is not None:
            mask = series.astype(str) == str(organ_id)

        if mask is None:
            continue

        sub = df[mask].copy()
        print(f"[{name}] filtered by column {col}, rows={len(sub)}")
        return sub

    print(f"[{name}] no organ column matched, using all rows={len(df)}")
    return df


def _get_score(df: pd.DataFrame, name: str) -> pd.Series:
    for col in ["score_pred", "y_pred", "score", "pred", "reward"]:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(s):
                print(f"[{name}] using column {col} as score")
                return s
    raise RuntimeError(
        f"[{name}] score column missing; available columns: {list(df.columns)}"
    )


def _load_scores(path: Path, label: str, organ_id: int | None, organ_name: str | None):
    if not path.exists():
        print(f"[WARN] {label}: missing file {path}, skip")
        return None, None

    df = pd.read_csv(path)
    df = _subset_organ(df, organ_id, organ_name, label)
    if not len(df):
        print(f"[WARN] {label}: empty after organ filter, skip")
        return None, None

    s = _get_score(df, label)
    if len(s) > 50000:
        s = s.sample(50000, random_state=0)
        print(f"[{label}] sampled 50000 rows for plotting")

    return label, s


def _resolve_organ_name(root: Path, organ_id: int | None) -> str | None:
    if organ_id is None:
        return None
    cfg_path = root / "configs" / "gen_predict.yaml"
    if not cfg_path.exists():
        return None
    cfg = yaml.safe_load(cfg_path.read_text())
    dataset_dir = cfg.get("dataset_dir")
    if not dataset_dir:
        return None
    manifest = Path(dataset_dir)
    if not manifest.is_absolute():
        manifest = root / manifest
    manifest = manifest / "manifest.json"
    if not manifest.exists():
        return None
    data = json.loads(manifest.read_text())
    vocab = data.get("organ_vocab", {})
    return vocab.get(str(organ_id))


def main():
    root = Path(__file__).resolve().parents[1]
    raw_dir = root / "data" / "raw"
    fig_dir = root / "figs" / "outputs"
    fig_dir.mkdir(parents=True, exist_ok=True)

    organ_id = 0
    organ_name = _resolve_organ_name(root, organ_id)

    datasets = [
        ("Real (eval)", raw_dir / "predict_eval.csv"),
        ("M1 top-k", raw_dir / "generated_topk.csv"),
        ("M2 cVAE", raw_dir / "m2_cvae_scored.csv"),
        ("M2 cGAN", raw_dir / "m2_cgan_scored.csv"),
        ("M3 RL", raw_dir / "m3_rl_scored.csv"),
    ]

    all_scores = []
    for label, path in datasets:
        lab, s = _load_scores(path, label, organ_id, organ_name)
        if lab is not None:
            all_scores.append((lab, s))

    if not all_scores:
        print("No valid dataset found. Check file paths.")
        return

    concat = pd.concat([s for _, s in all_scores], axis=0)
    xmin, xmax = concat.min(), concat.max()
    bins = 80

    plt.figure(figsize=(8, 5))
    for label, s in all_scores:
        plt.hist(
            s,
            bins=bins,
            range=(xmin, xmax),
            density=True,
            histtype="step",
            linewidth=1.5,
            label=label,
        )

    plt.xlabel("score_pred (same predictor)")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()

    out_fig = fig_dir / "fig_score_pred_all_models.png"
    plt.savefig(out_fig, dpi=200)
    print(f"[OK] saved figure to {out_fig}")

    def summarize(label, s):
        s = s.dropna()
        print(
            f"[{label}] n={len(s)}, "
            f"mean={s.mean():.4f}, "
            f"std={s.std():.4f}, "
            f"q10={s.quantile(0.10):.4f}, "
            f"q50={s.quantile(0.50):.4f}, "
            f"q90={s.quantile(0.90):.4f}"
        )

    print("\n=== Summary stats ===")
    for label, s in all_scores:
        summarize(label, s)


if __name__ == "__main__":
    main()
