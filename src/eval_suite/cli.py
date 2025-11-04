# --- file: src/eval_suite/cli.py ---
from __future__ import annotations
import argparse, os, json
import numpy as np
import yaml
import torch
from torch.utils.data import DataLoader
from ..cnn_v1.dataset import ShardedUTRDataset
from ..cnn_v1.model import DualBranchCNN
from ..cnn_v1.utils import ensure_dir
from .metrics import basic_metrics
from .calibration import expected_calibration_error
from .plots import save_confusion_matrix, save_reliability_diagram
from .io import save_preds_table


def _load_yaml(path):
    with open(path,'r',encoding='utf-8') as f:
        return yaml.safe_load(f)


def _load_checkpoint(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    return ckpt


def predict(dset, model, device, batch_size=64, num_workers=4):
    loader = DataLoader(dset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    model.eval()
    all_logits=[]; all_y=[]
    with torch.no_grad():
        for batch in loader:
            x5 = batch['utr5'].to(device)
            x3 = batch['utr3'].to(device)
            y  = batch['label'].to(device)
            logits = model(x5, x3)
            all_logits.append(logits.cpu())
            all_y.append(y.cpu())
    logits = torch.cat(all_logits, dim=0).numpy()
    y_true = torch.cat(all_y, dim=0).numpy().astype(int)
    y_prob = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    return y_true, y_prob


def run_predict_and_eval(cfg):
    inputs = cfg['inputs']
    dataset_dir = inputs['dataset_dir']
    mapping_path = inputs['mapping_path']
    checkpoint_path = inputs['checkpoint_path']
    splits = inputs.get('splits', ['val','test'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = _load_checkpoint(checkpoint_path)
    n_channels = int(ckpt.get('n_channels', 4))
    num_classes = int(cfg.get('num_classes', 54))
    model = DualBranchCNN(in_ch=n_channels, num_classes=num_classes)
    model.load_state_dict(ckpt['model'])
    model.to(device)

    out_dir = cfg.get('out_dir', 'outputs/cnn_v1')
    fig_dir = cfg.get('fig_dir', 'reports/figures/cnn_v1')
    ensure_dir(out_dir); ensure_dir(fig_dir)

    summary = {}

    for sp in splits:
        ds = ShardedUTRDataset(dataset_dir, sp)
        y_true, y_prob = predict(ds, model, device,
                                 batch_size=int(cfg.get('batch_size',64)),
                                 num_workers=int(cfg.get('num_workers',4)))
        # metrics
        m = basic_metrics(y_true, y_prob)
        ece, bin_stats = expected_calibration_error(y_true, y_prob, n_bins=int(cfg.get('ece_bins', 15)))
        m['ece'] = ece
        # save preds table
        save_preds_table(os.path.join(out_dir, f'preds_{sp}.parquet'), y_true, y_prob)
        # plots
        save_confusion_matrix(os.path.join(fig_dir, f'cm_{sp}.png'), m['confusion_matrix'])
        save_reliability_diagram(os.path.join(fig_dir, f'reliability_{sp}.png'), bin_stats)
        summary[sp] = {k: (float(v) if isinstance(v, (int,float,np.floating)) else None) for k,v in m.items() if k!='confusion_matrix'}
        summary[sp]['ece'] = float(ece)

    # markdown summary
    md_path = cfg.get('report_path', 'reports/metrics_round1.md')
    os.makedirs(os.path.dirname(md_path), exist_ok=True)
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('# CNN v1 Evaluation Summary\n\n')
        for sp in splits:
            f.write(f'## Split: {sp}\n')
            for k,v in summary[sp].items():
                f.write(f'- {k}: {v}\n')
            f.write('\n')
    print('Evaluation done. Summary written to', md_path)


def main():
    parser = argparse.ArgumentParser(description='eval suite')
    sub = parser.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('predict_and_eval', help='predict and evaluate')
    p.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    if args.cmd=='predict_and_eval':
        cfg = _load_yaml(args.config)
        run_predict_and_eval(cfg)

if __name__=='__main__':
    main()
