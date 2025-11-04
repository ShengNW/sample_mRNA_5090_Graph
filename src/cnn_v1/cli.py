# --- file: src/cnn_v1/cli.py ---
from __future__ import annotations
import argparse, os, json
from .utils import load_yaml
from .train_loop import run_training


def main():
    parser = argparse.ArgumentParser(description='CNN v1 trainer')
    sub = parser.add_subparsers(dest='cmd', required=True)
    p_run = sub.add_parser('run', help='train the model')
    p_run.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    if args.cmd == 'run':
        cfg = load_yaml(args.config)
        best = run_training(cfg)
        print('Best checkpoint at:', best)

if __name__ == '__main__':
    main()
