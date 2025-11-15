#!/usr/bin/env bash
set -euo pipefail

REMOTE_PORT=42273
REMOTE_HOST="connect.westb.seetacloud.com"
REMOTE_USER="root"
REMOTE_DIR="/root/autodl-tmp/Sample_mRNA_011_5090_phase2/src_phase2"

# 如果没有参数，就给个简单提示
if [ "$#" -eq 0 ]; then
  echo "Usage: $0 <command to run on remote>"
  echo "  e.g. $0 'make mfe'"
  echo "       $0 'ls -la data/raw'"
  echo "       $0 'nohup python train.py > logs/train.log 2>&1 &'"
  exit 1
fi

# 把传进来的整个命令拼成一条字符串
REMOTE_CMD="$*"

echo "[local] SSH to ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PORT}"
echo "[local] cd ${REMOTE_DIR} && git fetch origin --prune && git pull --ff-only && ${REMOTE_CMD}"
echo

ssh -p "${REMOTE_PORT}" "${REMOTE_USER}@${REMOTE_HOST}" "
  set -euo pipefail
  cd ${REMOTE_DIR}
  echo '[remote] pwd: ' \$(pwd)
  echo '[remote] git fetch origin --prune...'
  git fetch origin --prune
  echo '[remote] git pull --ff-only...'
  git pull --ff-only
  echo '[remote] run: ${REMOTE_CMD}'
  ${REMOTE_CMD}
"
