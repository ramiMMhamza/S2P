#!/usr/bin/env bash
# Multi-GPU training on one machine.
#   GPUS=2 scripts/dist_train.sh configs/mmt_msmt17_to_market1501.yaml RUN_NAME [extra s2p.train options]
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_DIR}:${REPO_DIR}/OpenUnReID${PYTHONPATH:+:${PYTHONPATH}}"
# required by torch.use_deterministic_algorithms (TRAIN.deterministic: True)
export CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG:-:16:8}

PYTHON=${PYTHON:-python3}
GPUS=${GPUS:-2}
CONFIG=$1
RUN_NAME=$2
shift 2

while true; do # find an unused tcp port
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    if ! nc -z 127.0.0.1 "$PORT" < /dev/null &> /dev/null; then break; fi
done

"$PYTHON" -m torch.distributed.launch --nproc_per_node="$GPUS" --master_port="$PORT" --use_env \
    -m s2p.train "$CONFIG" --work-dir "$RUN_NAME" --launcher pytorch --tcp-port "$PORT" "$@"
