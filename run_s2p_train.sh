#!/usr/bin/env bash
# Usage: ./run_s2p_train.sh RUN_NAME [--epochs N --iters N --KDloss W --MMDloss W ...]
# Can be launched from any directory: paths are resolved from this script's location.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CUBLAS_WORKSPACE_CONFIG=:16:8
export CUDA_LAUNCH_BLOCKING=1
export PYTHONPATH="${REPO_DIR}/OpenUnReID${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON=${PYTHON:-"python3"}
METHOD=${METHOD:-"MMT"} # MMT or SpCL
GPUS=${GPUS:-2}
RUN_NAME=$1
shift

while true # find unused tcp port
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done

"$PYTHON" -m torch.distributed.launch --nproc_per_node="$GPUS" --master_port="$PORT" --use_env \
    "${REPO_DIR}/OpenUnReID/tools/${METHOD}/main.py" "${REPO_DIR}/OpenUnReID/tools/${METHOD}/config.yaml" \
    --work-dir="${RUN_NAME}" --launcher="pytorch" --tcp-port="${PORT}" --exec=2. "$@"
