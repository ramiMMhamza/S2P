#!/usr/bin/env bash
# Example Slurm job (adapt partition, GPUs and environment to your cluster).
#   sbatch scripts/slurm_train.sh configs/mmt_msmt17_to_market1501.yaml RUN_NAME [extra options]
#SBATCH --job-name=s2p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --output=slurm-%j.out
set -euo pipefail

# e.g.: eval "$(conda shell.bash hook)" && conda activate s2p
cd "${SLURM_SUBMIT_DIR}"
GPUS=${SLURM_GPUS_ON_NODE:-2} scripts/dist_train.sh "$@"
