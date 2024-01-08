
# export https_proxy="http://129.183.4.13:8080" 
# export http_proxy="http://129.183.4.13:8080" 
export CUBLAS_WORKSPACE_CONFIG=:16:8
export CUDA_LAUNCH_BLOCKING=1
PYTHON=${PYTHON:-"python"}
METHOD="MMT" # SpCL(mean_net addet and cosidered as teacher)
path=$1
mkdir ./logs/$path
WORK_DIR="./$path/" 
export PYTHONPATH=./OpenUnReID/
GPUS=${GPUS:-2}

while true # find unused tcp port
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done
# echo $@
python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT --use_env \
./S2P/OpenUnReID/tools/$METHOD/main.py ./S2P/OpenUnReID/tools/$METHOD/config.yaml --work-dir=${WORK_DIR} \
    --launcher="pytorch" --tcp-port=${PORT}  --exec=2. ${@:2}

