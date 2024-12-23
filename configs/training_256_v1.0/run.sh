
DEBUG=$1

echo "DEBUG: $DEBUG"

# args
name="training_256_v1.0"
config_file=configs/${name}/config.yaml

# save root dir for logs, checkpoints, tensorboard record, etc.
save_root="/<path_to_results>/training" 

mkdir -p $save_root/$name

export HF_HOME="/<path_to_cache>" 


export HOST_GPU_NUM=8
echo "HOST_GPU_NUM: $HOST_GPU_NUM"


if [ "$DEBUG" -eq 1 ]; then
    # debugging
    CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch \
    --nproc_per_node=1 --nnodes=1 --master_addr=127.0.0.1 --master_port=12352 --node_rank=0 \
    ./main/trainer.py \
    --base $config_file \
    --train 1 \
    --name $name \
    --logdir $save_root \
    --devices 1 \
    --debug $DEBUG \
    lightning.trainer.num_nodes=1
else
    ## run
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m torch.distributed.launch \
    --nproc_per_node=$HOST_GPU_NUM --nnodes=1 --master_addr=127.0.0.1 --master_port=12352 --node_rank=0 \
    ./main/trainer.py \
    --base $config_file \
    --train 1 \
    --name $name \
    --logdir $save_root \
    --devices $HOST_GPU_NUM \
    --debug $DEBUG \
    lightning.trainer.num_nodes=1
fi

