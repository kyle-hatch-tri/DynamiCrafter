version=$1 ##1024, 512, 256
seed=123
name=dynamicrafter_$1_seed${seed}

ckpt="/<path_to_checkpoints>/epoch=XX-step=XXXX.ckpt"
config=configs/inference_$1_v1.0.yaml

echo "ckpt: $ckpt"

prompt_dir=prompts/$1/
res_dir="/<path_to_results>"

export HF_HOME="/<path_to_cache>"

if [ "$1" == "256" ]; then
    H=256
    FS=3  ## This model adopts frame stride=3, range recommended: 1-6 (larger value -> larger motion)
elif [ "$1" == "512" ]; then
    H=320
    FS=24 ## This model adopts FPS=24, range recommended: 15-30 (smaller value -> larger motion)
elif [ "$1" == "1024" ]; then
    H=576
    FS=10 ## This model adopts FPS=10, range recommended: 15-5 (smaller value -> larger motion)
else
    echo "Invalid input. Please enter 256, 512, or 1024."
    exit 1
fi

export CUDA_VISIBLE_DEVICES=0

if [ "$1" == "256" ]; then
python3 scripts/evaluation/inference.py \
--seed ${seed} \
--ckpt_path $ckpt \
--config $config \
--savedir $res_dir/$name \
--n_samples 1 \
--bs 2 --height ${H} --width $1 \
--unconditional_guidance_scale 7.5 \
--ddim_steps 50 \
--ddim_eta 1.0 \
--prompt_dir $prompt_dir \
--text_input \
--video_length 16 \
--frame_stride ${FS}
else
python3 scripts/evaluation/inference.py \
--seed ${seed} \
--ckpt_path $ckpt \
--config $config \
--savedir $res_dir/$name \
--n_samples 1 \
--bs 1 --height ${H} --width $1 \
--unconditional_guidance_scale 7.5 \
--ddim_steps 50 \
--ddim_eta 1.0 \
--prompt_dir $prompt_dir \
--text_input \
--video_length 16 \
--frame_stride ${FS} \
--timestep_spacing 'uniform_trailing' --guidance_rescale 0.7 --perframe_ae
fi
