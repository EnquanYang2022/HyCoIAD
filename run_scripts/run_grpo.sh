PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
export REPO_HOME="${PROJECT_ROOT}"
echo "REPO_HOME: $REPO_HOME"
data_paths="data_config/IAD.yaml"
image_folders="/mnt/data/yeq/Datasets/EMIT/"
model_path="/mnt/data/yeq/2025_6_6_3090_project/MLLM/Model_Huggface_Download/Qwen2.5-VL-3B-Instruct/" #命名中一定要带有Qwen2.5-VL
echo "data_paths: $data_paths"
echo "image_folders: $image_folders"
#优化显存
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#export CUDA_LAUNCH_BLOCKING=1
export EXP_NAME="Qwen2.5-VL-3B_HpyCo/" # TODO: change this to your own experiment name
TASK_TYPE="rec"
export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
# create the run directory and log file
mkdir -p ${REPO_HOME}/runs/${EXP_NAME}/log
export LOG_PATH="${REPO_HOME}/runs/${EXP_NAME}/log/debug_log.$(date +%Y-%m-%d-%H-%M-%S).txt"
# MAX_STEPS=1200 # TODO: change this to your own max steps
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    grpo.py \
    --use_vllm True \
    --output_dir ${REPO_HOME}/checkpoints/rl/${EXP_NAME} \
    --resume_from_checkpoint False \
    --model_name_or_path $model_path \
    --dataset_name $data_paths \
    --image_root $image_folders \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --gradient_checkpointing true \
    --logging_steps 1 \
    --num_train_epochs 2 \
    --max_completion_length 1024 \
    --bf16 \
    --torch_dtype bfloat16 \
    --attn_implementation flash_attention_2 \
    --data_seed 42 \
    --seed 42 \
    --save_steps 100 \
    --num_generations 8 \
    --reward_funcs accuracy format\
    --beta 0.04 \
    --deepspeed ${REPO_HOME}/run_scripts/deepspeed_config/zero2.json \
    --save_only_model true \
    --learning_rate 5e-6 \
    --freeze_vision_modules true
echo "Training completed for ${EXP_NAME}"

# 等待训练进程完全结束
wait

python eval/evaluate_batch_mmad_choice_mymodel.py
