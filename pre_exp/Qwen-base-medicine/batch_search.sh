#!/bin/bash

# ==========================================
# 1. 配置可用显卡与超参数组
# ==========================================
# 填入你想要用于并行的 GPU 编号
GPUS=(0 1 2 3) 
NUM_GPUS=${#GPUS[@]}

# 实验配置格式: "alpha gamma lam beta start_mode"
CONFIGS=(
    "0.4 0.1 0.0 0.0 splicedbase"
    "0.2 0.2 0.0 0.0 splicedbase"
    "0.0 1.0 0.0 0.0 splicedbase"
    "0.2 0.1 0.0005 10.0 splicedbase"
    "0.0 0.1 0.0 5.0 splicedbase"
    "0.8 0.1 0.0 0.0 splicedbase"
)

mkdir -p ./search_outputs/batch_logs

# ==========================================
# 2. 核心训练调度函数
# ==========================================
run_train_task() {
    local a=$1
    local g=$2
    local l=$3
    local b=$4
    local mode=$5
    local gpu=$6

    local exp_name="align_a${a}_g${g}_l${l}_b${b}_${mode}"
    local out_dir="./search_outputs/${exp_name}_ckpt_med"
    local merged_dir="./search_outputs/${exp_name}_merged_med"
    local log_file="./search_outputs/batch_logs/${exp_name}_train.log"

    {
        set -e
        echo "=================================================="
        echo "🚀 [GPU $gpu] 开始微调: $exp_name"
        export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
        
        CUDA_VISIBLE_DEVICES=$gpu python -u train_align.py \
            --alpha $a --gamma $g --lam $l --beta $b \
            --start_mode "$mode" \
            --output_dir "$out_dir" \
            --merged_save_dir "$merged_dir" 
        
        echo "✅ [GPU $gpu] 模型训练完毕: $exp_name"
        echo "=================================================="
    } > "$log_file" 2>&1
}

# ==========================================
# 3. 并发分发循环
# ==========================================
echo "🔥 开始批量分发超参搜索任务 (仅训练)，共 ${#CONFIGS[@]} 组配置..."

gpu_idx=0
for conf in "${CONFIGS[@]}"; do
    read a g l b mode <<< "$conf"
    current_gpu=${GPUS[$gpu_idx]}
    
    echo ">> 分发任务 [a=${a}, g=${g}, l=${l}, b=${b}] 至 GPU ${current_gpu}..."
    run_train_task $a $g $l $b $mode $current_gpu &

    gpu_idx=$(( (gpu_idx + 1) % NUM_GPUS ))

    if [ $gpu_idx -eq 0 ]; then
        echo "⏳ GPU 插槽已满，等待当前批次训练完毕..."
        wait
    fi
done

wait
echo "🎉 批量搜参训练结束！模型权重已保存在 ./search_outputs/ 目录下。"