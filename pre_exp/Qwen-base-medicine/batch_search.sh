#!/bin/bash

# ==========================================
# 1. 配置可用显卡与任务组
# ==========================================
# 填入你想要用于并行的 GPU 编号
GPUS=(1 2 3 4 5 6 7) 
NUM_GPUS=${#GPUS[@]}

# 实验配置格式: "alpha gamma lam beta start_mode"
# start_mode 选项: phase2 或 splicedbase
CONFIGS=(
    "0.4 0.1 0.0 0.0 splicedbase"
    "0.1 0.4 0.0 0.0 splicedbase"
    "0.2 0.2 0.0 0.0 splicedbase"
    "0.0 1.0 0.0 0.0 splicedbase"
    "0.2 0.1 0.0005 10.0 splicedbase"
    "0.0 0.1 0.0 5.0 splicedbase"
    "0.8 0.1 0.0 0.0 splicedbase"
)

mkdir -p ./search_outputs/eval_results
mkdir -p ./search_outputs/batch_logs

# ==========================================
# 2. 核心工作函数 (全链路执行)
# ==========================================
run_task() {
    local a=$1
    local g=$2
    local l=$3
    local b=$4
    local mode=$5
    local gpu=$6

    local exp_name="align_a${a}_g${g}_l${l}_b${b}_${mode}"
    local out_dir="./search_outputs/${exp_name}_ckpt"
    local merged_dir="./search_outputs/${exp_name}_merged"
    local log_file="./search_outputs/batch_logs/${exp_name}.log"

    local oracle_model="./outputs/Oracle_Model_Merged_Base"

    {
        set -e # 开启报错熔断
        echo "=================================================="
        echo "🚀 [GPU $gpu] 开始微调: $exp_name"
        CUDA_VISIBLE_DEVICES=$gpu python -u train_align.py \
            --alpha $a --gamma $g --lam $l --beta $b \
            --start_mode "$mode" \
            --output_dir "$out_dir" \
            --merged_save_dir "$merged_dir" 

        echo "📊 [GPU $gpu] 开始 CodeBLEU 测试: $exp_name"
        CUDA_VISIBLE_DEVICES=$gpu python -u eval_codebleu.py \
            --model_path "$merged_dir" \
            --dataset_path "./datasets/OpenCodeInstruct" \
            --output_json "./search_outputs/eval_results/${exp_name}.json" 

        echo "📉 [GPU $gpu] 开始特征映射距离量化: $exp_name"
        CUDA_VISIBLE_DEVICES=$gpu python -u eval_feature_integrated.py \
            --eval_model_path "$merged_dir" \
            --oracle_model_path "$oracle_model" \
            --dataset_path "./datasets/OpenCodeInstruct" 
        
        echo "✅ [GPU $gpu] 任务 $exp_name 全部完成！"
        echo "=================================================="
    } > "$log_file" 2>&1
}

# ==========================================
# 3. 任务分发与并发控制
# ==========================================
echo "🔥 开始批量分发超参数搜索任务，共 ${#CONFIGS[@]} 组配置..."

gpu_idx=0
for conf in "${CONFIGS[@]}"; do
    read a g l b mode <<< "$conf"
    current_gpu=${GPUS[$gpu_idx]}
    
    echo ">> 正在将任务 [a=${a}, g=${g}, l=${l}, b=${b}, mode=${mode}] 分发至 GPU ${current_gpu}..."
    run_task $a $g $l $b $mode $current_gpu &

    gpu_idx=$(( (gpu_idx + 1) % NUM_GPUS ))

    if [ $gpu_idx -eq 0 ]; then
        echo "⏳ GPU 插槽已满，等待当前批次任务全部执行完毕..."
        wait
    fi
done

wait
echo "🎉 所有批量搜参实验已圆满结束！请使用 parse_results.py 提取表格。"