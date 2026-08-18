#!/bin/bash

# ==========================================
# 1. 显卡与搜参任务池配置
# ==========================================
GPUS=(0 1 2 3) 
NUM_GPUS=${#GPUS[@]}

# "alpha gamma lam beta start_mode"
CONFIGS=(
    "0.4 0.1 0.0 0.0 splicedbase"
    "0.1 0.4 0.0 0.0 splicedbase"
    "0.2 0.2 0.0 0.0 splicedbase"
    "0.0 1.0 0.0 0.0 splicedbase"
    "0.2 0.1 0.0005 10.0 splicedbase"
    "0.0 0.1 0.0 5.0 splicedbase"
    "0.8 0.1 0.0 0.0 splicedbase"
)

# 创建所有必要目录
mkdir -p ./search_outputs/eval_results_med
mkdir -p ./search_outputs/batch_logs_integrated

# 共享常数
ORACLE_MODEL="./outputs/Oracle_Model_Merged_Medical"
DATASET="./datasets/processed_medical_data/private_medical_o1_hf"

# ==========================================
# 2. 全链路单节点工作流 (Pipeline)
# ==========================================
run_pipeline_task() {
    local a=$1
    local g=$2
    local l=$3
    local b=$4
    local mode=$5
    local gpu=$6

    local exp_name="align_a${a}_g${g}_l${l}_b${b}_${mode}"
    local out_dir="./search_outputs/${exp_name}_ckpt_med"
    local merged_dir="./search_outputs/${exp_name}_merged_med"
    local json_out="./search_outputs/eval_results_med/${exp_name}.json"
    local log_file="./search_outputs/batch_logs_integrated/${exp_name}.log"

    {
        set -e # 开启报错熔断
        export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
        
        echo "=================================================="
        echo "🚀 [1/2] [GPU $gpu] 开始微调: $exp_name"
        CUDA_VISIBLE_DEVICES=$gpu python -u train_align.py \
            --alpha $a --gamma $g --lam $l --beta $b \
            --start_mode "$mode" \
            --output_dir "$out_dir" \
            --merged_save_dir "$merged_dir" 

        echo "📊 [2/2] [GPU $gpu] 开始医学综合测评 (文本质量 + 特征保真度)..."
        CUDA_VISIBLE_DEVICES=$gpu python -u eval_medical_integrated.py \
            --mode standard \
            --eval_model_path "$merged_dir" \
            --oracle_model_path "$ORACLE_MODEL" \
            --dataset_path "$DATASET" \
            --output_json "$json_out" \
            --tasks generation feature
        
        echo "✅ [GPU $gpu] 任务 $exp_name 全部流水线完成！"
        echo "=================================================="
        
        # 节省硬盘：评估完毕后删除庞大的检查点文件夹，仅保留融合模型
        rm -rf "$out_dir"
    } > "$log_file" 2>&1
}

# ==========================================
# 3. 任务并发分发
# ==========================================
echo "🔥 开始 [自动训练 + 测评] 并发流水线，共 ${#CONFIGS[@]} 组配置..."

gpu_idx=0
for conf in "${CONFIGS[@]}"; do
    read a g l b mode <<< "$conf"
    current_gpu=${GPUS[$gpu_idx]}
    
    echo ">> 下发流水线 [a=${a}, g=${g}, l=${l}, b=${b}] 至 GPU ${current_gpu}..."
    run_pipeline_task $a $g $l $b $mode $current_gpu &

    gpu_idx=$(( (gpu_idx + 1) % NUM_GPUS ))

    if [ $gpu_idx -eq 0 ]; then
        echo "⏳ GPU 插槽已满，等待当前批次流水线运行完毕..."
        wait
    fi
done

wait
echo "🎉 全自动参数搜索与综合测评已圆满结束！所有日志均保存在 ./search_outputs/batch_logs_integrated/ 中。"