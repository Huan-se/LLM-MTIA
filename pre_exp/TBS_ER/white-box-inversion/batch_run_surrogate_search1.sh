#!/bin/bash

# ==========================================
# 1. 配置可用显卡与任务组
# ==========================================
GPUS=(3 4) 
NUM_GPUS=${#GPUS[@]}

# 实验配置格式: "lr layer method w_dm w_l1 steps loss changevar init range"
CONFIGS=(
    "0.00005 4 tbs 150 0.0 150000 cos atan:5 randn 0:5"
    "0.00005 4 tbs 100 0.0 150000 cos tanh:1 randn 0:5"
    "0.00005 4 tbs 100 0.0 200000 cos atan:5 randn 0:5"
    "0.00005 4 tbs 150 0.0 200000 cos tanh:1 randn 0:5"
)
mkdir -p ../resultsx/batch_logs

# ==========================================
# 2. 核心工作函数
# ==========================================
run_task() {
    local lr=$1
    local layer=$2
    local method=$3
    local w_dm=$4
    local w_l1=$5
    local steps=$6
    local loss=$7
    local changevar=$8
    local init=$9
    local range=${10}
    local gpu=${11}

    # 处理冒号: 防止文件名解析问题
    local cv_safename="cv${changevar//:/-}"
    local init_safename="in${init}"

    local target_model="Oracle_Model_Merged_Base"
    local surrogate_model="Qwen2.5-1.5B-Base"
    local dataset="OpenCodeInstruct/data"

    # 【核心修改】对齐跨模型实战的命名规范，加入 surrogate_model 名称
    local exp_name="Transfer_${surrogate_model}_l${layer}_${method}_lr${lr}_wdm${w_dm}_wl1${w_l1}_ep${steps}_${loss}_${cv_safename}_${init_safename}"
    # 在最前面加入 eval_transfer_区间，方便统一管理
    local folder_name="eval_transfer_${range//:/-}_${exp_name}"
    local log_file="../resultsx/batch_logs/${folder_name}_gpu${gpu}.log"

    {
        set -e 
        echo "=================================================="
        echo "🚀 [GPU $gpu] 开始代理模型跨模型窃取: $exp_name"
        
        CUDA_VISIBLE_DEVICES=$gpu python -u attack_batch.py \
            --target-model "$target_model" \
            --surrogate-model "$surrogate_model" \
            --dataset "$dataset" \
            --range "$range" \
            --folder "$folder_name" \
            --attack "$method" \
            --access-layer-id "$layer" \
            --num-steps "$steps" \
            --lr "$lr" \
            --w-dm "$w_dm" \
            --wd-l1 "$w_l1" \
            --optim "AdamW" \
            --in-state-loss "$loss" \
            --tbs-changevar "$changevar" \
            --init "$init" \
            --dtype "bfloat16" \
            --device "cuda" \
            --verbose

        echo "✅ [GPU $gpu] 任务 $folder_name 全部完成！"
    } > "$log_file" 2>&1
}

# ==========================================
# 3. 任务分发与并发控制
# ==========================================
echo "🔥 开始批量分发代理模型搜索，共 ${#CONFIGS[@]} 组配置..."

gpu_idx=0
for conf in "${CONFIGS[@]}"; do
    read lr layer method w_dm w_l1 steps loss changevar init range <<< "$conf"
    current_gpu=${GPUS[$gpu_idx]}
    
    run_task $lr $layer $method $w_dm $w_l1 $steps $loss $changevar $init $range $current_gpu &

    sleep 20 

    gpu_idx=$(( (gpu_idx + 1) % NUM_GPUS ))

    if [ $gpu_idx -eq 0 ]; then
        echo "⏳ GPU 插槽已满，等待当前批次并发任务执行完毕..."
        wait
    fi
done

wait
echo "🎉 跨模型代理搜参完毕！"