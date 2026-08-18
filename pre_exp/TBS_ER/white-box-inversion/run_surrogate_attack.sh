#!/bin/bash

# ==========================================
# 1. 配置跨模型实战环境
# ==========================================
GPUS=(6 7) 
NUM_GPUS=${#GPUS[@]}

# 明确区分目标模型和代理模型 (根据你的环境路径设定)
TARGET_MODEL="Oracle_Model_Merged_Base"
SURROGATE_MODEL="Proxy_a0.2_g0.1_l0.0_b5.0_phase2"
# 固定当前探索出的最强“黄金参数”
LR=0.002
LAYER=4
METHOD="tbs"
W_DM=50
W_L1=0.0
STEPS=150000
LOSS="cos"
CHANGEVAR="atan:5"
INIT="randn"

# 将数据集划分成多个不重叠的小区间，利用 8 卡并行测出 40 条数据的结果
RANGES=("0:5" "5:10")

mkdir -p ../resultsx/batch_logs

# ==========================================
# 2. 核心工作函数
# ==========================================
run_surrogate_task() {
    local range=$1
    local gpu=$2

    local cv_safename="cv${CHANGEVAR//:/-}"
    local init_safename="in${INIT}"

    # 文件夹命名加入 Proxy 标识
    local exp_name="Transfer_${SURROGATE_MODEL}_l${LAYER}_${METHOD}_lr${LR}_wdm${W_DM}_ep${STEPS}_${LOSS}_${init_safename}"
    # 在外层加上 range，避免多个进程写同一个文件夹
    local folder_name="eval_transfer_${range//:/-}_${exp_name}"
    local log_file="../resultsx/batch_logs/${folder_name}_gpu${gpu}.log"

    local dataset="OpenCodeInstruct/data"

    {
        set -e 
        echo "=================================================="
        echo "🔥 [GPU $gpu] 开始跨模型窃取实战 | Target: $TARGET_MODEL -> Proxy: $SURROGATE_MODEL"
        echo "📍 攻击区间: $range | 步数: $STEPS"
        
        CUDA_VISIBLE_DEVICES=$gpu python -u attack_batch.py \
            --target-model "$TARGET_MODEL" \
            --surrogate-model "$SURROGATE_MODEL" \
            --dataset "$dataset" \
            --range "$range" \
            --folder "$folder_name" \
            --attack "$METHOD" \
            --access-layer-id "$LAYER" \
            --num-steps "$STEPS" \
            --lr "$LR" \
            --w-dm "$W_DM" \
            --wd-l1 "$W_L1" \
            --optim "AdamW" \
            --in-state-loss "$LOSS" \
            --tbs-changevar "$CHANGEVAR" \
            --init "$INIT" \
            --dtype "bfloat16" \
            --device "cuda" \
            --verbose

        echo "✅ [GPU $gpu] 跨模型任务 $folder_name 全部完成！"
    } > "$log_file" 2>&1
}

# ==========================================
# 3. 任务分发与并发控制
# ==========================================
echo "⚔️ 启动 Target vs Proxy 跨模型攻击评估，并发处理 ${#RANGES[@]} 个数据区间..."

gpu_idx=0
for range in "${RANGES[@]}"; do
    current_gpu=${GPUS[$gpu_idx]}
    
    echo ">> 分发样本区间 [$range] 至 GPU $current_gpu..."
    run_surrogate_task $range $current_gpu &

    # 错峰加载，保护系统物理内存
    sleep 20 

    gpu_idx=$(( (gpu_idx + 1) % NUM_GPUS ))

    if [ $gpu_idx -eq 0 ]; then
        echo "⏳ 8 张 GPU 已全部占满，等待当前批次跨模型任务跑完 (这可能需要较长时间)..."
        wait
    fi
done

wait
echo "🎉 跨模型实战评估完毕！"