#!/bin/bash

# 遇到错误即刻终止
set -e
export CUDA_VISIBLE_DEVICES="0"
# ================= 核心配置区 =================
TARGET_MODEL_DIR="Oracle_Model_Merged"      
SURROGATE_MODEL_DIR="Oracle_Model_Merged"   # 基准测试：自己攻击自己
DATASET_NAME="Magicoder-OSS-Instruct/data-oss_instruct-decontaminated"         

ACCESS_LAYER_ID=1            # ！！先从 0 层开始测试 ！！
ATTACK_METHOD="ts"                # ！！改为 ts (Token Selection) ！！
OPTIMIZER="AdamW"
LOSS_TYPE="mse"                   

RANGES=("0:1")                    
LEARNING_RATES=(0.05)             # 稍微加大一点学习率
NUM_STEPS=5000                    # TS 攻击收敛较快

# --- 新增关键正则化参数 ---
W_DM=500                          # 开启分布匹配正则化
W_L1=0.01                         # 开启 L1 稀疏正则
# ==============================================

echo "启动 TS 攻击基准测试 (Target vs Target, Layer 0)..."

for RANGE in "${RANGES[@]}"; do
    for LR in "${LEARNING_RATES[@]}"; do
        
        FOLDER_NAME="SanityCheck_Layer${ACCESS_LAYER_ID}_TS_lr${LR}"
        
        echo "=================================================="
        echo "Range: $RANGE | LR: $LR | W_DM: $W_DM | W_L1: $W_L1"
        echo "=================================================="

        python attack_batch.py \
            --target-model "$TARGET_MODEL_DIR" \
            --surrogate-model "$SURROGATE_MODEL_DIR" \
            --dataset "$DATASET_NAME" \
            --range "$RANGE" \
            --folder "$FOLDER_NAME" \
            --attack "$ATTACK_METHOD" \
            --access-layer-id "$ACCESS_LAYER_ID" \
            --num-steps "$NUM_STEPS" \
            --lr "$LR" \
            --w-dm "$W_DM" \
            --wd-l1 "$W_L1" \
            --optim "$OPTIMIZER" \
            --in-state-loss "$LOSS_TYPE" \
            --dtype "bfloat16" \
            --device "cuda" \
            --verbose
            
    done
done

echo "测试执行完毕！"