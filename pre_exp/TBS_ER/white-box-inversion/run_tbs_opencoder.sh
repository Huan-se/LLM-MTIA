#!/bin/bash

# 遇到错误即刻终止
set -e
export CUDA_VISIBLE_DEVICES=0
# ================= 核心配置区 =================
# 目标与代理模型指向同一基础模型，用于验证攻击链路
TARGET_MODEL_DIR="Oracle_Model_Merged_Base"      
SURROGATE_MODEL_DIR="Oracle_Model_Merged_Base"   

# 使用目标包含 parquet 文件的底层目录名
DATASET_NAME="OpenCodeInstruct/data"         

# ================= 验证超参数 =================
# 为了验证是否陷入优化陷阱，必须从最浅的第 0 层开始
ACCESS_LAYER_ID=0                 
ATTACK_METHOD="tbs"               
OPTIMIZER="AdamW"
LOSS_TYPE="mse"                   

RANGES=("0:1" "1:2")                    
LEARNING_RATES=(0.01 0.05)
NUM_STEPS=50000

# 强制开启分布对齐与稀疏正则，防止反转出多语言/特殊符号乱码
W_DM=500                          
W_L1=0.01                         
# ==============================================

echo "启动 OpenCodeInstruct 数据集上的 TBS 自攻击验证测试..."

for RANGE in "${RANGES[@]}"; do
    for LR in "${LEARNING_RATES[@]}"; do
        
        FOLDER_NAME="SanityCheck_${TARGET_MODEL_DIR}_layer${ACCESS_LAYER_ID}_tbs_lr${LR}"
        
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