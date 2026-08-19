#!/bin/bash

# 遇到错误即刻终止
set -e
export CUDA_VISIBLE_DEVICES="1"
# ================= 核心配置区 =================
TARGET_MODEL_DIR="Oracle_Model_Merged_Base"      
# 使用带有子目录的相对路径 (省略后缀)
DATASET_NAME="Magicoder-OSS-Instruct/data-oss_instruct-decontaminated"         

# 代理模型队列 (根据你的 tree 结构列出)
SURROGATE_MODELS=(
    "Oracle_Model_Merged"
    # "Phase2_Baseline_Merged"
    # "Phase3_Proposed_Merged"
    # "Phase3_Proposed_Merged_0_0_1_10"
    # "Phase3_Proposed_Merged_0_2_0_0"
    # "Phase3_Proposed_Merged_1_2_0_0"
    # "Phase3_Proposed_Merged_2_0_0_0"
    # "Phase3_Proposed_Merged_2_1_0_0"
    # "Phase3_Proposed_Merged_2_2_0_0"
    # "Phase3_Proposed_Merged_2_2_0_10"
)

ACCESS_LAYER_ID=4                 # 代理模型前缀层数
OPTIMIZER="AdamW"
LOSS_TYPE="mse"
W_DM=500                          # 开启分布匹配正则化
W_L1=0.01   

# 实验超参探索
RANGES=("0:5")                 # 预实验先测前5条样本跑通流程
LEARNING_RATES=(0.0005)
NUM_STEPS=40000
# ==============================================

echo "启动 Magicoder-OSS-Instruct 数据集上的 MTIA 批量测试..."

for SURROGATE_MODEL_DIR in "${SURROGATE_MODELS[@]}"; do
    for RANGE in "${RANGES[@]}"; do
        for LR in "${LEARNING_RATES[@]}"; do
            
            # 动态生成结果文件夹名称
            FOLDER_NAME="exp_${TARGET_MODEL_DIR}_vs_${SURROGATE_MODEL_DIR}_layer${ACCESS_LAYER_ID}_lr${LR}"
            
            echo "=================================================="
            echo "正在攻击: Target ($TARGET_MODEL_DIR) <- Surrogate ($SURROGATE_MODEL_DIR)"
            echo "样本区间: $RANGE | 学习率: $LR | 步数: $NUM_STEPS"
            echo "=================================================="

            # python attack_batch.py \
            #     --target-model "$TARGET_MODEL_DIR" \
            #     --surrogate-model "$SURROGATE_MODEL_DIR" \
            #     --dataset "$DATASET_NAME" \
            #     --range "$RANGE" \
            #     --folder "$FOLDER_NAME" \
            #     --attack "tbs" \
            #     --access-layer-id "$ACCESS_LAYER_ID" \
            #     --num-steps "$NUM_STEPS" \
            #     --lr "$LR" \
            #     --optim "$OPTIMIZER" \
            #     --in-state-loss "$LOSS_TYPE" \
            #     --dtype "bfloat16" \
            #     --device "cuda" \
            #     --verbose

            python attack_batch.py \
                --target-model "$TARGET_MODEL_DIR" \
                --surrogate-model "$SURROGATE_MODEL_DIR" \
                --dataset "$DATASET_NAME" \
                --range "$RANGE" \
                --folder "$FOLDER_NAME" \
                --attack "tbs" \
                --access-layer-id "$ACCESS_LAYER_ID" \
                --num-steps "$NUM_STEPS" \
                --lr "$LR" \
                --w-dm "$W_DM" \
                --wd-l1 "$W_L1" \
                --optim "$OPTIMIZER" \
                --in-state-loss "$LOSS_TYPE" \
                --device "cuda" \
                --verbose

            echo "[$SURROGATE_MODEL_DIR] 测试完成 (LR: $LR, Range: $RANGE)。"
            echo "清理显存，休眠 5 秒..."
            sleep 5
        done
    done
done

echo "所有的代理模型攻击实验均已执行完毕！"