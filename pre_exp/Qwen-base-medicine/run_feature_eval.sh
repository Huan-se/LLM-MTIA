#!/bin/bash
set -e

if [ -z "$1" ]; then
    echo "❌ 错误: 未指定评估阶段。"
    echo "💡 用法: ./run_feature_eval.sh [0|2|3] [GPU_ID]"
    echo "  0 - 评估 纯基座模型 (Base) vs Oracle"
    echo "  2 - 评估 Phase 2 (Baseline 基线模型) vs Oracle"
    echo "  3 - 评估 Phase 3 (Proposed 对齐模型) vs Oracle"
    exit 1
fi

PHASE=$1
GPU_ID=${2:-"7"} # 如果没有提供第二个参数，默认使用显卡 7

# 注入显存优化全局变量
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "=================================================="
echo "🔍 准备执行特征映射差异量化 (GPU: $GPU_ID)..."

case $PHASE in
    0)
        echo "🚀 评估 [纯基座模型] vs Oracle..."
        python -u eval_feature_distance.py \
            --eval_model_path "./models/Qwen2.5-1.5B-Base" \
            --oracle_model_path "./outputs/Oracle_Model_Merged_Base" \
            --dataset_path "./datasets/OpenCodeInstruct" \
            --gpu_id $GPU_ID
        ;;
    2)
        echo "🚀 评估 [Phase 2 Baseline] vs Oracle..."
        python -u eval_feature_distance.py \
            --eval_model_path "./outputs/Phase2_Baseline_Merged_Base" \
            --oracle_model_path "./outputs/Oracle_Model_Merged_Base" \
            --dataset_path "./datasets/OpenCodeInstruct" \
            --gpu_id $GPU_ID
        ;;
    3)
        echo "🚀 评估 [Phase 3 Proposed] vs Oracle..."
        python -u eval_feature_distance.py \
            --eval_model_path "./outputs/Phase3_Proposed_Merged_Base" \
            --oracle_model_path "./outputs/Oracle_Model_Merged_Base" \
            --dataset_path "./datasets/OpenCodeInstruct" \
            --gpu_id $GPU_ID
        ;;
    *)
        echo "❌ 错误: 无效的阶段参数 '$PHASE'"
        exit 1
        ;;
esac

echo "=================================================="
echo "✅ 阶段 $PHASE 特征量化评估完毕！"