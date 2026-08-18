#!/bin/bash
set -e

if [ -z "$1" ]; then
    echo "❌ 错误: 未指定评估目标。"
    echo "💡 用法: ./run_evaluation.sh [0|1|2|3] [GPU_ID]"
    echo "  0 - 评估 纯基座模型 (Base) vs Oracle"
    echo "  1 - 评估 Oracle 上限模型自身"
    echo "  2 - 评估 Phase 2 (Baseline 基线模型)"
    echo "  3 - 评估 Phase 3 (Proposed 默认对齐模型)"
    exit 1
fi

PHASE=$1
GPU_ID=${2:-"0"}

export CUDA_VISIBLE_DEVICES=$GPU_ID
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# 统一定义路径
DATASET="./datasets/processed_medical_data/private_medical_o1_hf"
ORACLE_MODEL="./outputs/Oracle_Model_Merged_Medical"
BASE_MODEL="./models/Qwen2.5-1.5B-Base"

mkdir -p ./outputs/eval_results_medical

echo "=================================================="
echo "🔍 准备执行医学综合评估 (文本质量 & 特征距离) | GPU: $GPU_ID"

case $PHASE in
    0)
        echo "🚀 评估 [0] 纯基座模型 (Base-Prefix + Oracle-Suffix) ..."
        python -u eval_medical_integrated.py \
            --mode spliced \
            --eval_model_path "$BASE_MODEL" \
            --oracle_model_path "$ORACLE_MODEL" \
            --dataset_path "$DATASET" \
            --output_json "./outputs/eval_results_medical/base_spliced.json" \
            --tasks generation feature
        ;;
    1)
        echo "🚀 评估 [1] Oracle 目标模型自身 ..."
        python -u eval_medical_integrated.py \
            --mode standard \
            --eval_model_path "$ORACLE_MODEL" \
            --oracle_model_path "$ORACLE_MODEL" \
            --dataset_path "$DATASET" \
            --output_json "./outputs/eval_results_medical/oracle.json" \
            --tasks generation feature
        ;;
    2)
        echo "🚀 评估 [2] Phase 2 Baseline 代理模型 ..."
        python -u eval_medical_integrated.py \
            --mode standard \
            --eval_model_path "./outputs/Phase2_Baseline_Merged_Medical" \
            --oracle_model_path "$ORACLE_MODEL" \
            --dataset_path "$DATASET" \
            --output_json "./outputs/eval_results_medical/baseline.json" \
            --tasks generation feature
        ;;
    3)
        echo "🚀 评估 [3] Phase 3 Proposed 对齐模型 ..."
        python -u eval_medical_integrated.py \
            --mode standard \
            --eval_model_path "./outputs/Phase3_Proposed_Merged_Medical" \
            --oracle_model_path "$ORACLE_MODEL" \
            --dataset_path "$DATASET" \
            --output_json "./outputs/eval_results_medical/proposed.json" \
            --tasks generation feature
        ;;
    *)
        echo "❌ 错误: 无效的阶段参数 '$PHASE'"
        exit 1
        ;;
esac

echo "=================================================="
echo "✅ 阶段 $PHASE 综合评估完毕！"