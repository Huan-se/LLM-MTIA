#!/bin/bash
set -e

if [ -z "$1" ]; then
    echo "❌ 错误: 未指定评估阶段。"
    echo "💡 用法: ./run_codebleu.sh [1|2|3] [GPU_ID]"
    exit 1
fi

PHASE=$1
GPU_ID=${2:-"7"}

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
mkdir -p ./outputs/eval_results

echo "=================================================="
echo "📊 准备执行 CodeBLEU 评估 (GPU: $GPU_ID)..."

case $PHASE in
    1)
        python -u eval_codebleu.py \
            --model_path "./outputs/Oracle_Model_Merged_Base" \
            --output_json "./outputs/eval_results/oracle_base.json" \
            --gpu_id $GPU_ID
        ;;
    2)
        python -u eval_codebleu.py \
            --model_path "./outputs/Phase2_Baseline_Merged_Base" \
            --output_json "./outputs/eval_results/baseline_base.json" \
            --gpu_id $GPU_ID
        ;;
    3)
        python -u eval_codebleu.py \
            --model_path "./outputs/Phase3_Proposed_Merged_Base" \
            --output_json "./outputs/eval_results/proposed_base.json" \
            --gpu_id $GPU_ID
        ;;
    *)
        echo "❌ 错误: 无效的阶段参数 '$PHASE'"
        exit 1
        ;;
esac

echo "=================================================="