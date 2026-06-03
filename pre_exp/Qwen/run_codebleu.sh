#!/bin/bash

# 检查是否传入了参数
if [ -z "$1" ]; then
    echo "❌ 错误: 未指定评估阶段。"
    echo "💡 用法: ./run_codebleu.sh [1|2|3]"
    echo "  1 - 评估 Phase 1 (Oracle 上限模型)"
    echo "  2 - 评估 Phase 2 (Baseline 基线模型)"
    echo "  3 - 评估 Phase 3 (Proposed 对齐模型)"
    exit 1
fi

PHASE=$1
GPU_ID="2" # 默认使用显卡2，你可以在这里统一修改

# 创建存放评测结果的文件夹
mkdir -p ./outputs/eval_results

echo "=================================================="

case $PHASE in
    1)
        echo "🚀 正在启动 Phase 1 (Oracle) CodeBLEU 评估..."
        python -u eval_codebleu.py \
            --model_path "./outputs/Oracle_Model_Merged" \
            --output_json "./outputs/eval_results/oracle.json" \
            --gpu_id $GPU_ID
        ;;
    2)
        echo "🚀 正在启动 Phase 2 (Baseline) CodeBLEU 评估..."
        python -u eval_codebleu.py \
            --model_path "./outputs/Phase2_Baseline_Merged" \
            --output_json "./outputs/eval_results/baseline.json" \
            --gpu_id $GPU_ID
        ;;
    3)
        echo "🚀 正在启动 Phase 3 (Proposed) CodeBLEU 评估..."
        python -u eval_codebleu.py \
            --model_path "./outputs/Phase3_Proposed_Merged" \
            --output_json "./outputs/eval_results/proposed.json" \
            --gpu_id $GPU_ID
        ;;
    *)
        echo "❌ 错误: 无效的参数 '$PHASE'"
        echo "💡 请输入 1, 2 或 3"
        exit 1
        ;;
esac

echo "=================================================="
echo "✅ 阶段 $PHASE 评估指令执行完毕！"