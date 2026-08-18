#!/bin/bash
set -e  

if [ -z "$1" ]; then
    echo "❌ 错误: 未指定训练阶段。"
    echo "💡 用法: ./run_finetuning.sh [1|2|3] [GPU_ID]"
    echo "  1 - 训练 Phase 1 (Oracle 目标上限模型)"
    echo "  2 - 训练 Phase 2 (Baseline 代理基线模型)"
    echo "  3 - 训练 Phase 3 (Proposed 默认参数对齐模型)"
    exit 1
fi

PHASE=$1
GPU_ID=${2:-"0"}

# 全局注入 GPU 资源分配与显存优化配置
export CUDA_VISIBLE_DEVICES=$GPU_ID
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "=================================================="
echo "🔥 启动医学模型微调训练 | 分配显卡编号: $GPU_ID"
echo "=================================================="

case $PHASE in
    1)
        echo ">>> [Phase 1] 正在私有医学数据上微调 Oracle 目标模型..."
        python -u train_oracle.py
        ;;
    2)
        echo ">>> [Phase 2] 正在公开代理数据上训练 Baseline 基线模型..."
        python -u train_baseline.py
        ;;
    3)
        echo ">>> [Phase 3] 正在执行 Proposed 动态自适应对齐训练..."
        # 给出一个默认的对齐超参数进行测试
        python -u train_align.py \
            --alpha 0.4 --gamma 0.1 --lam 0.0 --beta 0.0 \
            --start_mode "splicedbase" \
            --output_dir "./outputs/Phase3_Proposed_Medical" \
            --merged_save_dir "./outputs/Phase3_Proposed_Merged_Medical"
        ;;
    *)
        echo "❌ 错误: 无效的阶段参数 '$PHASE'"
        exit 1
        ;;
esac

echo "=================================================="
echo "✅ 阶段 $PHASE 训练执行完毕！"