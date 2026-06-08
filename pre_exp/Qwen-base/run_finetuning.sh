#!/bin/bash
set -e  
# 检查是否传入了参数
if [ -z "$1" ]; then
    echo "❌ 错误: 未指定训练阶段。"
    echo "💡 用法: ./run_finetuning.sh [1|2|3]"
    echo "  1 - 训练 Phase 1 (Oracle 上限模型: train_oracle.py)"
    echo "  2 - 训练 Phase 2 (Baseline 基线模型: train_baseline.py)"
    echo "  3 - 训练 Phase 3 (Proposed 对齐模型: train_align.py)"
    exit 1
fi

PHASE=$1

echo "=================================================="

case $PHASE in
    1)
        echo "🔥 正在启动 Phase 1: Oracle 上限模型微调..."
        python -u train_oracle.py
        ./run_codebleu.sh 1
        ;;
    2)
        echo "🔥 正在启动 Phase 2: Baseline 基线模型微调..."
        python -u train_baseline.py
        ./run_codebleu.sh 2
        ;;
    3)
        echo "🔥 正在启动 Phase 3: Proposed 动态自适应对齐微调..."
        python -u train_align.py
        ./run_codebleu.sh 3
        ;;
    *)
        echo "❌ 错误: 无效的参数 '$PHASE'"
        echo "💡 请输入 1, 2 或 3"
        exit 1
        ;;
esac

echo "=================================================="
echo "✅ 阶段 $PHASE 训练指令执行完毕！"