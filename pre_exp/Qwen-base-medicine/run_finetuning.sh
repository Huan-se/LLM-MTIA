#!/bin/bash
set -e  

if [ -z "$1" ]; then
    echo "❌ 错误: 未指定训练阶段。"
    echo "💡 用法: ./run_finetuning.sh [1|2|3] [GPU_ID]"
    echo "  1 - 训练 Phase 1 (Oracle 上限模型)"
    echo "  2 - 训练 Phase 2 (Baseline 基线模型)"
    echo "  3 - 训练 Phase 3 (Proposed 对齐模型)"
    exit 1
fi

PHASE=$1
GPU_ID=${2:-"7"}

# 全局注入 GPU 资源分配与显存优化配置
export CUDA_VISIBLE_DEVICES=$GPU_ID
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "=================================================="
echo "🔥 启动全链路自动化流程 | 分配显卡编号: $GPU_ID"
echo "=================================================="

case $PHASE in
    1)
        echo ">>> [1/2] 正在微调 Oracle 上限模型..."
        python -u train_oracle.py
        
        echo ">>> [2/2] 正在评估 Oracle CodeBLEU..."
        ./run_codebleu.sh 1 $GPU_ID
        
        # Oracle 自身不需要跑特征量化评估
        ;;
    2)
        echo ">>> [1/3] 正在微调 Baseline 基线模型..."
        python -u train_baseline.py
        
        echo ">>> [2/3] 正在评估 Baseline CodeBLEU..."
        ./run_codebleu.sh 2 $GPU_ID
        
        echo ">>> [3/3] 正在量化 Baseline 特征映射距离..."
        ./run_feature_eval.sh 2 $GPU_ID
        ;;
    3)
        echo ">>> [1/3] 正在微调 Proposed 动态自适应对齐模型..."
        python -u train_align.py
        
        echo ">>> [2/3] 正在评估 Proposed CodeBLEU..."
        ./run_codebleu.sh 3 $GPU_ID
        
        echo ">>> [3/3] 正在量化 Proposed 特征映射距离..."
        ./run_feature_eval.sh 3 $GPU_ID
        ;;
    *)
        echo "❌ 错误: 无效的阶段参数 '$PHASE'"
        exit 1
        ;;
esac

echo "=================================================="
echo "✅执行完毕！"