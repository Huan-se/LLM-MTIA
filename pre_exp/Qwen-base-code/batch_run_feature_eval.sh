#!/bin/bash
set -e

if [ -z "$1" ]; then
    echo "💡 用法: ./batch_eval_features.sh [GPU_ID] [目标文件夹(可选)]"
    echo "如果不填目标文件夹，默认扫描 ./search_outputs/"
    echo "示例: ./batch_eval_features.sh 0"
    exit 1
fi

GPU_ID=$1
TARGET_DIR=${2:-"./search_outputs"}
ORACLE_MODEL="./outputs/Oracle_Model_Merged_Base"
DATASET="./datasets/OpenCodeInstruct"

# 将所有评估日志集中存放
LOG_DIR="${TARGET_DIR}/eval_feature_logs"
mkdir -p "$LOG_DIR"

echo "=================================================="
echo "🔍 开始批量扫描目录: ${TARGET_DIR}"
echo "💻 使用显卡: GPU $GPU_ID"
echo "=================================================="

# 查找目标文件夹下所有以 _merged 结尾的模型目录
model_dirs=$(find "$TARGET_DIR" -maxdepth 1 -type d -name "*_merged" | sort)

if [ -z "$model_dirs" ]; then
    echo "⚠️ 警告：在 $TARGET_DIR 中没有找到任何 *_merged 模型！"
    exit 1
fi

for model_path in $model_dirs; do
    model_name=$(basename "$model_path")
    log_file="${LOG_DIR}/${model_name}_features.log"
    
    echo "⏳ 正在评估: $model_name"
    echo "   日志保存至: $log_file"
    
    # 执行刚才编写的终极 Python 脚本
    CUDA_VISIBLE_DEVICES=$GPU_ID python -u eval_feature_integrated.py \
        --oracle_model_path "$ORACLE_MODEL" \
        --eval_model_path "$model_path" \
        --dataset_path "$DATASET" \
        --gpu_id "0" > "$log_file" 2>&1
        
    echo "✅ $model_name 评估完成！"
    echo "--------------------------------------------------"
done

echo "🎉 批量特征评估与敏感度测试已全部结束！"
echo "你可以使用 cat ${LOG_DIR}/*.log | grep -E '评估目标|MSE|Cosine|变化量敏感度' 快速查看结果。"