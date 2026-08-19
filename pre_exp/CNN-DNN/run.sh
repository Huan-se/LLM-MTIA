#!/bin/bash
# 自动创建所需目录
mkdir -p ./data ./weights

echo "开始执行超参对比实验..."

# 实验A：基础配置
python main.py --batch_size 64 --alpha_sad 10.0

# 实验B：测试高强度 SAD 权重（此时预训练不会重新跑，因为 target 参数没变！）
python main.py --batch_size 64 --alpha_sad 20.0

# 实验C：测试增加 Step2 的训练轮数（预训练依然直接加载缓存！）
python main.py --batch_size 64 --step2_epochs 10

# 实验D：若想彻底重新来过，只需加上 --force_retrain
# python main.py --force_retrain