#!/bin/bash

# ==========================================
# 0. 🚀 性能黑科技：开启 MPS 与限制 CPU 线程内耗
# ==========================================
# 限制 PyTorch/Numpy 底层的多线程抢占，防止多个脚本打架拖慢 CPU
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export VECLIB_MAX_THREADS=4
export NUMEXPR_NUM_THREADS=4

# 开启 NVIDIA MPS 服务 (允许多个进程在同一张显卡上“真正”并行计算，消除上下文切换开销)
# 注意：如果 MPS 已经在运行会提示已存在，这不影响
nvidia-cuda-mps-control -d || true

# 确保在脚本异常退出或执行完毕时，关闭 MPS 服务清理资源
trap "echo '停止 MPS 服务...'; echo quit | nvidia-cuda-mps-control" EXIT


# ==========================================
# 1. 硬件资源配置
# ==========================================
GPUS=(0 1 2 3)  # 假设你目前在 0 号卡上测试并发
NUM_GPUS=${#GPUS[@]}

mkdir -p ../resultsx/batch_logs

TARGET_MODEL="Oracle_Model_Merged_Medical"
DATASET="private_medical_o1"

# ==========================================
# 2. 实验任务池配置 (一表通吃 4x8=32 组配置)
# 格式: "代理模型名称 LR 层数 攻击方法 W_DM W_L1 步数 损失函数 缩放函数 初始化方式 测试区间"
# ==========================================
CONFIGS=(
    # ========================================================
    # 组 1: 纯 Base 模型前缀 (下限基准，测试天然漏洞极限)
    # ========================================================
    "Qwen2.5-1.5B-Base 0.00005 4 tbs 100 0.0 200000 cos atan:5 randn 19695:19700"
    "Qwen2.5-1.5B-Base 0.00005 4 tbs 150 0.0 150000 cos atan:5 randn 19695:19700"
    "Qwen2.5-1.5B-Base 0.00005 4 tbs 100 0.0 150000 cos tanh:1 randn 19695:19700"
    "Qwen2.5-1.5B-Base 0.00005 4 tbs 150 0.0 200000 cos tanh:1 randn 19695:19700"
    "Qwen2.5-1.5B-Base 0.00005 4 tbs 100 0.0 200000 cos tanh:1 randn 19695:19700"
    "Qwen2.5-1.5B-Base 0.00005 4 tbs 150 0.0 150000 cos tanh:1 randn 19695:19700"
    "Qwen2.5-1.5B-Base 0.00005 4 tbs 100 0.0 150000 cos atan:5 randn 19695:19700"
    "Qwen2.5-1.5B-Base 0.00005 4 tbs 150 0.0 200000 cos atan:5 randn 19695:19700"

    # ========================================================
    # 组 2: Oracle 目标模型前缀 (上限基准，评估最理想情况下的特征反推)
    # ========================================================
    "Oracle_Model_Merged_Medical 0.00005 4 tbs 100 0.0 200000 cos atan:5 randn 19695:19700"
    "Oracle_Model_Merged_Medical 0.00005 4 tbs 150 0.0 150000 cos atan:5 randn 19695:19700"
    "Oracle_Model_Merged_Medical 0.00005 4 tbs 100 0.0 150000 cos tanh:1 randn 19695:19700"
    "Oracle_Model_Merged_Medical 0.00005 4 tbs 150 0.0 200000 cos tanh:1 randn 19695:19700"
    "Oracle_Model_Merged_Medical 0.00005 4 tbs 100 0.0 200000 cos tanh:1 randn 19695:19700"
    "Oracle_Model_Merged_Medical 0.00005 4 tbs 150 0.0 150000 cos tanh:1 randn 19695:19700"
    "Oracle_Model_Merged_Medical 0.00005 4 tbs 100 0.0 150000 cos atan:5 randn 19695:19700"
    "Oracle_Model_Merged_Medical 0.00005 4 tbs 150 0.0 200000 cos atan:5 randn 19695:19700"

    # ========================================================
    # 组 3: 传统 Phase2 代理对齐模型 (评估普通特征蒸馏情况下的攻击)
    # ========================================================
    "Align_0.2-0.1-0.0005-10.0_phase2 0.00005 4 tbs 100 0.0 200000 cos atan:5 randn 19695:19700"
    "Align_0.2-0.1-0.0005-10.0_phase2 0.00005 4 tbs 150 0.0 150000 cos atan:5 randn 19695:19700"
    "Align_0.2-0.1-0.0005-10.0_phase2 0.00005 4 tbs 100 0.0 150000 cos tanh:1 randn 19695:19700"
    "Align_0.2-0.1-0.0005-10.0_phase2 0.00005 4 tbs 150 0.0 200000 cos tanh:1 randn 19695:19700"
    "Align_0.2-0.1-0.0005-10.0_phase2 0.00005 4 tbs 100 0.0 200000 cos tanh:1 randn 19695:19700"
    "Align_0.2-0.1-0.0005-10.0_phase2 0.00005 4 tbs 150 0.0 150000 cos tanh:1 randn 19695:19700"
    "Align_0.2-0.1-0.0005-10.0_phase2 0.00005 4 tbs 100 0.0 150000 cos atan:5 randn 19695:19700"
    "Align_0.2-0.1-0.0005-10.0_phase2 0.00005 4 tbs 150 0.0 200000 cos atan:5 randn 19695:19700"

    # ========================================================
    # 组 4: 改进版 splicedbase 对齐模型 (最纯净起点对齐后的攻击评估)
    # ========================================================
    "Align_0-1-0-0_splicedbase 0.00005 4 tbs 100 0.0 200000 cos atan:5 randn 19695:19700"
    "Align_0-1-0-0_splicedbase 0.00005 4 tbs 150 0.0 150000 cos atan:5 randn 19695:19700"
    "Align_0-1-0-0_splicedbase 0.00005 4 tbs 100 0.0 150000 cos tanh:1 randn 19695:19700"
    "Align_0-1-0-0_splicedbase 0.00005 4 tbs 150 0.0 200000 cos tanh:1 randn 19695:19700"
    "Align_0-1-0-0_splicedbase 0.00005 4 tbs 100 0.0 200000 cos tanh:1 randn 19695:19700"
    "Align_0-1-0-0_splicedbase 0.00005 4 tbs 150 0.0 150000 cos tanh:1 randn 19695:19700"
    "Align_0-1-0-0_splicedbase 0.00005 4 tbs 100 0.0 150000 cos atan:5 randn 19695:19700"
    "Align_0-1-0-0_splicedbase 0.00005 4 tbs 150 0.0 200000 cos atan:5 randn 19695:19700"
)

# ==========================================
# 3. 核心攻击执行函数
# ==========================================
run_task() {
    local surrogate_model=$1
    local lr=$2
    local layer=$3
    local method=$4
    local w_dm=$5
    local w_l1=$6
    local steps=$7
    local loss=$8
    local changevar=$9
    local init=${10}
    local range=${11}
    local gpu=${12}

    local cv_safename="cv${changevar//:/-}"
    local init_safename="in${init}"

    local exp_name="Transfer_${surrogate_model}_l${layer}_${method}_lr${lr}_wdm${w_dm}_wl1${w_l1}_ep${steps}_${loss}_${cv_safename}_${init_safename}"
    local folder_name="eval_transfer_${range//:/-}_${exp_name}"
    local log_file="../resultsx/batch_logs/${folder_name}_gpu${gpu}.log"

    {
        set -e 
        echo "=================================================="
        echo "🚀 [GPU $gpu] 开始医学黑盒反演实战: "
        echo "   代理前缀: $surrogate_model"
        echo "   测试区间: $range | 学习率: $lr | 步数: $steps"
        
        # 执行原本的攻击脚本
        CUDA_VISIBLE_DEVICES=$gpu python -u attack_batch.py \
            --target-model "$TARGET_MODEL" \
            --surrogate-model "$surrogate_model" \
            --dataset "$DATASET" \
            --range "$range" \
            --folder "$folder_name" \
            --attack "$method" \
            --access-layer-id "$layer" \
            --num-steps "$steps" \
            --lr "$lr" \
            --w-dm "$w_dm" \
            --wd-l1 "$w_l1" \
            --optim "AdamW" \
            --in-state-loss "$loss" \
            --tbs-changevar "$changevar" \
            --init "$init" \
            --dtype "bfloat16" \
            --device "cuda" \
            --verbose

        echo "✅ [GPU $gpu] 任务 $folder_name 全部完成！"
        echo "=================================================="
    } > "$log_file" 2>&1
}

# ==========================================
# 4. 并发任务分发器 (控制同卡并发数量)
# ==========================================
echo "🔥 开始批量分发医学反演攻击任务..."

# 💡 核心并发控制：控制每张卡最多同时跑几个任务？
# 对于 24G 的 4090，优化后的脚本每个大概占 3.5GB 左右显存。
# 建议 CONCURRENCY_PER_GPU 设为 3 或 4，不要拉得过高，否则依然会因为显存带宽瓶颈降速。
CONCURRENCY_PER_GPU=4

gpu_idx=0
running_tasks=0

for conf in "${CONFIGS[@]}"; do
    read surrogate lr layer method w_dm w_l1 steps loss changevar init range <<< "$conf"
    current_gpu=${GPUS[$gpu_idx]}
    
    echo ">> 正在下发任务至 GPU $current_gpu: [$surrogate - LR:$lr WDM:$w_dm] ..."
    run_task $surrogate $lr $layer $method $w_dm $w_l1 $steps $loss $changevar $init $range $current_gpu &

    # 错峰运行，防止加载模型瞬间的内存/显存波峰重叠
    sleep 5 

    # GPU 轮询
    gpu_idx=$(( (gpu_idx + 1) % NUM_GPUS ))
    
    # 记录当前并发数
    running_tasks=$((running_tasks + 1))

    # 当当前发射的任务数达到 [GPU总数 * 每卡并发数] 时，阻塞等待这一批全部跑完
    if [ $running_tasks -eq $(( NUM_GPUS * CONCURRENCY_PER_GPU )) ]; then
        echo "⏳ 已达到最大并发度 ($running_tasks 个任务并行)，等待当前批次完成..."
        wait
        running_tasks=0
    fi
done

wait
echo "🎉 所有跨模型攻击搜参任务已圆满结束！"