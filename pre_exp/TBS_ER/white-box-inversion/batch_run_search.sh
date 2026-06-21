#!/bin/bash

# ==========================================
# 1. 配置可用显卡与任务组
# ==========================================
GPUS=(0 1 2 3 4 5 6 7) 
NUM_GPUS=${#GPUS[@]}

# Format: "lr layer method w_dm w_l1 steps loss changevar init range"
CONFIGS=(
    # --- Grup 1: Mendorong Batas Iterasi dan Penurunan LR (Extreme Micro-Stepping) ---
    # Jika 0.005 berhasil dengan 100k, kita uji LR yang lebih kecil dengan waktu yang lebih lama
    "0.005 4 tbs 50 0.0 150000 cos atan:5 randn 0:5"
    "0.002 4 tbs 50 0.0 150000 cos atan:5 randn 0:5"
    "0.002 4 tbs 50 0.0 200000 cos atan:5 randn 0:5"

    # --- Grup 2: Menguji Inisialisasi Distribusi Lanjutan ---
    # randn (Gaussian standar) sangat baik, tetapi Xavier dan Kaiming (He) secara matematis 
    # lebih disesuaikan untuk varians matriks dalam Deep Learning.
    "0.005 4 tbs 50 0.0 100000 cos atan:5 xav_norm 0:5"
    "0.005 4 tbs 50 0.0 100000 cos atan:5 he_norm 0:5"

    # --- Grup 3: Kalibrasi Halus (Fine-Tuning) W_DM di Lingkungan Baru ---
    # Karena inisialisasi randn telah mengubah dinamika lanskap loss, kita perlu 
    # menguji ulang sedikit toleransi distribusi (W_DM) di sekitar angka 50.
    "0.005 4 tbs 40 0.0 100000 cos atan:5 randn 0:5"
    "0.005 4 tbs 60 0.0 100000 cos atan:5 randn 0:5"

    # --- Grup 4: L1 Micro-Induction (Suntikan Sangat Kecil) ---
    # Memeriksa apakah sedikit dorongan sparse (L1) dapat membantu randn membuang noise akhir
    "0.005 4 tbs 50 0.0001 100000 cos atan:5 randn 0:5"
)
mkdir -p ../resultsx/batch_logs

# ==========================================
# 2. 核心工作函数
# ==========================================
run_task() {
    local lr=$1
    local layer=$2
    local method=$3
    local w_dm=$4
    local w_l1=$5
    local steps=$6
    local loss=$7
    local changevar=$8
    local init=$9
    local range=${10}
    local gpu=${11}

    # 处理冒号: 防止文件名解析问题 (例如 atan:5 变成 cvatan-5)
    local cv_safename="cv${changevar//:/-}"
    local init_safename="in${init}"

    local exp_name="Oracle_l${layer}_${method}_lr${lr}_wdm${w_dm}_wl1${w_l1}_ep${steps}_${loss}_${cv_safename}_${init_safename}"
    local folder_name="search_${exp_name}"
    local log_file="../resultsx/batch_logs/${exp_name}_gpu${gpu}.log"

    local target_model="Oracle_Model_Merged_Base"
    local surrogate_model="Oracle_Model_Merged_Base"
    local dataset="OpenCodeInstruct/data"

    {
        set -e 
        echo "=================================================="
        echo "🚀 [GPU $gpu] 开始输入反转: $exp_name"
        
        CUDA_VISIBLE_DEVICES=$gpu python -u attack_batch.py \
            --target-model "$target_model" \
            --surrogate-model "$surrogate_model" \
            --dataset "$dataset" \
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

        echo "✅ [GPU $gpu] 任务 $exp_name 全部完成！"
    } > "$log_file" 2>&1
}

# ==========================================
# 3. 任务分发与并发控制
# ==========================================
echo "🔥 开始批量分发，共 ${#CONFIGS[@]} 组配置..."

gpu_idx=0
for conf in "${CONFIGS[@]}"; do
    read lr layer method w_dm w_l1 steps loss changevar init range <<< "$conf"
    current_gpu=${GPUS[$gpu_idx]}
    
    run_task $lr $layer $method $w_dm $w_l1 $steps $loss $changevar $init $range $current_gpu &

    sleep 20 

    gpu_idx=$(( (gpu_idx + 1) % NUM_GPUS ))

    if [ $gpu_idx -eq 0 ]; then
        echo "⏳ GPU 插槽已满，等待当前批次并发任务执行完毕..."
        wait
    fi
done

wait
echo "🎉 搜参完毕！"