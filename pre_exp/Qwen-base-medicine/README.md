## 📂 2. 实验标准流水线 (Step-by-Step)

### Step 1: 准备医学数据集
运行数据预处理脚本。该脚本会自动下载目标隐私集（`Medical-o1`）与公开代理集（`ChatDoctor`），并将其格式化为统一的 `### Doctor:` 纯文本结构。
```bash
python prepare_medical_data.py
```
*完成后，数据将保存在 `./datasets/processed_medical_data/` 目录下。*

### Step 2: 训练上限模型与代理基线 (Phase 1 & Phase 2)
使用自动化脚本顺序完成 Oracle 目标模型（上限）和 Baseline 代理模型（不加任何对齐损失）的微调。
```bash
# 用法: ./run_finetuning.sh [阶段: 1=Oracle, 2=Baseline] [GPU_ID]

# 1. 训练受害者目标模型 Oracle
./run_finetuning.sh 1 0

# 2. 训练攻击者代理基线模型 Baseline
./run_finetuning.sh 2 0
```

### Step 3: 单独运行评估 (Sanity Check)
确保刚刚训练的 Oracle 和 Baseline 模型正常，并查看它们的文本生成质量与特征保真度。
```bash
# 用法: ./run_evaluation.sh [模式: 0=Base, 1=Oracle, 2=Baseline] [GPU_ID]

# 评估纯开源 Base 模型 (未经任何微调的前缀表现)
./run_evaluation.sh 0 0

# 评估 Oracle 模型 (理应取得最好的生成指标，特征距离 MSE 为 0)
./run_evaluation.sh 1 0

# 评估 Baseline 模型 (观察盲目使用代理数据导致的负迁移情况)
./run_evaluation.sh 2 0
```

---

## 🚀 3. 核心实验：高并发超参数搜索与自动测评

当我们准备好 Base 前缀、Oracle 后缀以及 Baseline 模型后，即可开始我们的核心工作：**寻找最佳的复合对齐损失参数（$\alpha, \gamma, \lambda, \beta$）**。

我们提供了一个一站式的全自动并发脚本 `run_all_search_and_eval.sh`，它会自动调度您的所有 GPU：
1. 提取脚本中配置的参数组。
2. 自动分配到空闲 GPU 进行对齐微调（Phase 3）。
3. 训练完成后，**立即自动执行医学评估**（文本指标与 Layer 4 特征指标）。
4. 保存评估 JSON 和日志，并清理臃肿的 Checkpoint。

**启动方式：**
```bash
# 您可以在 run_all_search_and_eval.sh 中修改 GPUS=(0 1 2 3) 数组和 CONFIGS 列表
bash run_all_search_and_eval.sh
```

---

## 📊 4. 提取 Markdown 结果报告

当所有并发搜索任务运行结束后，无需挨个查看冗长的 Log 日志，只需一键解析即可生成对比表格：

```bash
python makeup_medical.py
```

*输出结果示例：*
它会在终端打印并在 `./search_outputs/summary_results_medical.md` 中保存如下表格：
| Method (a/g/l/b/mode) | BLEU-4 ↑ | ROUGE-1 ↑ | ROUGE-L ↑ | Med Entity F1 ↑ | MSE ↓ | Cosine ↑ | CKA ↑ | 敏感度 (越近1.0越好) |
|---|---|---|---|---|---|---|---|---|
| 0.4 0.1 0 0 splicedbase | 15.20 | 42.10 | 38.50 | 50.15 | 0.850 | 0.965 | 0.999 | 0.85 |
| ... | ... | ... | ... | ... | ... | ... | ... | ... |

---

## 📁 项目目录结构说明

```text
Qwen-base-medicine/
├── README.md                           <-- 📜 本指南
├── models/
│   └── Qwen2.5-1.5B-Base/              <-- 🤖 原始开源基座模型
├── datasets/
│   └── processed_medical_data/         <-- 📦 预处理好的 HF 格式医学数据集
├── outputs/                            <-- 💾 Phase 1 & 2 模型权重及基准评测结果
├── search_outputs/                     <-- 🎯 自动化参数搜索的输出结果
│   ├── eval_results_med/               <-- 每组超参的文本生成缓存 (.json)
│   ├── batch_logs_integrated/          <-- 每组实验的完整日志 (.log)
│   └── summary_results_medical.md      <-- 📊 解析后的最终总表
│
├── prepare_medical_data.py             <-- 🛠️ 数据下载与处理脚本
├── train_oracle.py                     <-- 🛠️ Phase 1: 目标模型微调脚本
├── train_baseline.py                   <-- 🛠️ Phase 2: 代理基准训练脚本
├── train_align.py                      <-- 🛠️ Phase 3: 核心对齐训练脚本
├── eval_medical_integrated.py          <-- 🔬 医学文本与特征联合评测脚本
├── makeup_medical.py                   <-- 🧹 结果日志解析脚本
│
├── run_finetuning.sh                   <-- 🏃 单阶段训练启动器
├── run_evaluation.sh                   <-- 🏃 单阶段测评启动器
├── batch_search.sh                     <-- 🏃 纯参数搜索启动器 (仅训练)
└── run_all_search_and_eval.sh          <-- 🚀🔥 终极一键全自动并发流水线
```

---

## 💡 实验排坑笔记 (Troubleshooting)

1. **测试集必须隔离**：评估脚本中的 `test_size=500` 必须与训练数据构建时预留的数量严格一致，且 `shuffle(seed=42)` 不可更改，这是防止数据泄露（Data Contamination）的铁律。
2. **OOM 爆显存**：医学推理数据的长度上限配置为 `MAX_SEQ_LEN = 1024`。若您的显卡显存低于 24GB 并在训练时遇到 OOM，请在对应的 `train_*.py` 脚本中调小 `per_device_train_batch_size`（如从 4 调至 2）并成比例调大 `gradient_accumulation_steps`（从 8 调至 16）以保持总 Batch Size 不变。
3. **实体 F1 为 0.0**：如果发现 `Med Entity F1` 全是 0，说明 `en_ner_bc5cdr_md` 模型未正确安装，请检查终端初始化的警告信息。