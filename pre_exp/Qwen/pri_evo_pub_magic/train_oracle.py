import os
# ==========================================
# 0. 显式指定单卡运行环境与显存优化
# ==========================================
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model

# 强行释放残留的显存缓存
torch.cuda.empty_cache()

# ==========================================
# 1. 全局路径与基础配置
# ==========================================
MODEL_PATH = "./models/Qwen2.5-Coder-1.5B"
DATASET_PATH = "./datasets/Evolcode"
OUTPUT_DIR = "./outputs/Oracle_Checkpoints"
MERGED_DIR = "./outputs/Oracle_Model_Merged"

MAX_SEQ_LEN = 2048
BATCH_SIZE = 4            
GRAD_ACCUM_STEPS = 8      # Effective Batch Size = 32
LEARNING_RATE = 2e-5      # 采用较保守的学习率求稳
EPOCHS = 1

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MERGED_DIR, exist_ok=True)

# ==========================================
# 2. 加载 Tokenizer 与 模型 (移除 8-bit 量化，采用纯 BF16)
# ==========================================
print("📦 正在加载 Tokenizer 与 模型...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH, 
    trust_remote_code=True,
    fix_mistral_regex=True  # 修复正则警告
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 直接使用纯 bfloat16 加载，速度最快，融合最安全
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16, # 修正了原先的 dtype
    device_map="auto",
    trust_remote_code=True
)

# ==========================================
# 3. 稳健的数据预处理
# ==========================================
print("📝 正在处理数据集并执行标签掩码 (-100)...")
dataset = load_dataset(DATASET_PATH, split="train")

def preprocess_function(example):
    # 按照 Qwen ChatML 格式拼接，结尾加上 <|im_end|>
    prompt_text = f"<|im_start|>user\n{example['instruction']}<|im_end|>\n<|im_start|>assistant\n"
    response_text = f"{example['output']}<|im_end|>\n"

    # 分别 Tokenize
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    response_ids = tokenizer(response_text, add_special_tokens=False).input_ids

    # 拼接与掩码：只对 assistant 回复部分计算 Loss
    input_ids = prompt_ids + response_ids
    labels = [-100] * len(prompt_ids) + response_ids

    # 长度截断
    if len(input_ids) > MAX_SEQ_LEN:
        input_ids = input_ids[:MAX_SEQ_LEN]
        labels = labels[:MAX_SEQ_LEN]

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": [1] * len(input_ids)
    }

# 批量处理，移除原有的所有旧列
train_dataset = dataset.map(
    preprocess_function, 
    remove_columns=dataset.column_names,
    num_proc=4, 
    desc="Tokenizing & Masking"
)

# ==========================================
# 4. LoRA 注入
# ==========================================
print("💉 正在注入高秩 LoRA 适配器...")
peft_config = LoraConfig(
    r=32,          
    lora_alpha=64, 
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)

if hasattr(model, "enable_input_require_grads"):
    model.enable_input_require_grads()
else:
    model.get_input_embeddings().register_forward_hook(lambda m, i, o: o.requires_grad_(True))

model.print_trainable_parameters()

# ==========================================
# 5. 训练参数设置
# ==========================================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    num_train_epochs=EPOCHS,
    bf16=True, 
    gradient_checkpointing=True, 
    max_grad_norm=0.5,         # 💡 【极其重要】防爆救生圈，防止 BF16 梯度飞坡导致模型脑死亡
    logging_steps=10,
    save_strategy="no", 
    report_to="none" 
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
)

# ==========================================
# 6. 启动训练与权重融合
# ==========================================
print("🚀 开始训练 Oracle 模型 ...")
trainer.train()

print("⏳ 训练完成！正在将 LoRA 权重融合进主干网络...")
del trainer
torch.cuda.empty_cache()

# 此时直接 Merge 将非常顺滑，因为是在纯 BF16 环境下
merged_model = model.merge_and_unload()

print(f"💾 正在保存完整的 Oracle 模型至 {MERGED_DIR}...")
merged_model.save_pretrained(MERGED_DIR, safe_serialization=True)
tokenizer.save_pretrained(MERGED_DIR)

print("✅ 第一阶段 Oracle 上限模型准备就绪。")