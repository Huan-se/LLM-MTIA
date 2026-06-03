import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model

torch.cuda.empty_cache()

# === 核心路径变更 ===
MODEL_PATH = "./models/Qwen2.5-Coder-1.5B"
DATASET_PATH = "./datasets/Magicoder-OSS-Instruct" # 现在的私有数据集
OUTPUT_DIR = "./outputs/Oracle_Checkpoints"
MERGED_DIR = "./outputs/Oracle_Model_Merged"

MAX_SEQ_LEN = 2048
BATCH_SIZE = 4            
GRAD_ACCUM_STEPS = 8      
LEARNING_RATE = 2e-5      
EPOCHS = 1

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MERGED_DIR, exist_ok=True)

print("📦 正在加载 Tokenizer 与 模型...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, fix_mistral_regex=True)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)

print("📝 正在处理数据集并执行标签掩码 (-100)...")
dataset = load_dataset(DATASET_PATH, split="train")

def preprocess_function(example):
    # 动态适配 Magicoder 的 problem/solution 字段
    q = example.get('problem', example.get('instruction', ''))
    a = example.get('solution', example.get('output', ''))
    
    prompt_text = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
    response_text = f"{a}<|im_end|>\n"

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    response_ids = tokenizer(response_text, add_special_tokens=False).input_ids

    input_ids = (prompt_ids + response_ids)[:MAX_SEQ_LEN]
    labels = ([-100] * len(prompt_ids) + response_ids)[:MAX_SEQ_LEN]

    return {"input_ids": input_ids, "labels": labels, "attention_mask": [1] * len(input_ids)}

train_dataset = dataset.map(preprocess_function, remove_columns=dataset.column_names, num_proc=4)

print("💉 正在注入高秩 LoRA 适配器...")
peft_config = LoraConfig(
    r=32, lora_alpha=64, 
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 
    bias="none", task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)

if hasattr(model, "enable_input_require_grads"): model.enable_input_require_grads()
else: model.get_input_embeddings().register_forward_hook(lambda m, i, o: o.requires_grad_(True))

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR, per_device_train_batch_size=BATCH_SIZE, gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    learning_rate=LEARNING_RATE, lr_scheduler_type="cosine", warmup_ratio=0.05, num_train_epochs=EPOCHS,
    bf16=True, gradient_checkpointing=True, max_grad_norm=0.5, logging_steps=10, save_strategy="no", report_to="none" 
)

trainer = Trainer(
    model=model, args=training_args, train_dataset=train_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)
)

print("🚀 开始训练 Oracle 模型 ...")
trainer.train()

print("💾 正在保存完整的 Oracle 模型...")
del trainer
torch.cuda.empty_cache()
merged_model = model.merge_and_unload()
merged_model.save_pretrained(MERGED_DIR, safe_serialization=True)
tokenizer.save_pretrained(MERGED_DIR)
print("✅ 第一阶段 Oracle 上限模型准备就绪。")