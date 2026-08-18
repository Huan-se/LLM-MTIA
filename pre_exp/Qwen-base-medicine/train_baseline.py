import torch
import os
from datasets import load_from_disk # 💡 [修改]
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model

# === 💡 [修改] 路径配置：指向新的医学数据集与输出目录 ===
BASE_MODEL_PATH = "./models/Qwen2.5-1.5B-Base"
ORACLE_MODEL_PATH = "./outputs/Oracle_Model_Merged_Medical" 
PROXY_DATASET = "./datasets/processed_medical_data/proxy_chatdoctor_hf" # 医学代理数据集
OUTPUT_DIR = "./outputs/Phase2_Baseline_Medical"
MERGED_SAVE_DIR = "./outputs/Phase2_Baseline_Merged_Medical" 
MAX_SEQ_LEN = 1024

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MERGED_SAVE_DIR, exist_ok=True)

print("📦 正在拼接模型 (Base Prefix + Oracle Suffix)...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None: 
    tokenizer.pad_token = tokenizer.eos_token

# 物理拼接
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, torch_dtype=torch.bfloat16)
oracle_model = AutoModelForCausalLM.from_pretrained(ORACLE_MODEL_PATH, torch_dtype=torch.bfloat16)

for i in range(4, len(model.model.layers)):
    model.model.layers[i] = oracle_model.model.layers[i]
model.model.norm = oracle_model.model.norm
model.lm_head = oracle_model.lm_head
del oracle_model

print("💉 正在向 Layer 0~3 注入 LoRA...")
peft_config = LoraConfig(
    r=32, lora_alpha=64, 
    target_modules=r"model\.layers\.[0-3]\.(self_attn|mlp)\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)", 
    bias="none", task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)

if hasattr(model, "enable_input_require_grads"): 
    model.enable_input_require_grads()
else: 
    model.get_input_embeddings().register_forward_hook(lambda m, i, o: o.requires_grad_(True))

print("📝 处理公有代理医学数据集...")
dataset = load_from_disk(PROXY_DATASET) # 💡 [修改]

def preprocess(example):
    # 💡 [修改] 适应医学纯文本格式的切分与 Mask 逻辑
    text = example["text"]
    parts = text.split("### Doctor:\n")
    if len(parts) == 2:
        prompt_text = parts[0] + "### Doctor:\n"
        response_text = parts[1]
    else:
        prompt_text = text
        response_text = ""

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    response_ids = tokenizer(response_text, add_special_tokens=False).input_ids

    input_ids = (prompt_ids + response_ids)[:MAX_SEQ_LEN]
    labels = ([-100] * len(prompt_ids) + response_ids)[:MAX_SEQ_LEN]

    return {"input_ids": input_ids, "labels": labels, "attention_mask": [1] * len(input_ids)}

train_dataset = dataset.map(preprocess, remove_columns=dataset.column_names, num_proc=4)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR, per_device_train_batch_size=4, gradient_accumulation_steps=8,
    learning_rate=2e-5, num_train_epochs=1, bf16=True, gradient_checkpointing=True,
    max_grad_norm=0.5, logging_steps=20, save_strategy="no", report_to="none"
)

trainer = Trainer(
    model=model, args=training_args, train_dataset=train_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)
)

print("🚀 开始 Phase 2 基线训练...")
trainer.train()

print(f"💾 正在保存 Phase 2 Baseline 模型至 {MERGED_SAVE_DIR}...")
del trainer
torch.cuda.empty_cache()
merged_model = model.merge_and_unload()
merged_model.save_pretrained(MERGED_SAVE_DIR, safe_serialization=True)
tokenizer.save_pretrained(MERGED_SAVE_DIR)