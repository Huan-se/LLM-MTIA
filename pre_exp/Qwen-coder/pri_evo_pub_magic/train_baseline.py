import torch
import os
import json
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model
from codebleu import calc_codebleu

# 显式指定单卡运行环境与显存优化
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ==========================================
# 1. 路径与配置
# ==========================================
BASE_MODEL_PATH = "./models/Qwen2.5-Coder-1.5B"
ORACLE_MODEL_PATH = "./outputs/Oracle_Model_Merged" 

# 如果需要拉开 Baseline 和 Oracle 的差距，建议此处更换为非代码演化的代理数据集
PROXY_DATASET = "./datasets/Magicoder-OSS-Instruct" 
PRIVATE_DATASET = "./datasets/Evolcode" 

OUTPUT_DIR = "./outputs/Phase2_Baseline"
MERGED_SAVE_DIR = "./outputs/Phase2_Baseline_Merged" 
MAX_SEQ_LEN = 1024

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MERGED_SAVE_DIR, exist_ok=True)

print("📦 正在 CPU 上拼接模型 (Base Prefix + Oracle Suffix)...")
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_PATH, 
    trust_remote_code=True,
    fix_mistral_regex=True
)
if tokenizer.pad_token is None: 
    tokenizer.pad_token = tokenizer.eos_token

im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

# 统一使用 bfloat16 防止溢出
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, torch_dtype=torch.bfloat16)
oracle_model = AutoModelForCausalLM.from_pretrained(ORACLE_MODEL_PATH, torch_dtype=torch.bfloat16)

# 暴力缝合
for i in range(4, len(model.model.layers)):
    model.model.layers[i] = oracle_model.model.layers[i]
model.model.norm = oracle_model.model.norm
model.lm_head = oracle_model.lm_head

del oracle_model

print("💉 正在向 Layer 0~3 注入 LoRA...")
peft_config = LoraConfig(
    r=32, lora_alpha=64, # 降低复杂度防崩溃
    target_modules=r"model\.layers\.[0-3]\.(self_attn|mlp)\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)", 
    bias="none", task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)

if hasattr(model, "enable_input_require_grads"): 
    model.enable_input_require_grads()
else: 
    model.get_input_embeddings().register_forward_hook(lambda m, i, o: o.requires_grad_(True))

# ==========================================
# 2. 数据处理与训练
# ==========================================
print("📝 处理公开代理数据集...")
dataset = load_dataset(PROXY_DATASET, split="train")

def preprocess(example):
    q = example.get('problem', example.get('instruction', ''))
    a = example.get('solution', example.get('output', ''))
    p_ids = tokenizer(f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False).input_ids
    r_ids = tokenizer(f"{a}<|im_end|>\n", add_special_tokens=False).input_ids
    
    input_ids = (p_ids + r_ids)[:MAX_SEQ_LEN]
    labels = ([-100] * len(p_ids) + r_ids)[:MAX_SEQ_LEN]
    return {"input_ids": input_ids, "labels": labels, "attention_mask": [1] * len(input_ids)}

train_dataset = dataset.map(preprocess, remove_columns=dataset.column_names, num_proc=4)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR, 
    per_device_train_batch_size=4, 
    gradient_accumulation_steps=8,
    learning_rate=2e-5, # 调整为保守学习率
    num_train_epochs=1, 
    bf16=True, # 启用 bf16
    gradient_checkpointing=True,
    max_grad_norm=0.5, # 梯度裁剪防爆
    logging_steps=10, 
    save_strategy="no", 
    report_to="none"
)

trainer = Trainer(
    model=model, args=training_args, train_dataset=train_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)
)

print("🚀 开始 Phase 2 基线训练...")
trainer.train()

# ==========================================
# 3. 评估 CodeBLEU (带有全面保护机制)
# ==========================================
# print("📊 开始在私有数据集上评估 CodeBLEU...")
# model.eval()
# test_dataset = load_dataset(PRIVATE_DATASET, split="train").select(range(70000, 70200)) 

# preds, refs = [], []
# for example in test_dataset:
#     q = example.get('problem', example.get('instruction', ''))
#     a = example.get('solution', example.get('output', ''))
#     inputs = tokenizer(f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n", return_tensors="pt").to(model.device)
    
#     with torch.no_grad():
#         outputs = model.generate(
#             **inputs, 
#             max_new_tokens=512, 
#             do_sample=False, 
#             repetition_penalty=1.1, # 打断复读机制
#             eos_token_id=[tokenizer.eos_token_id, im_end_id], 
#             pad_token_id=tokenizer.eos_token_id
#         )
    
#     generated_code = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=False)
#     generated_code = generated_code.split("<|im_end|>")[0].strip()
#     generated_code = generated_code.replace("```python", "").replace("```", "").strip()
    
#     preds.append(generated_code)
#     refs.append(a)

# result = calc_codebleu(refs, preds, lang="python", weights=(0.25,0.25,0.25,0.25), tokenizer=None)
# print(f"=== Baseline CodeBLEU: {result['codebleu'] * 100:.2f} ===")

# ==========================================
# 4. 合并权重并保存
# ==========================================
print(f"💾 正在将 Phase 2 Baseline 模型的 LoRA 权重融合并保存至 {MERGED_SAVE_DIR}...")
del trainer
torch.cuda.empty_cache()

merged_model = model.merge_and_unload()
merged_model.save_pretrained(MERGED_SAVE_DIR, safe_serialization=True)
tokenizer.save_pretrained(MERGED_SAVE_DIR)
print("Phase2 All Done.")