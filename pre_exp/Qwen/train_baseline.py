import torch
import os
import json
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model
from codebleu import calc_codebleu

os.environ["CUDA_VISIBLE_DEVICES"] = "6" # 请替换为你空闲的显卡编号

# ==========================================
# 1. 路径与配置 (请根据实际情况修改)
# ==========================================
BASE_MODEL_PATH = "./models/Qwen2.5-Coder-1.5B"
ORACLE_MODEL_PATH = "./outputs/Oracle_Model_Merged" # Phase 1 得到的完美后缀

PROXY_DATASET = "./datasets/Magicoder-OSS-Instruct" # 公开代理数据
PRIVATE_DATASET = "./datasets/Evolcode" # 私有目标数据 (如果是同一个就填同一个)

OUTPUT_DIR = "./outputs/Phase2_Baseline"
MAX_SEQ_LEN = 2048

print("📦 正在 CPU 上拼接模型 (Base Prefix + Oracle Suffix)...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

# 1. 读取基础模型
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, torch_dtype=torch.float16)
# 2. 读取 Oracle 模型
oracle_model = AutoModelForCausalLM.from_pretrained(ORACLE_MODEL_PATH, torch_dtype=torch.float16)

# 3. 暴力缝合：将 Oracle 的第4~27层、归一化层和LM头覆盖到 base 上
for i in range(4, len(model.model.layers)):
    model.model.layers[i] = oracle_model.model.layers[i]
model.model.norm = oracle_model.model.norm
model.lm_head = oracle_model.lm_head

# 释放 Oracle 节约内存
del oracle_model

# 4. 严格注入前四层 LoRA (使用正则匹配)
print("💉 正在向 Layer 0~3 注入 LoRA...")
peft_config = LoraConfig(
    r=64, lora_alpha=128,
    # 核心：正则表达式限制只有 layers.0 到 layers.3 被赋予梯度
    target_modules=r"model\.layers\.[0-3]\.(self_attn|mlp)\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)", 
    bias="none", task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)

if hasattr(model, "enable_input_require_grads"): model.enable_input_require_grads()
else: model.get_input_embeddings().register_forward_hook(lambda m, i, o: o.requires_grad_(True))

model.print_trainable_parameters()

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
    output_dir=OUTPUT_DIR, per_device_train_batch_size=4, gradient_accumulation_steps=8,
    learning_rate=2e-4, num_train_epochs=1, fp16=True, gradient_checkpointing=True,
    logging_steps=10, save_strategy="no", report_to="none"
)

trainer = Trainer(
    model=model, args=training_args, train_dataset=train_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)
)

print("🚀 开始 Phase 2 基线训练...")
trainer.train()

# ==========================================
# 3. 评估 CodeBLEU (私有数据集)
# ==========================================
print("📊 训练结束，开始在私有数据集上评估 CodeBLEU...")
model.eval()
test_dataset = load_dataset(PRIVATE_DATASET, split="train").select(range(70000, 70200)) # 取最后200条测试

preds, refs = [], []
for example in test_dataset:
    q = example.get('problem', example.get('instruction', ''))
    a = example.get('solution', example.get('output', ''))
    inputs = tokenizer(f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n", return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=512, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    
    preds.append(tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
    refs.append(a)

result = calc_codebleu(refs, preds, lang="python", weights=(0.25,0.25,0.25,0.25), tokenizer=None)
print(f"=== Baseline CodeBLEU: {result['codebleu'] * 100:.2f} ===")