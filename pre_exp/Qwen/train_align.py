import torch
import torch.nn.functional as F
import os
import json
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model
from codebleu import calc_codebleu

os.environ["CUDA_VISIBLE_DEVICES"] = "6"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ==========================================
# 1. 路径与配置 (继承 Phase 2)
# ==========================================
BASE_MODEL_PATH = "./models/Qwen2.5-Coder-1.5B"
BASELINE_MERGED_PATH = "./outputs/Phase2_Baseline_Merged" 

PROXY_DATASET = "./datasets/Magicoder-OSS-Instruct"
PRIVATE_DATASET = "./datasets/Evolcode"

MAX_SEQ_LEN = 1024 
OUTPUT_DIR = "./outputs/Phase3_Proposed"
MERGED_SAVE_DIR = "./outputs/Phase3_Proposed_Merged"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MERGED_SAVE_DIR, exist_ok=True)

# ==========================================
# 2. 初始化模型与基座锚点
# ==========================================
print("📦 初始化与模型加载 (继承 Phase 2 结果)...")
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_PATH, 
    trust_remote_code=True,
    fix_mistral_regex=True
)
if tokenizer.pad_token is None: 
    tokenizer.pad_token = tokenizer.eos_token
im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>") 

# 直接加载 Phase 2 融合模型，使用 bfloat16
model = AutoModelForCausalLM.from_pretrained(BASELINE_MERGED_PATH, torch_dtype=torch.bfloat16)

# Anchor 锚点模型，同样使用 bfloat16
anchor_prefix = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, torch_dtype=torch.bfloat16)
anchor_prefix.model.layers = torch.nn.ModuleList(list(anchor_prefix.model.layers)[:4]) 
anchor_prefix.lm_head = torch.nn.Identity() 
anchor_prefix.eval()
anchor_prefix.requires_grad_(False)

print("💉 正在向 Layer 0~3 注入 LoRA 进行自适应对齐...")
peft_config = LoraConfig(
    r=32, lora_alpha=64,
    target_modules=r"model\.layers\.[0-3]\.(self_attn|mlp)\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)", 
    bias="none", task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)
if hasattr(model, "enable_input_require_grads"): 
    model.enable_input_require_grads()
else:
    model.get_input_embeddings().register_forward_hook(lambda m, i, o: o.requires_grad_(True))

# ==========================================
# 3. 数据处理
# ==========================================
dataset = load_dataset(PROXY_DATASET, split="train")

def preprocess(ex):
    q = ex.get('problem', ex.get('instruction', ''))
    a = ex.get('solution', ex.get('output', ''))
    p_ids = tokenizer(f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False).input_ids
    r_ids = tokenizer(f"{a}<|im_end|>\n", add_special_tokens=False).input_ids
    input_ids = (p_ids + r_ids)[:MAX_SEQ_LEN]
    labels = ([-100] * len(p_ids) + r_ids)[:MAX_SEQ_LEN]
    return {"input_ids": input_ids, "labels": labels, "attention_mask": [1] * len(input_ids)}

train_dataset = dataset.map(preprocess, remove_columns=dataset.column_names, num_proc=4)

# ==========================================
# 4. 多重约束 Trainer
# ==========================================
class MTIA_AlignTrainer(Trainer):
    def __init__(self, anchor_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.anchor_model = anchor_model.cuda() 

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        input_ids = inputs["input_ids"]
        labels = inputs.get("labels")

        with torch.no_grad():
            anchor_out = self.anchor_model(input_ids=input_ids, output_hidden_states=True)
            H_base = anchor_out.hidden_states[-1] 

        outputs = model(input_ids=input_ids, labels=labels, output_hidden_states=True)
        
        loss_ce = outputs.loss

        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        probs = F.softmax(shift_logits, dim=-1)
        log_probs = F.log_softmax(shift_logits, dim=-1)
        loss_entropy = -torch.sum(probs * log_probs, dim=-1).mean()

        H_dummy = outputs.hidden_states[4]
        loss_anchor = 1.0 - F.cosine_similarity(H_dummy, H_base, dim=-1).mean()

        H_suffix = outputs.hidden_states[-1] 
        H_d_norm = F.normalize(H_dummy, p=2, dim=-1)
        rel_dummy = F.log_softmax(torch.matmul(H_d_norm, H_d_norm.transpose(-1, -2)) / 0.1, dim=-1)
        
        H_s_norm = F.normalize(H_suffix, p=2, dim=-1)
        rel_suffix = F.softmax(torch.matmul(H_s_norm, H_s_norm.transpose(-1, -2)) / 0.1, dim=-1)
        loss_vr = F.kl_div(rel_dummy, rel_suffix, reduction="batchmean")

        # 归一化后的科学配比权重
        alpha = 0.2     
        gamma = 0.1     
        lam = 0.005     
        beta = 5.0    

        total_loss = alpha * loss_ce + gamma * loss_entropy + lam * loss_vr + beta * loss_anchor

        if self.state.global_step % 20 == 0:
            print(f"\n[Step {self.state.global_step}] CE: {alpha*loss_ce.item():.3f} | Ent: {gamma*loss_entropy.item():.3f} | VR: {lam*loss_vr.item():.3f} | Anch: {beta*loss_anchor.item():.3f} | Total: {total_loss.item():.3f}")

        return (total_loss, outputs) if return_outputs else total_loss

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR, per_device_train_batch_size=4, gradient_accumulation_steps=8,
    learning_rate=2e-5, num_train_epochs=1, bf16=True, gradient_checkpointing=True,
    max_grad_norm=0.5, logging_steps=20, save_strategy="no", report_to="none"
)

trainer = MTIA_AlignTrainer(
    anchor_model=anchor_prefix, model=model, args=training_args, train_dataset=train_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)
)

print("🚀 开始 Phase 3 动态对齐训练...")
trainer.train()

# ==========================================
# 5. 评估 CodeBLEU 
# ==========================================
print("📊 训练结束，开始在私有数据集上评估 Ours CodeBLEU...")
model.eval()
test_dataset = load_dataset(PRIVATE_DATASET, split="train").select(range(70000, 70200))

preds, refs = [], []
for example in test_dataset:
    q = example.get('problem', example.get('instruction', ''))
    a = example.get('solution', example.get('output', ''))
    inputs = tokenizer(f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n", return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=512, do_sample=False, 
            repetition_penalty=1.1,
            eos_token_id=[tokenizer.eos_token_id, im_end_id],
            pad_token_id=tokenizer.eos_token_id
        )
        
    generated_code = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=False)
    generated_code = generated_code.split("<|im_end|>")[0].strip()
    generated_code = generated_code.replace("```python", "").replace("```", "").strip()
    
    preds.append(generated_code)
    refs.append(a)

result = calc_codebleu(refs, preds, lang="python", weights=(0.25,0.25,0.25,0.25), tokenizer=None)
print(f"=== Phase 3 Proposed CodeBLEU: {result['codebleu'] * 100:.2f} ===")

# ==========================================
# 6. 保存最终对齐模型
# ==========================================
print(f"💾 正在保存 Phase 3 最终模型至 {MERGED_SAVE_DIR}...")
del trainer
torch.cuda.empty_cache()

merged_model = model.merge_and_unload()
merged_model.save_pretrained(MERGED_SAVE_DIR, safe_serialization=True)
tokenizer.save_pretrained(MERGED_SAVE_DIR)
print("Phase3 All Done.")