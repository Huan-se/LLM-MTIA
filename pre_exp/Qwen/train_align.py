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

os.environ["CUDA_VISIBLE_DEVICES"] = "2" # 确保显卡编号正确

# ==========================================
# 1. 路径与配置 (逻辑修正：继承 Phase 2)
# ==========================================
BASE_MODEL_PATH = "./models/Qwen2.5-Coder-1.5B"
# 【修正核心】：直接加载我们在第二阶段微调并融合好的基线模型
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
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None: 
    tokenizer.pad_token = tokenizer.eos_token
im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>") 

# 2.1 主模型 (直接加载 Phase 2 缝合且粗调过的完整模型)
model = AutoModelForCausalLM.from_pretrained(BASELINE_MERGED_PATH, torch_dtype=torch.float16)

# 2.2 Anchor 锚点模型 (纯净 Base 模型，用于防崩溃)
anchor_prefix = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, torch_dtype=torch.float16)
anchor_prefix.model.layers = torch.nn.ModuleList(list(anchor_prefix.model.layers)[:4]) 
anchor_prefix.lm_head = torch.nn.Identity() 
anchor_prefix.eval()
anchor_prefix.requires_grad_(False)

# 2.3 注入 LoRA (仅对前4层注入，后半部分自动被 PEFT 冻结，维持黑盒特性)
print("💉 正在向 Layer 0~3 注入 LoRA 进行自适应对齐...")
peft_config = LoraConfig(
    r=64, lora_alpha=128,
    target_modules=r"model\.layers\.[0-3]\.(self_attn|mlp)\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)", 
    bias="none", task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)
if hasattr(model, "enable_input_require_grads"): 
    model.enable_input_require_grads()
else:
    model.get_input_embeddings().register_forward_hook(lambda m, i, o: o.requires_grad_(True))

model.print_trainable_parameters()

# ==========================================
# 3. 数据处理
# ==========================================
print("📝 处理公开代理数据集...")
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
# 4. 核心：多重约束 Trainer (权重修正版)
# ==========================================
class MTIA_AlignTrainer(Trainer):
    def __init__(self, anchor_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.anchor_model = anchor_model.cuda() 

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        input_ids = inputs["input_ids"]
        labels = inputs.get("labels")

        # 1. 锚点前向传播 
        with torch.no_grad():
            anchor_out = self.anchor_model(input_ids=input_ids, output_hidden_states=True)
            H_base = anchor_out.hidden_states[-1] 

        # 2. 主模型前向传播
        outputs = model(input_ids=input_ids, labels=labels, output_hidden_states=True)
        
        # Loss 1: CE
        loss_ce = outputs.loss

        # Loss 2: Entropy
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        probs = F.softmax(shift_logits, dim=-1)
        log_probs = F.log_softmax(shift_logits, dim=-1)
        loss_entropy = -torch.sum(probs * log_probs, dim=-1).mean()

        # Loss 3: Anchor (Layer 4 输出)
        H_dummy = outputs.hidden_states[4]
        loss_anchor = 1.0 - F.cosine_similarity(H_dummy, H_base, dim=-1).mean()

        # Loss 4: VR (深层关系)
        H_suffix = outputs.hidden_states[-1] 
        H_d_norm = F.normalize(H_dummy, p=2, dim=-1)
        rel_dummy = F.log_softmax(torch.matmul(H_d_norm, H_d_norm.transpose(-1, -2)) / 0.1, dim=-1)
        
        H_s_norm = F.normalize(H_suffix, p=2, dim=-1)
        rel_suffix = F.softmax(torch.matmul(H_s_norm, H_s_norm.transpose(-1, -2)) / 0.1, dim=-1)
        loss_vr = F.kl_div(rel_dummy, rel_suffix, reduction="batchmean")

        # ⚖️ 【权重科学配比】
        alpha = 0.1     # 降权 CE，避免代理数据集过拟合
        gamma = 0.1     # 平滑分布
        lam = 0.005     # 压制 VR 的庞大数值，使其贡献收敛到 ~2.5 左右
        beta = 20.0     # 放大极小的 Cosine 距离，使其产生足够的锚定拉力

        total_loss = alpha * loss_ce + gamma * loss_entropy + lam * loss_vr + beta * loss_anchor

        # 监控尺度 (可选开启)
        if self.state.global_step % 20 == 0:
            print(f"\n[Step {self.state.global_step}] Scaled Contributions:")
            print(f"CE: {alpha*loss_ce.item():.3f} | Ent: {gamma*loss_entropy.item():.3f} | VR: {lam*loss_vr.item():.3f} | Anch: {beta*loss_anchor.item():.3f} | Total: {total_loss.item():.3f}")

        return (total_loss, outputs) if return_outputs else total_loss

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR, per_device_train_batch_size=4, gradient_accumulation_steps=8,
    learning_rate=2e-4, num_train_epochs=1, fp16=True, gradient_checkpointing=True,
    max_grad_norm=1.0, logging_steps=20, save_strategy="no", report_to="none"
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
        out = model.generate(
            **inputs, max_new_tokens=512, do_sample=False, 
            eos_token_id=[tokenizer.eos_token_id, im_end_id],
            pad_token_id=tokenizer.eos_token_id
        )
    preds.append(tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
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
print("✅ Phase 3 结束")