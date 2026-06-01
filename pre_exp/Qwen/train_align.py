import torch
import torch.nn.functional as F
import os
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model
from codebleu import calc_codebleu

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

BASE_MODEL_PATH = "./models/Qwen2.5-Coder-1.5B"
ORACLE_MODEL_PATH = "./outputs/Oracle_Model_Merged"
PROXY_DATASET = "./datasets/Magicoder-OSS-Instruct"
PRIVATE_DATASET = "./datasets/Evolcode"

MAX_SEQ_LEN = 2048

# ==========================================
# 1. 初始化拼接模型与基座锚点
# ==========================================
print("📦 初始化与模型拼接...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

# 主模型 (带梯度)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, torch_dtype=torch.float16)
oracle_model = AutoModelForCausalLM.from_pretrained(ORACLE_MODEL_PATH, torch_dtype=torch.float16)
anchor_prefix.model.layers = torch.nn.ModuleList(list(anchor_prefix.model.layers)[:4])
model.model.norm = oracle_model.model.norm
model.lm_head = oracle_model.lm_head
del oracle_model

# Anchor 锚点模型 (完全冻结，只保留前4层，节约显存)
anchor_prefix = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, torch_dtype=torch.float16)
anchor_prefix.model.layers = anchor_prefix.model.layers[:4]
anchor_prefix.lm_head = torch.nn.Identity() # 丢掉不用的头
anchor_prefix.eval()
anchor_prefix.requires_grad_(False)

# 注入 LoRA
peft_config = LoraConfig(
    r=64, lora_alpha=128,
    target_modules=r"model\.layers\.[0-3]\.(self_attn|mlp)\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)", 
    bias="none", task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)
if hasattr(model, "enable_input_require_grads"): model.enable_input_require_grads()

# ==========================================
# 2. 数据处理
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
# 3. 核心：自定义多重约束 Trainer
# ==========================================
class MTIA_AlignTrainer(Trainer):
    def __init__(self, anchor_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.anchor_model = anchor_model.cuda() # 部署 Anchor 到 GPU

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # 提取输入
        input_ids = inputs["input_ids"]
        labels = inputs.get("labels")

        # 1. 锚点前向传播 (获取原始基座 Layer 3 的输出 H_base)
        with torch.no_grad():
            anchor_out = self.anchor_model(input_ids=input_ids, output_hidden_states=True)
            H_base = anchor_out.hidden_states[-1] # Shape: [batch, seq, dim]

        # 2. 主模型前向传播 (必须开启 output_hidden_states 以获取各层特征)
        outputs = model(input_ids=input_ids, labels=labels, output_hidden_states=True)
        
        # Loss 1: 常规交叉熵 (L_CE)
        loss_ce = outputs.loss

        # Loss 2: 置信度平滑 (L_Entropy)
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous() # 错位对齐预测
        probs = F.softmax(shift_logits, dim=-1)
        log_probs = F.log_softmax(shift_logits, dim=-1)
        loss_entropy = -torch.sum(probs * log_probs, dim=-1).mean()

        # Loss 3: 底层特征防崩溃锚点 (L_Anchor)
        # H_dummy 位于 index 4 (Embed=0, L0=1, L1=2, L2=3, L3=4)
        H_dummy = outputs.hidden_states[4]
        loss_anchor = 1.0 - F.cosine_similarity(H_dummy, H_base, dim=-1).mean()

        # Loss 4: 深层关系对齐 (L_VR) - 替代版结构对齐
        H_suffix = outputs.hidden_states[-1] # Suffix 最终输出层特征
        
        # 计算 Dummy L3 的 Token 关系矩阵
        H_d_norm = F.normalize(H_dummy, p=2, dim=-1)
        rel_dummy = F.log_softmax(torch.matmul(H_d_norm, H_d_norm.transpose(-1, -2)) / 0.1, dim=-1)
        
        # 计算 Suffix L27 的 Token 关系矩阵
        H_s_norm = F.normalize(H_suffix, p=2, dim=-1)
        rel_suffix = F.softmax(torch.matmul(H_s_norm, H_s_norm.transpose(-1, -2)) / 0.1, dim=-1)
        
        loss_vr = F.kl_div(rel_dummy, rel_suffix, reduction="batchmean")

        # ========================================
        # ⚖️ 【探针打印】在此处观察绝对值并调整权重 
        # ========================================
        if self.state.global_step % 20 == 0:
            print(f"\n[Step {self.state.global_step}] Loss Scale Watch:")
            print(f"L_CE: {loss_ce.item():.4f} | L_Ent: {loss_entropy.item():.4f} | L_VR: {loss_vr.item():.4f} | L_Anchor: {loss_anchor.item():.4f}")

        # 初始假设权重，你需要根据上面打印出的绝对值进行修改平衡！
        alpha, gamma, lam, beta = 1.0, 0.1, 1.0, 5.0
        total_loss = alpha * loss_ce + gamma * loss_entropy + lam * loss_vr + beta * loss_anchor

        return (total_loss, outputs) if return_outputs else total_loss

training_args = TrainingArguments(
    output_dir="./outputs/Phase3_Proposed", per_device_train_batch_size=4, gradient_accumulation_steps=8,
    learning_rate=2e-4, num_train_epochs=1, fp16=True, gradient_checkpointing=True,
    logging_steps=20, save_strategy="no", report_to="none"
)

trainer = MTIA_AlignTrainer(
    anchor_model=anchor_prefix, model=model, args=training_args, train_dataset=train_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)
)

print("🚀 开始 Phase 3 动态对齐训练...")
trainer.train()

# ==========================================
# 4. 评估 CodeBLEU (期待分数的反弹！)
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
        out = model.generate(**inputs, max_new_tokens=512, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    preds.append(tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
    refs.append(a)

result = calc_codebleu(refs, preds, lang="python", weights=(0.25,0.25,0.25,0.25), tokenizer=None)
print(f"=== Phase 3 Proposed CodeBLEU: {result['codebleu'] * 100:.2f} ===")