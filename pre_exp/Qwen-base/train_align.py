import torch
import torch.nn.functional as F
import os
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# === 路径配置 ===
BASE_MODEL_PATH = "./models/Qwen2.5-1.5B-Base"
BASELINE_MERGED_PATH = "./outputs/Phase2_Baseline_Merged_Base" 
PROXY_DATASET = "./datasets/Magicoder-OSS-Instruct"

MAX_SEQ_LEN = 1024 
OUTPUT_DIR = "./outputs/Phase3_Proposed_Base"
MERGED_SAVE_DIR = "./outputs/Phase3_Proposed_Merged_Base"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MERGED_SAVE_DIR, exist_ok=True)

print("📦 加载模型 (继承 Phase 2 结果)...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None: 
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(BASELINE_MERGED_PATH, torch_dtype=torch.bfloat16)

# 虽然消融实验中 beta=0 (不用 Anchor)，但保留代码结构以备不时之需
anchor_prefix = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, torch_dtype=torch.bfloat16)
anchor_prefix.model.layers = torch.nn.ModuleList(list(anchor_prefix.model.layers)[:4]) 
anchor_prefix.lm_head = torch.nn.Identity() 
anchor_prefix.eval()
anchor_prefix.requires_grad_(False)

print("💉 注入对齐 LoRA...")
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

print("📝 处理公有代理数据集...")
dataset = load_dataset(PROXY_DATASET, split="train")

def preprocess(example):
    q = example.get('instruction', example.get('problem', example.get('prompt', '')))
    a = example.get('response', example.get('output', example.get('solution', '')))
    if not q and 'messages' in example:
        msgs = example['messages']
        q = msgs[0]['content'] if len(msgs) > 0 else ''
        a = msgs[1]['content'] if len(msgs) > 1 else ''

    p_ids = tokenizer(f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False).input_ids
    r_ids = tokenizer(f"{a}<|im_end|>\n", add_special_tokens=False).input_ids
    input_ids = (p_ids + r_ids)[:MAX_SEQ_LEN]
    labels = ([-100] * len(p_ids) + r_ids)[:MAX_SEQ_LEN]
    return {"input_ids": input_ids, "labels": labels, "attention_mask": [1] * len(input_ids)}

train_dataset = dataset.map(preprocess, remove_columns=dataset.column_names, num_proc=4)

class MTIA_AlignTrainer(Trainer):
    def __init__(self, anchor_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.anchor_model = anchor_model.cuda() 

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        input_ids = inputs["input_ids"]
        labels = inputs.get("labels")

        outputs = model(input_ids=input_ids, labels=labels, output_hidden_states=True)
        loss_ce = outputs.loss
        
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        probs = F.softmax(shift_logits, dim=-1)
        log_probs = F.log_softmax(shift_logits, dim=-1)
        loss_entropy = -torch.sum(probs * log_probs, dim=-1).mean()

        # 基于消融实验的最优配置：去除特征拉扯(VR)和回退锚点(Anchor)
        alpha, gamma, lam, beta = 0.2, 0.1, 0.0, 0.0     
        total_loss = alpha * loss_ce + gamma * loss_entropy 

        if self.state.global_step % 20 == 0:
            print(f"\n[Step {self.state.global_step}] Total: {total_loss.item():.3f} (CE:{alpha*loss_ce.item():.3f}|Ent:{gamma*loss_entropy.item():.3f})")

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

print("🚀 开始 Phase 3 最优对齐训练...")
trainer.train()

print(f"💾 保存 Phase 3 最终模型至 {MERGED_SAVE_DIR}...")
del trainer
torch.cuda.empty_cache()
merged_model = model.merge_and_unload()
merged_model.save_pretrained(MERGED_SAVE_DIR, safe_serialization=True)
tokenizer.save_pretrained(MERGED_SAVE_DIR)