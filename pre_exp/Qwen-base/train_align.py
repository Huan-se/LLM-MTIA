import torch
import torch.nn.functional as F
import os
import argparse
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model

def main():
    parser = argparse.ArgumentParser(description="Phase 3: Dynamic Alignment Fine-Tuning")
    # 动态超参数
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--lam", type=float, default=0.0)
    parser.add_argument("--beta", type=float, default=0.0)
    
    # 💡 新增：动态控制起点模式
    parser.add_argument("--start_mode", type=str, required=True, choices=["phase2", "splicedbase"], help="起点模式: phase2 或 splicedbase")
    
    # 动态路径
    parser.add_argument("--base_model_path", type=str, default="./models/Qwen2.5-1.5B-Base")
    parser.add_argument("--oracle_model_path", type=str, default="./outputs/Oracle_Model_Merged_Base")
    parser.add_argument("--phase2_model_path", type=str, default="./outputs/Phase2_Baseline_Merged_Base")
    parser.add_argument("--output_dir", type=str, required=True, help="训练过程输出路径")
    parser.add_argument("--merged_save_dir", type=str, required=True, help="最终融合模型保存路径")
    args = parser.parse_args()

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    PROXY_DATASET = "./datasets/Magicoder-OSS-Instruct"
    MAX_SEQ_LEN = 1024 

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.merged_save_dir, exist_ok=True)

    print(f"📦 加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None: 
        tokenizer.pad_token = tokenizer.eos_token

    # ==========================================
    # 💡 核心修复：根据 start_mode 动态构建起点模型
    # ==========================================
    if args.start_mode == "phase2":
        print(f"📦 起点模式 [phase2]：直接加载 Baseline 模型: {args.phase2_model_path}")
        model = AutoModelForCausalLM.from_pretrained(args.phase2_model_path, torch_dtype=torch.bfloat16)
        
    elif args.start_mode == "splicedbase":
        print("📦 起点模式 [splicedbase]：正在物理拼接 (Base 前缀 + Oracle 后缀) 作为训练起点...")
        base_model = AutoModelForCausalLM.from_pretrained(args.base_model_path, torch_dtype=torch.bfloat16)
        oracle_model = AutoModelForCausalLM.from_pretrained(args.oracle_model_path, torch_dtype=torch.bfloat16)
        
        # 将 Oracle 的 4~23 层、norm 和 lm_head 覆盖给 Base
        for i in range(4, len(base_model.model.layers)):
            base_model.model.layers[i] = oracle_model.model.layers[i]
        base_model.model.norm = oracle_model.model.norm
        base_model.lm_head = oracle_model.lm_head
        del oracle_model # 释放内存
        
        model = base_model
        print("✅ 缝合完成！")

    # Anchor 锚点模型始终为 Base
    anchor_prefix = AutoModelForCausalLM.from_pretrained(args.base_model_path, torch_dtype=torch.bfloat16)
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
        q = example.get('instruction', example.get('problem', example.get('prompt', example.get('input', example.get('query', '')))))
        a = example.get('output', example.get('solution', example.get('response', '')))
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
        def __init__(self, anchor_model, custom_weights, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.anchor_model = anchor_model.cuda() 
            self.cw = custom_weights

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            input_ids = inputs["input_ids"]
            labels = inputs.get("labels")

            # 💡 防 OOM 优化：动态判断是否需要提取沉重的隐藏层状态
            need_hidden = (self.cw['lam'] > 0) or (self.cw['beta'] > 0)
            
            loss_anchor = torch.tensor(0.0, device=model.device)
            loss_vr = torch.tensor(0.0, device=model.device)

            if need_hidden:
                with torch.no_grad():
                    anchor_out = self.anchor_model(input_ids=input_ids, output_hidden_states=True)
                    H_base = anchor_out.hidden_states[-1] 

                outputs = model(input_ids=input_ids, labels=labels, output_hidden_states=True)
                H_dummy = outputs.hidden_states[4]
                loss_anchor = 1.0 - F.cosine_similarity(H_dummy, H_base, dim=-1).mean()

                H_suffix = outputs.hidden_states[-1] 
                H_d_norm = F.normalize(H_dummy, p=2, dim=-1)
                rel_dummy = F.log_softmax(torch.matmul(H_d_norm, H_d_norm.transpose(-1, -2)) / 0.1, dim=-1)
                H_s_norm = F.normalize(H_suffix, p=2, dim=-1)
                rel_suffix = F.softmax(torch.matmul(H_s_norm, H_s_norm.transpose(-1, -2)) / 0.1, dim=-1)
                loss_vr = F.kl_div(rel_dummy, rel_suffix, reduction="batchmean")
            else:
                outputs = model(input_ids=input_ids, labels=labels, output_hidden_states=False)

            loss_ce = outputs.loss
            
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            probs = F.softmax(shift_logits, dim=-1)
            log_probs = F.log_softmax(shift_logits, dim=-1)
            loss_entropy = -torch.sum(probs * log_probs, dim=-1).mean()

            total_loss = self.cw['alpha'] * loss_ce + self.cw['gamma'] * loss_entropy + self.cw['lam'] * loss_vr + self.cw['beta'] * loss_anchor

            if self.state.global_step % 20 == 0:
                print(f"\n[Step {self.state.global_step}] Total: {total_loss.item():.3f} (CE:{self.cw['alpha']*loss_ce.item():.3f}|Ent:{self.cw['gamma']*loss_entropy.item():.3f}|VR:{self.cw['lam']*loss_vr.item():.3f}|Anch:{self.cw['beta']*loss_anchor.item():.3f})")

            return (total_loss, outputs) if return_outputs else total_loss

    training_args = TrainingArguments(
        output_dir=args.output_dir, per_device_train_batch_size=4, gradient_accumulation_steps=8,
        learning_rate=2e-5, num_train_epochs=1, bf16=True, gradient_checkpointing=True,
        max_grad_norm=0.5, logging_steps=20, save_strategy="no", report_to="none"
    )

    trainer = MTIA_AlignTrainer(
        anchor_model=anchor_prefix, 
        custom_weights={'alpha': args.alpha, 'gamma': args.gamma, 'lam': args.lam, 'beta': args.beta},
        model=model, args=training_args, train_dataset=train_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)
    )

    print(f"🚀 开始 Phase 3 批量对齐训练 | 模式={args.start_mode} | a={args.alpha}, g={args.gamma}, l={args.lam}, b={args.beta}")
    trainer.train()

    print(f"💾 保存融合模型至 {args.merged_save_dir}...")
    del trainer
    torch.cuda.empty_cache()
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(args.merged_save_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.merged_save_dir)

if __name__ == "__main__":
    main()