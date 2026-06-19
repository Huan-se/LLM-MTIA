import torch
import torch.nn.functional as F
import os
import argparse
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

def compute_linear_cka(f1, f2):
    f1_c = f1 - f1.mean(dim=0, keepdim=True)
    f2_c = f2 - f2.mean(dim=0, keepdim=True)
    
    dot_12 = torch.norm(torch.matmul(f1_c.t(), f2_c), p='fro') ** 2
    dot_11 = torch.norm(torch.matmul(f1_c.t(), f1_c), p='fro')
    dot_22 = torch.norm(torch.matmul(f2_c.t(), f2_c), p='fro')
    
    cka = dot_12 / (dot_11 * dot_22 + 1e-8)
    return cka.item()

def main():
    parser = argparse.ArgumentParser(description="Integrated Prefix Feature & Sensitivity Evaluator")
    parser.add_argument("--mode", type=str, default="standard", choices=["standard", "spliced_base"])
    parser.add_argument("--oracle_model_path", type=str, default="./outputs/Oracle_Model_Merged_Base")
    parser.add_argument("--eval_model_path", type=str, default=None)
    parser.add_argument("--base_model_path", type=str, default="./models/Qwen2.5-1.5B-Base")
    parser.add_argument("--dataset_path", type=str, default="./datasets/OpenCodeInstruct")
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--noise_std", type=float, default=1e-3)
    # 💡 删除了 --gpu_id 参数
    args = parser.parse_args()

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    if args.mode == "spliced_base":
        target_eval_path = args.base_model_path
        eval_name = "Raw_Spliced_Prefix (Pure Base Model)"
    else:
        if not args.eval_model_path:
            raise ValueError("在 standard 模式下，必须提供 --eval_model_path 参数！")
        target_eval_path = args.eval_model_path
        eval_name = os.path.basename(target_eval_path)

    tokenizer = AutoTokenizer.from_pretrained(args.oracle_model_path, trust_remote_code=True, fix_mistral_regex=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    print(f"📦 加载目标 Oracle 前缀: {args.oracle_model_path}")
    # 💡 强行锁定设备 0，杜绝 Auto 分配引发跨设备 CPU 逃逸
    oracle_model = AutoModelForCausalLM.from_pretrained(args.oracle_model_path, torch_dtype=torch.bfloat16, device_map={"": 0}).eval()
    
    print(f"📦 加载评估模型 前缀: {target_eval_path}")
    eval_model = AutoModelForCausalLM.from_pretrained(target_eval_path, torch_dtype=torch.bfloat16, device_map={"": 0}).eval()

    dataset = load_dataset(args.dataset_path, split="train")
    test_dataset = dataset.select(range(len(dataset) - args.num_samples, len(dataset)))

    total_mse, total_cos, total_cka = 0.0, 0.0, 0.0
    total_delta_oracle, total_delta_eval = 0.0, 0.0
    valid_batches = 0

    print(f"🚀 开始多维度特征映射距离与敏感度计算 [{args.mode} 模式]...")
    for i in tqdm(range(0, len(test_dataset), args.batch_size)):
        batch_examples = [test_dataset[k] for k in range(i, min(i + args.batch_size, len(test_dataset)))]
        
        prompts = []
        for example in batch_examples:
            q = example.get('instruction', example.get('problem', example.get('prompt', example.get('input', example.get('query', '')))))
            a = example.get('output', example.get('solution', example.get('response', '')))
            if not q and 'messages' in example:
                msgs = example['messages']
                q = msgs[0]['content'] if len(msgs) > 0 else ''
                a = msgs[1]['content'] if len(msgs) > 1 else ''
            prompts.append(f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>\n")
            
        inputs = tokenizer(prompts, padding=True, truncation=True, max_length=1024, return_tensors="pt").to("cuda:0")
        attention_mask = inputs.attention_mask
        
        with torch.no_grad():
            embeds_oracle = oracle_model.get_input_embeddings()(inputs.input_ids)
            embeds_eval = eval_model.get_input_embeddings()(inputs.input_ids)
            
            noise = torch.randn_like(embeds_oracle) * args.noise_std
            noisy_embeds_oracle = embeds_oracle + noise
            noisy_embeds_eval = embeds_eval + noise

            out_oracle = oracle_model(inputs_embeds=embeds_oracle, attention_mask=attention_mask, output_hidden_states=True)
            out_eval = eval_model(inputs_embeds=embeds_eval, attention_mask=attention_mask, output_hidden_states=True)
            H_oracle = out_oracle.hidden_states[4]
            H_eval = out_eval.hidden_states[4]

            out_oracle_noisy = oracle_model(inputs_embeds=noisy_embeds_oracle, attention_mask=attention_mask, output_hidden_states=True)
            out_eval_noisy = eval_model(inputs_embeds=noisy_embeds_eval, attention_mask=attention_mask, output_hidden_states=True)
            H_oracle_noisy = out_oracle_noisy.hidden_states[4]
            H_eval_noisy = out_eval_noisy.hidden_states[4]

            mask = attention_mask.bool()
            H_o_valid, H_e_valid = H_oracle[mask], H_eval[mask]
            H_o_n_valid, H_e_n_valid = H_oracle_noisy[mask], H_eval_noisy[mask]
            
            if H_o_valid.size(0) == 0: continue
            
            total_mse += F.mse_loss(H_e_valid, H_o_valid).item()
            total_cos += F.cosine_similarity(H_e_valid, H_o_valid, dim=-1).mean().item()
            total_cka += compute_linear_cka(H_e_valid.to(torch.float32), H_o_valid.to(torch.float32))
            
            delta_o = torch.norm(H_o_n_valid - H_o_valid, p=2, dim=-1).mean().item()
            delta_e = torch.norm(H_e_n_valid - H_e_valid, p=2, dim=-1).mean().item()
            
            total_delta_oracle += delta_o
            total_delta_eval += delta_e
            
            valid_batches += 1

    if valid_batches == 0:
        print("❌ 未提取到有效批次。")
        return

    avg_mse = total_mse / valid_batches
    avg_cos = total_cos / valid_batches
    avg_cka = total_cka / valid_batches
    avg_delta_o = total_delta_oracle / valid_batches
    avg_delta_e = total_delta_eval / valid_batches
    sensitivity_ratio = avg_delta_e / (avg_delta_o + 1e-9)

    print("\n" + "="*60)
    print(f"📉 前缀映射综合量化报告 (Layer 4)")
    print(f"评估目标 : {eval_name}")
    print("-" * 60)
    print(f"🔸 MSE (均方误差)         : {avg_mse:.4f}")
    print(f"🔸 Cosine (余弦相似度)    : {avg_cos:.4f}")
    print(f"🔸 CKA (中心化核对齐)     : {avg_cka:.4f}")
    print("-" * 60)
    print(f"🔹 目标模型特征变化量 ΔHo : {avg_delta_o:.6f}")
    print(f"🔹 评估模型特征变化量 ΔHe : {avg_delta_e:.6f}")
    print(f"🔹 变化量敏感度 (ΔHe/ΔHo) : {sensitivity_ratio:.4f}  (完美复刻 = 1.0000)")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()