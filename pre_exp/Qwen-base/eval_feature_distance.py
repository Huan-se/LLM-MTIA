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
    parser = argparse.ArgumentParser(description="Prefix Feature Mapping Distance Evaluator")
    parser.add_argument("--oracle_model_path", type=str, default="./outputs/Oracle_Model_Merged_Base", help="目标上限模型路径")
    parser.add_argument("--eval_model_path", type=str, required=True, help="待评估的模型路径")
    parser.add_argument("--dataset_path", type=str, default="./datasets/OpenCodeInstruct", help="私有数据集路径")
    parser.add_argument("--num_samples", type=int, default=1000, help="测试条数")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gpu_id", type=str, default="2")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    tokenizer = AutoTokenizer.from_pretrained(args.oracle_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    print("📦 正在加载 Oracle (目标) 与待评估模型...")
    oracle_model = AutoModelForCausalLM.from_pretrained(args.oracle_model_path, torch_dtype=torch.bfloat16, device_map="auto").eval()
    eval_model = AutoModelForCausalLM.from_pretrained(args.eval_model_path, torch_dtype=torch.bfloat16, device_map="auto").eval()

    dataset = load_dataset(args.dataset_path, split="train")
    # 严格读取隔离的测试集数据
    test_dataset = dataset.select(range(len(dataset) - args.num_samples, len(dataset)))

    total_mse, total_cos, total_cka, valid_batches = 0.0, 0.0, 0.0, 0

    print("🚀 开始特征提取与距离计算...")
    for i in tqdm(range(0, len(test_dataset), args.batch_size)):
        batch_data = test_dataset[i : i + args.batch_size]
        
        prompts = []
        for j in range(len(batch_data.get('instruction', batch_data.get('problem', ['']*args.batch_size)))):
            q = batch_data.get('instruction', batch_data.get('problem', ['']*args.batch_size))[j]
            a = batch_data.get('output', batch_data.get('solution', ['']*args.batch_size))[j]
            # 兼容 messages 格式
            if not q and 'messages' in batch_data:
                msgs = batch_data['messages'][j]
                q = msgs[0]['content'] if len(msgs) > 0 else ''
                a = msgs[1]['content'] if len(msgs) > 1 else ''
            
            prompts.append(f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>\n")
            
        inputs = tokenizer(prompts, padding=True, truncation=True, max_length=1024, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            out_oracle = oracle_model(**inputs, output_hidden_states=True)
            out_eval = eval_model(**inputs, output_hidden_states=True)
            
            H_oracle = out_oracle.hidden_states[4] 
            H_eval = out_eval.hidden_states[4]
            
            mask = inputs.attention_mask.bool()
            H_oracle_valid = H_oracle[mask]
            H_eval_valid = H_eval[mask]
            
            if H_oracle_valid.size(0) == 0: continue
                
            total_mse += F.mse_loss(H_eval_valid, H_oracle_valid).item()
            total_cos += F.cosine_similarity(H_eval_valid, H_oracle_valid, dim=-1).mean().item()
            total_cka += compute_linear_cka(H_eval_valid.to(torch.float32), H_oracle_valid.to(torch.float32))
            valid_batches += 1

    avg_mse = total_mse / valid_batches
    avg_cos = total_cos / valid_batches
    avg_cka = total_cka / valid_batches

    print("\n" + "="*50)
    print(f"📉 特征映射函数量化报告 (Layer 4 输出)")
    print(f"评估目标 : {os.path.basename(args.eval_model_path)}")
    print("-" * 50)
    print(f"🔸 MSE (均方误差)         : {avg_mse:.4f}  (越低越好)")
    print(f"🔸 Cosine (余弦相似度)    : {avg_cos:.4f}  (越近 1.0 越好)")
    print(f"🔸 CKA (中心化核对齐)     : {avg_cka:.4f}  (越近 1.0 越好)")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()