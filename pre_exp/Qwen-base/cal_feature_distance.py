import torch
import torch.nn.functional as F
import os
import argparse
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# ==========================================
# 📐 核心数学工具：线性中心化核对齐 (Linear CKA)
# ==========================================
def compute_linear_cka(f1, f2):
    """
    计算两个特征矩阵的线性 CKA。
    f1, f2 形状为 [N_valid_tokens, Hidden_Dim]
    """
    # 1. 中心化 (Centering)
    f1_c = f1 - f1.mean(dim=0, keepdim=True)
    f2_c = f2 - f2.mean(dim=0, keepdim=True)
    
    # 2. 利用恒等式将 N*N 矩阵的迹转换为 D*D 的 Frobenius 范数，极大地节约显存
    # Tr(X X^T Y Y^T) == ||X^T Y||_F^2
    dot_12 = torch.norm(torch.matmul(f1_c.t(), f2_c), p='fro') ** 2
    dot_11 = torch.norm(torch.matmul(f1_c.t(), f1_c), p='fro')
    dot_22 = torch.norm(torch.matmul(f2_c.t(), f2_c), p='fro')
    
    cka = dot_12 / (dot_11 * dot_22 + 1e-8)
    return cka.item()

def main():
    parser = argparse.ArgumentParser(description="Prefix Feature Mapping Distance Evaluator")
    parser.add_argument("--oracle_model_path", type=str, default="./outputs/Oracle_Model_Merged", help="目标上限模型路径")
    parser.add_argument("--eval_model_path", type=str, required=True, help="待评估的模型路径 (Baseline 或 Phase 3)")
    parser.add_argument("--dataset_path", type=str, default="./datasets/Magicoder-OSS-Instruct", help="私有数据集路径")
    parser.add_argument("--num_samples", type=int, default=1000, help="用于评估的数据条数")
    parser.add_argument("--batch_size", type=int, default=4, help="批处理大小")
    parser.add_argument("--gpu_id", type=str, default="2", help="显卡编号")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    print("📦 正在加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.oracle_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None: 
        tokenizer.pad_token = tokenizer.eos_token

    print("📦 正在加载 Oracle (目标) 模型...")
    oracle_model = AutoModelForCausalLM.from_pretrained(
        args.oracle_model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    oracle_model.eval()

    print(f"📦 正在加载 待评估 模型: {args.eval_model_path}...")
    # 为了避免设备冲突，如果是在同一张卡上，device_map="auto" 能够自动管理
    eval_model = AutoModelForCausalLM.from_pretrained(
        args.eval_model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    eval_model.eval()

    print(f"📝 加载测试数据: {args.dataset_path}")
    dataset = load_dataset(args.dataset_path, split="train")
    # 取前 N 条数据进行特征对齐测试
    test_dataset = dataset.select(range(args.num_samples))

    # 结果累加器
    total_mse = 0.0
    total_cos = 0.0
    total_cka = 0.0
    valid_batches = 0

    print("🚀 开始特征提取与距离计算...")
    # 手动组装 Batch
    for i in tqdm(range(0, len(test_dataset), args.batch_size)):
        batch_data = test_dataset[i : i + args.batch_size]
        
        # 组装 Prompt
        prompts = []
        for q, a in zip(batch_data.get('instruction', batch_data.get('problem', ['']*args.batch_size)), 
                        batch_data.get('response', batch_data.get('solution', ['']*args.batch_size))):
            prompts.append(f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>\n")
            
        inputs = tokenizer(prompts, padding=True, truncation=True, max_length=1024, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            # 必须开启 output_hidden_states=True
            out_oracle = oracle_model(**inputs, output_hidden_states=True)
            out_eval = eval_model(**inputs, output_hidden_states=True)
            
            # 💡 核心对齐层：提取前 4 层 (Embedding=0, L0=1, L1=2, L2=3, L3=4) 的最终输出
            # 这里的 index 4 正好对应 Prefix 进入 Suffix 之前的分界点特征
            H_oracle = out_oracle.hidden_states[4] 
            H_eval = out_eval.hidden_states[4]
            
            # 💡 核心过滤：剥离所有的 Padding Token，只比较有效特征
            mask = inputs.attention_mask.bool()
            # 展平特征，形状变为 [N_valid_tokens, Hidden_Dim]
            H_oracle_valid = H_oracle[mask]
            H_eval_valid = H_eval[mask]
            
            if H_oracle_valid.size(0) == 0:
                continue
                
            # 1. 计算 MSE
            mse = F.mse_loss(H_eval_valid, H_oracle_valid).item()
            # 2. 计算 Cosine Similarity
            cos = F.cosine_similarity(H_eval_valid, H_oracle_valid, dim=-1).mean().item()
            # 3. 计算 CKA
            cka = compute_linear_cka(H_eval_valid.to(torch.float32), H_oracle_valid.to(torch.float32)) # 转 float32 防平方溢出
            
            total_mse += mse
            total_cos += cos
            total_cka += cka
            valid_batches += 1

    # 结算平均值
    avg_mse = total_mse / valid_batches
    avg_cos = total_cos / valid_batches
    avg_cka = total_cka / valid_batches

    print("\n" + "="*50)
    print(f"📉 特征映射函数量化报告 (Layer 4 输出)")
    print(f"评估目标 : {os.path.basename(args.eval_model_path)}")
    print(f"参考基准 : Oracle 上限模型")
    print(f"测试样本 : {args.num_samples} 条 (私有数据集)")
    print("-" * 50)
    print(f"🔸 MSE (均方误差)         : {avg_mse:.4f}  (越低越好)")
    print(f"🔸 Cosine (余弦相似度)    : {avg_cos:.4f}  (越近 1.0 越好)")
    print(f"🔸 CKA (中心化核对齐)     : {avg_cka:.4f}  (越近 1.0 越好)")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()