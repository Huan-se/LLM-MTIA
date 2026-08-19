import torch
import torch.nn.functional as F
import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
import evaluate

try:
    import spacy
    # 加载生物医学实体识别模型 (疾病 Disease & 药物 Chemical)
    nlp_med = spacy.load("en_ner_bc5cdr_md")
    print("✅ 成功加载 scispacy 医学实体识别模型 (en_ner_bc5cdr_md)。")
except Exception as e:
    nlp_med = None
    print("⚠️ 未找到 scispacy 医学模型，实体识别 F1 将跳过计算。")
    print("👉 修复命令: pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz")


# ==========================================
# 核心指标计算函数
# ==========================================
def compute_linear_cka(f1, f2):
    """计算特征分布的线性中心化核对齐 (CKA)"""
    f1_c = f1 - f1.mean(dim=0, keepdim=True)
    f2_c = f2 - f2.mean(dim=0, keepdim=True)
    
    dot_12 = torch.norm(torch.matmul(f1_c.t(), f2_c), p='fro') ** 2
    dot_11 = torch.norm(torch.matmul(f1_c.t(), f1_c), p='fro')
    dot_22 = torch.norm(torch.matmul(f2_c.t(), f2_c), p='fro')
    
    cka = dot_12 / (dot_11 * dot_22 + 1e-8)
    return cka.item()

def extract_medical_entities(text_list):
    """批量抽取医学文本中的专业实体集合"""
    if not nlp_med: return [set() for _ in text_list]
    all_entities = []
    # 禁用非必要的 pipeline 组件加速抽取
    for doc in nlp_med.pipe(text_list, disable=["tagger", "parser", "lemmatizer"]):
        ents = {ent.text.lower().strip() for ent in doc.ents}
        all_entities.append(ents)
    return all_entities

def compute_medical_metrics(predictions, references):
    """计算 BLEU, ROUGE 以及核心的医学实体 F1"""
    rouge_metric = evaluate.load("rouge")
    bleu_metric = evaluate.load("bleu")
    
    rouge_res = rouge_metric.compute(predictions=predictions, references=references)
    # 避免空生成引发报错
    preds_for_bleu = [p if p.strip() else "Empty" for p in predictions]
    bleu_res = bleu_metric.compute(predictions=preds_for_bleu, references=references, max_order=4)
    
    pred_ents = extract_medical_entities(predictions)
    ref_ents = extract_medical_entities(references)
    
    f1_list = []
    for p_set, r_set in zip(pred_ents, ref_ents):
        if len(p_set) == 0 and len(r_set) == 0:
            f1_list.append(1.0)
        elif len(p_set) == 0 or len(r_set) == 0:
            f1_list.append(0.0)
        else:
            inter = len(p_set & r_set)
            prec = inter / len(p_set)
            rec = inter / len(r_set)
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
            f1_list.append(f1)
            
    return {
        "BLEU-4": bleu_res.get("bleu", 0.0) * 100,
        "ROUGE-1": rouge_res.get("rouge1", 0.0) * 100,
        "ROUGE-L": rouge_res.get("rougeL", 0.0) * 100,
        "Med_Entity_F1": np.mean(f1_list) * 100 if f1_list else 0.0
    }

# ==========================================
# 模型构建工厂
# ==========================================
def build_eval_model(args, oracle_model=None):
    """动态构建用于评估的模型（标准加载 或 CPU物理拼接后推入GPU）"""
    if args.mode == "standard":
        print(f"📦 正在加载标准模型: {args.eval_model_path}")
        model = AutoModelForCausalLM.from_pretrained(args.eval_model_path, torch_dtype=torch.bfloat16, device_map={"": 0})
        return model
        
    elif args.mode == "spliced":
        print(f"📦 正在 CPU 上执行模型物理拼接 (Prefix: {args.eval_model_path} + Suffix: Oracle)...")
        eval_prefix_model = AutoModelForCausalLM.from_pretrained(args.eval_model_path, torch_dtype=torch.bfloat16, device_map={"": "cpu"})
        
        # 如果未传入 oracle_model，临时在 CPU 加载一个用于提取后缀
        if oracle_model is None:
            oracle_model_cpu = AutoModelForCausalLM.from_pretrained(args.oracle_model_path, torch_dtype=torch.bfloat16, device_map={"": "cpu"})
        else:
            oracle_model_cpu = oracle_model.to("cpu")
            
        # 替换第4层及之后的层、Norm 和 LM Head
        for i in range(4, len(eval_prefix_model.model.layers)):
            eval_prefix_model.model.layers[i] = oracle_model_cpu.model.layers[i]
        eval_prefix_model.model.norm = oracle_model_cpu.model.norm
        eval_prefix_model.lm_head = oracle_model_cpu.lm_head
        
        # 恢复 Oracle 到 GPU（如果它是传入的）
        if oracle_model is not None:
            oracle_model.to("cuda:0")
            
        print("🚀 拼接完成，正在将缝合模型推入 GPU...")
        return eval_prefix_model.to("cuda:0")
    else:
        raise ValueError(f"不支持的模式: {args.mode}")

# ==========================================
# 主流程
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Integrated Medical Model Evaluator")
    parser.add_argument("--tasks", nargs='+', default=["generation", "feature"], choices=["generation", "feature"], help="执行的评估任务")
    parser.add_argument("--mode", type=str, default="standard", choices=["standard", "spliced"], help="评估模型加载模式")
    
    # 模型路径参数
    parser.add_argument("--oracle_model_path", type=str, default="./outputs/Oracle_Model_Merged_Medical")
    parser.add_argument("--eval_model_path", type=str, required=True, help="待评估的模型/前缀路径")
    
    # 数据与评估参数
    parser.add_argument("--dataset_path", type=str, default="./datasets/processed_medical_data/private_medical_o1_hf")
    parser.add_argument("--test_size", type=int, default=500, help="测试集大小，必须与训练时预留的保持一致！")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--noise_std", type=float, default=1e-3, help="特征敏感度测试引入的噪声标准差")
    parser.add_argument("--output_json", type=str, default=None, help="文本生成缓存保存/读取路径")
    args = parser.parse_args()

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # 1. 严格加载并划分测试集 (保持与 train_oracle.py 完全一致的随机种子 42)
    print("📝 加载并严格隔离医学测试集 (Seed 42)...")
    dataset = load_from_disk(args.dataset_path)
    dataset = dataset.shuffle(seed=42)
    # 取最后 test_size 条作为验证集
    test_dataset = dataset.select(range(len(dataset) - args.test_size, len(dataset)))
    
    # 2. 准备 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.oracle_model_path, trust_remote_code=True, fix_mistral_regex=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    # 3. 加载模型
    oracle_model = AutoModelForCausalLM.from_pretrained(args.oracle_model_path, torch_dtype=torch.bfloat16, device_map={"": 0}).eval()
    eval_model = build_eval_model(args, oracle_model=oracle_model).eval()

    # ==========================================
    # 任务 A: 文本生成与医学 NLP 指标评估
    # ==========================================
    if "generation" in args.tasks:
        print("\n" + "="*50)
        print("🎯 开始执行任务: 医学文本生成与临床指标评估")
        
        predictions, references = [], []
        if args.output_json and os.path.exists(args.output_json):
            print(f"✅ 发现生成结果缓存 {args.output_json}，跳过模型推理...")
            with open(args.output_json, "r", encoding="utf-8") as f:
                results_cache = json.load(f)
            for res in results_cache:
                predictions.append(res.get("generated", ""))
                references.append(res.get("ground_truth", ""))
        else:
            results_cache = []
            for example in tqdm(test_dataset, desc="Generating"):
                text = example["text"]
                parts = text.split("### Doctor:\n")
                if len(parts) != 2: continue
                
                prompt_text = parts[0] + "### Doctor:\n"
                truth_text = parts[1].replace("<|endoftext|>", "").strip()
                
                inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda:0")
                
                with torch.no_grad():
                    outputs = eval_model.generate(
                        **inputs, max_new_tokens=512, do_sample=False, 
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                raw_generated = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
                
                predictions.append(raw_generated)
                references.append(truth_text)
                
                if args.output_json:
                    results_cache.append({
                        "prompt": prompt_text, "ground_truth": truth_text, "generated": raw_generated
                    })

            if args.output_json:
                os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
                with open(args.output_json, "w", encoding="utf-8") as f:
                    json.dump(results_cache, f, ensure_ascii=False, indent=2)

        print("📊 计算医学指标中...")
        med_metrics = compute_medical_metrics(predictions, references)
        
        print("\n" + "🌟 "*20)
        print(f"📊 医学文本生成质量报告 (Mode: {args.mode})")
        print(f"🔹 评估模型: {os.path.basename(args.eval_model_path)}")
        print("-" * 40)
        print(f"🔸 BLEU-4         : {med_metrics['BLEU-4']:.2f}")
        print(f"🔸 ROUGE-1        : {med_metrics['ROUGE-1']:.2f}")
        print(f"🔸 ROUGE-L        : {med_metrics['ROUGE-L']:.2f}")
        print(f"🔸 临床实体 F1    : {med_metrics['Med_Entity_F1']:.2f} %")
        print("🌟 "*20 + "\n")

    # ==========================================
    # 任务 B: 特征映射距离与敏感度评估 (Layer 4)
    # ==========================================
    if "feature" in args.tasks:
        print("\n" + "="*50)
        print("🎯 开始执行任务: 前缀特征映射距离与敏感度评估 (Layer 4)")
        
        total_mse, total_cos, total_cka = 0.0, 0.0, 0.0
        total_delta_oracle, total_delta_eval = 0.0, 0.0
        valid_batches = 0

        for i in tqdm(range(0, len(test_dataset), args.batch_size), desc="Extracting Features"):
            batch_examples = [test_dataset[k] for k in range(i, min(i + args.batch_size, len(test_dataset)))]
            prompts = [ex["text"].replace("<|endoftext|>", "") for ex in batch_examples]
                
            inputs = tokenizer(prompts, padding=True, truncation=True, max_length=1024, return_tensors="pt").to("cuda:0")
            attention_mask = inputs.attention_mask
            
            with torch.no_grad():
                # 提取 Embedding
                embeds_oracle = oracle_model.get_input_embeddings()(inputs.input_ids)
                embeds_eval = eval_model.get_input_embeddings()(inputs.input_ids)
                
                # 施加微弱的高斯噪声
                noise = torch.randn_like(embeds_oracle) * args.noise_std
                noisy_embeds_oracle = embeds_oracle + noise
                noisy_embeds_eval = embeds_eval + noise

                # 正常 Forward
                out_oracle = oracle_model(inputs_embeds=embeds_oracle, attention_mask=attention_mask, output_hidden_states=True)
                out_eval = eval_model(inputs_embeds=embeds_eval, attention_mask=attention_mask, output_hidden_states=True)
                H_oracle = out_oracle.hidden_states[4]
                H_eval = out_eval.hidden_states[4]

                # 噪声 Forward
                out_oracle_noisy = oracle_model(inputs_embeds=noisy_embeds_oracle, attention_mask=attention_mask, output_hidden_states=True)
                out_eval_noisy = eval_model(inputs_embeds=noisy_embeds_eval, attention_mask=attention_mask, output_hidden_states=True)
                H_oracle_noisy = out_oracle_noisy.hidden_states[4]
                H_eval_noisy = out_eval_noisy.hidden_states[4]

                # 应用有效 Token 掩码
                mask = attention_mask.bool()
                H_o_valid, H_e_valid = H_oracle[mask], H_eval[mask]
                H_o_n_valid, H_e_n_valid = H_oracle_noisy[mask], H_eval_noisy[mask]
                
                if H_o_valid.size(0) == 0: continue
                
                # 距离指标计算
                total_mse += F.mse_loss(H_e_valid, H_o_valid).item()
                total_cos += F.cosine_similarity(H_e_valid, H_o_valid, dim=-1).mean().item()
                total_cka += compute_linear_cka(H_e_valid.to(torch.float32), H_o_valid.to(torch.float32))
                
                # 敏感度追踪计算
                delta_o = torch.norm(H_o_n_valid - H_o_valid, p=2, dim=-1).mean().item()
                delta_e = torch.norm(H_e_n_valid - H_e_valid, p=2, dim=-1).mean().item()
                
                total_delta_oracle += delta_o
                total_delta_eval += delta_e
                valid_batches += 1

        if valid_batches > 0:
            avg_mse = total_mse / valid_batches
            avg_cos = total_cos / valid_batches
            avg_cka = total_cka / valid_batches
            avg_delta_o = total_delta_oracle / valid_batches
            avg_delta_e = total_delta_eval / valid_batches
            sensitivity_ratio = avg_delta_e / (avg_delta_o + 1e-9)

            print("\n" + "📉 "*20)
            print(f"📊 前缀特征映射综合量化报告 (Layer 4)")
            print(f"🔹 评估模型: {os.path.basename(args.eval_model_path)}")
            print("-" * 40)
            print(f"🔸 MSE (均方误差)         : {avg_mse:.4f}")
            print(f"🔸 Cosine (余弦相似度)    : {avg_cos:.4f}")
            print(f"🔸 CKA (中心化核对齐)     : {avg_cka:.4f}")
            print("-" * 40)
            print(f"🔹 目标模型特征敏感度 ΔHo : {avg_delta_o:.6f}")
            print(f"🔹 评估模型特征敏感度 ΔHe : {avg_delta_e:.6f}")
            print(f"🔹 变化量敏感度 (ΔHe/ΔHo) : {sensitivity_ratio:.4f}  (完美复刻 = 1.0000)")
            print("📉 "*20 + "\n")
        else:
            print("❌ 未提取到有效批次。")

if __name__ == "__main__":
    main()