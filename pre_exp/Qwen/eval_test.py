import torch
import json
import os
import re
import argparse
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from codebleu import calc_codebleu

# ==========================================
# 1. 核心功能：正则化代码提取器
# ==========================================
def extract_pure_code(text):
    """提取纯净代码，剔除 Markdown 标记以及模型多余的自然语言废话"""
    pattern = r"```[a-zA-Z]*\n(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match: 
        return match.group(1).strip()
    
    cutoff_keywords = ["Note:", "Explanation:", "Here is", "MessageLookupuser", "推", "加", "<|im_end|>", "\n\n\n"]
    cleaned_text = text
    for kw in cutoff_keywords:
        if kw in cleaned_text: 
            cleaned_text = cleaned_text.split(kw)[0]
            
    return cleaned_text.strip()

# ==========================================
# 2. 核心功能：动态模型构建工厂
# ==========================================
def build_model(args, tokenizer):
    """根据 mode 参数动态构建内存结构，避免资源浪费"""
    if args.mode == "base":
        print(f"📦 正在加载 [纯基座模型]: {args.base_model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model_path, 
            torch_dtype=torch.bfloat16, 
            device_map="auto"
        )
        return model
        
    elif args.mode == "spliced":
        if not args.suffix_model_path:
            raise ValueError("在 spliced 模式下，必须提供 --suffix_model_path 参数！")
            
        print("📦 正在 CPU 上执行模型拼接 (Base Prefix + Finetuned Suffix)...")
        # 1. 先在 CPU 上以 bfloat16 加载两个模型以节省显存
        base_model = AutoModelForCausalLM.from_pretrained(args.base_model_path, torch_dtype=torch.bfloat16)
        suffix_model = AutoModelForCausalLM.from_pretrained(args.suffix_model_path, torch_dtype=torch.bfloat16)
        
        # 2. 物理缝合 Layer 4~27, norm, 以及 lm_head
        for i in range(4, len(base_model.model.layers)):
            base_model.model.layers[i] = suffix_model.model.layers[i]
        base_model.model.norm = suffix_model.model.norm
        base_model.lm_head = suffix_model.lm_head
        
        # 3. 释放后缀模型内存
        del suffix_model
        
        # 4. 手动推入 GPU
        print("🚀 拼接完成，正在将缝合模型推入 GPU...")
        base_model = base_model.cuda()
        return base_model
        
    else:
        raise ValueError(f"不支持的模式: {args.mode}")

# ==========================================
# 3. 主评估流程
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Comparative CodeBLEU Evaluator")
    parser.add_argument("--mode", type=str, choices=["base", "spliced"], required=True, help="评估模式：纯基座(base) 或 拼接模型(spliced)")
    parser.add_argument("--base_model_path", type=str, default="./models/Qwen2.5-Coder-1.5B", help="基座模型路径")
    parser.add_argument("--suffix_model_path", type=str, default=None, help="提供微调后缀的模型路径 (仅 spliced 模式需要)")
    parser.add_argument("--dataset_path", type=str, default="./datasets/Magicoder-OSS-Instruct", help="私有测试数据集路径")
    parser.add_argument("--output_json", type=str, default=None, help="结果保存路径")
    parser.add_argument("--gpu_id", type=str, default="6", help="CUDA_VISIBLE_DEVICES index")
    args = parser.parse_args()

    # 缓存直连逻辑
    predictions = []
    references = []
    
    if args.output_json and os.path.exists(args.output_json):
        print(f"✅ 发现已有的生成结果文件 {args.output_json}，跳过模型推理！")
        with open(args.output_json, "r", encoding="utf-8") as f:
            results_cache = json.load(f)
        for res in results_cache:
            predictions.append(res.get("pure_extracted_code", ""))
            references.append(res.get("ground_truth", ""))
    else:
        # 配置环境
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        # 准备 Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True, fix_mistral_regex=True)
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

        # 构建并加载模型
        model = build_model(args, tokenizer)
        model.eval()

        print(f"📝 加载测试数据: {args.dataset_path}")
        dataset = load_dataset(args.dataset_path, split="train")
        test_dataset = dataset.select(range(70000, 70200)) # 固定测试集范围保证公平比对

        results_cache = []

        print(f"🚀 开始在 {args.mode} 模式下生成代码与提取...")
        for example in tqdm(test_dataset):
            instruction_text = example.get('problem', example.get('instruction', ''))
            ground_truth_text = example.get('solution', example.get('output', ''))
            
            prompt_text = f"<|im_start|>user\n{instruction_text}<|im_end|>\n<|im_start|>assistant\n"
            inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=512, 
                    do_sample=False, 
                    eos_token_id=[tokenizer.eos_token_id, im_end_id], 
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # 解码与正则化提取
            raw_generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=False)
            pure_code = extract_pure_code(raw_generated_text)
            pure_reference = extract_pure_code(ground_truth_text) if "```" in ground_truth_text else ground_truth_text
            
            predictions.append(pure_code)
            references.append(pure_reference)
            
            if args.output_json:
                results_cache.append({
                    "instruction": instruction_text, 
                    "ground_truth": pure_reference, 
                    "raw_generated": raw_generated_text, 
                    "pure_extracted_code": pure_code
                })

        if args.output_json:
            os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
            with open(args.output_json, "w", encoding="utf-8") as f:
                json.dump(results_cache, f, ensure_ascii=False, indent=2)

    # 详细分项输出
    print("📊 正在计算 CodeBLEU...")
    result = calc_codebleu(references, predictions, lang="python", weights=(0.25, 0.25, 0.25, 0.25), tokenizer=None)
    
    print("\n" + "="*50)
    print(f"📉 评估模式: {args.mode.upper()}")
    print(f"=== 总分 (CodeBLEU): {result['codebleu'] * 100:.2f} ===")
    print(f"- N-gram 匹配: {result['ngram_match_score'] * 100:.2f}")
    print(f"- 关键词匹配 : {result['weighted_ngram_match_score'] * 100:.2f}")
    print(f"- 语法树匹配 : {result['syntax_match_score'] * 100:.2f}")
    print(f"- 数据流匹配 : {result['dataflow_match_score'] * 100:.2f}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()