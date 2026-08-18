import torch
import json
import os
import re
import argparse
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from codebleu import calc_codebleu

def extract_pure_code(text):
    pattern = r"```[a-zA-Z]*\n(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match: return match.group(1).strip()
    
    cutoff_keywords = ["Note:", "Explanation:", "Here is", "MessageLookupuser", "推", "加", "<|im_end|>", "\n\n\n"]
    cleaned_text = text
    for kw in cutoff_keywords:
        if kw in cleaned_text: cleaned_text = cleaned_text.split(kw)[0]
    return cleaned_text.strip()

def build_model(args):
    """动态模型构建工厂：支持直接加载完整模型，或在内存中物理拼接模型"""
    if args.mode == "standard":
        print(f"📦 正在加载标准模型: {args.model_path}")
        # 💡 强制放入当前可见的唯一显卡 (cuda:0)，严禁向 CPU 溢出
        model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, device_map={"": 0})
        return model
        
    elif args.mode == "spliced":
        print("📦 正在 CPU 上执行模型物理拼接 (Base Prefix + Oracle Suffix)...")
        # 💡 在 CPU 上安全拼接，防止 OOM
        base_model = AutoModelForCausalLM.from_pretrained(args.base_model_path, torch_dtype=torch.bfloat16, device_map={"": "cpu"})
        suffix_model = AutoModelForCausalLM.from_pretrained(args.suffix_model_path, torch_dtype=torch.bfloat16, device_map={"": "cpu"})
        
        for i in range(4, len(base_model.model.layers)):
            base_model.model.layers[i] = suffix_model.model.layers[i]
        base_model.model.norm = suffix_model.model.norm
        base_model.lm_head = suffix_model.lm_head
        
        del suffix_model
        
        print("🚀 拼接完成，正在将缝合模型推入 GPU...")
        # 💡 拼接完成后，作为一个整体推入当前可见的唯一显卡
        return base_model.to("cuda:0")
    else:
        raise ValueError(f"不支持的模式: {args.mode}")

def main():
    parser = argparse.ArgumentParser(description="Universal & Comparative CodeBLEU Evaluator")
    parser.add_argument("--mode", type=str, default="standard", choices=["standard", "spliced"])
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--base_model_path", type=str, default="./models/Qwen2.5-1.5B-Base")
    parser.add_argument("--suffix_model_path", type=str, default="./outputs/Oracle_Model_Merged_Base")
    parser.add_argument("--dataset_path", type=str, default="./datasets/OpenCodeInstruct")
    parser.add_argument("--output_json", type=str, default=None)
    # 💡 删除了 --gpu_id 参数，不再在 Python 中控制设备
    args = parser.parse_args()

    predictions, references = [], []
    
    if args.output_json and os.path.exists(args.output_json):
        print(f"✅ 发现生成结果缓存 {args.output_json}，跳过模型推理，直接计算 CodeBLEU！")
        with open(args.output_json, "r", encoding="utf-8") as f:
            results_cache = json.load(f)
        for res in results_cache:
            predictions.append(res.get("pure_extracted_code", ""))
            references.append(res.get("ground_truth", ""))
    else:
        # 💡 保留显存优化，但删除 CUDA_VISIBLE_DEVICES 环境变量设置
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        
        tok_path = args.model_path if args.mode == "standard" else args.base_model_path
        tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True, fix_mistral_regex=True)
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

        model = build_model(args)
        model.eval()

        dataset = load_dataset(args.dataset_path, split="train")
        test_dataset = dataset.select(range(len(dataset)-200, len(dataset))) 

        results_cache = []
        print(f"🚀 开始在 [{args.mode}] 模式下生成与提取代码...")
        for example in tqdm(test_dataset):
            q = example.get('instruction', example.get('problem', example.get('prompt', example.get('input', example.get('query', '')))))
            a = example.get('output', example.get('solution', example.get('response', '')))
            
            if not q and 'messages' in example:
                msgs = example['messages']
                q = msgs[0]['content'] if len(msgs) > 0 else ''
                a = msgs[1]['content'] if len(msgs) > 1 else ''
                
            prompt_text = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
            inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda:0")
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=512, do_sample=False, 
                    eos_token_id=[tokenizer.eos_token_id, im_end_id], pad_token_id=tokenizer.eos_token_id
                )
            
            raw_generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=False)
            pure_code = extract_pure_code(raw_generated_text)
            pure_reference = extract_pure_code(a) if "```" in a else a
            
            predictions.append(pure_code)
            references.append(pure_reference)
            
            if args.output_json:
                results_cache.append({
                    "instruction": q, "ground_truth": pure_reference, 
                    "raw_generated": raw_generated_text, "pure_extracted_code": pure_code
                })

        if args.output_json:
            os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
            with open(args.output_json, "w", encoding="utf-8") as f:
                json.dump(results_cache, f, ensure_ascii=False, indent=2)

    print("📊 正在计算 CodeBLEU...")
    result = calc_codebleu(references, predictions, lang="python", weights=(0.25, 0.25, 0.25, 0.25), tokenizer=None)
    
    print("\n" + "="*50)
    eval_name = os.path.basename(args.model_path) if args.mode == "standard" else "Raw_Spliced (Base_Prefix + Oracle_Suffix)"
    print(f"📉 评估模型: {eval_name}")
    print(f"=== 总分 (CodeBLEU): {result['codebleu'] * 100:.2f} ===")
    print(f"- N-gram 匹配: {result['ngram_match_score'] * 100:.2f}")
    print(f"- 关键词匹配 : {result['weighted_ngram_match_score'] * 100:.2f}")
    print(f"- 语法树匹配 : {result['syntax_match_score'] * 100:.2f}")
    print(f"- 数据流匹配 : {result['dataflow_match_score'] * 100:.2f}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()