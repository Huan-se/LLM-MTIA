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
# 💡 核心功能：正则化代码提取器
# ==========================================
def extract_pure_code(text):
    """
    行业标准的代码提取逻辑：
    1. 优先提取 Markdown 代码块中的内容
    2. 如果没有代码块，则通过关键字截断后续的“大模型废话”和乱码
    """
    # 1. 尝试匹配标准的 Markdown 代码块 (如 ```python ... ``` 或 ```html ... ```)
    # re.DOTALL 使得 '.' 可以匹配换行符
    pattern = r"```[a-zA-Z]*\n(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # 2. 兜底方案：如果没有 Markdown 代码块，截断常见的自然语言废话和乱码标志
    cutoff_keywords = [
        "Note:", "Explanation:", "Here is", "This code", 
        "MessageLookupuser", "推", "加", "<|im_end|>", "\n\n\n"
    ]
    
    cleaned_text = text
    for keyword in cutoff_keywords:
        if keyword in cleaned_text:
            cleaned_text = cleaned_text.split(keyword)[0]
            
    return cleaned_text.strip()

def main():
    parser = argparse.ArgumentParser(description="Universal CodeBLEU Evaluator for MTIA Models")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the merged model directory")
    parser.add_argument("--dataset_path", type=str, default="./datasets/Evolcode", help="Path to the private test dataset")
    parser.add_argument("--output_json", type=str, default=None, help="Optional: Path to save generation results")
    parser.add_argument("--gpu_id", type=str, default="2", help="CUDA_VISIBLE_DEVICES index")
    args = parser.parse_args()

    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    print(f"📦 正在加载模型和 Tokenizer: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, fix_mistral_regex=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    model.eval()

    print(f"📝 加载测试数据: {args.dataset_path}")
    dataset = load_dataset(args.dataset_path, split="train")
    test_dataset = dataset.select(range(70000, 70200)) 

    predictions = []
    references = []
    results_cache = []

    print("🚀 开始代码生成与正则提取净化...")
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
                # 💡 移除 repetition_penalty，防止破坏正常代码语法，交由正则来截断废话
                eos_token_id=[tokenizer.eos_token_id, im_end_id], 
                pad_token_id=tokenizer.eos_token_id
            )
        
        # 1. 原始解码
        raw_generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=False)
        
        # 2. 💡 核心：正则化提取纯净代码
        pure_code = extract_pure_code(raw_generated_text)
        
        # 同样对 Ground Truth 提取一次，保证比较的公平性
        pure_reference = extract_pure_code(ground_truth_text) if "```" in ground_truth_text else ground_truth_text
        
        predictions.append(pure_code)
        references.append(pure_reference)
        
        if args.output_json:
            results_cache.append({
                "instruction": instruction_text,
                "raw_generated": raw_generated_text, # 保留原始输出用于对照
                "pure_extracted_code": pure_code     # 实际用于算分的代码
            })

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(results_cache, f, ensure_ascii=False, indent=2)
        print(f"💾 详细生成结果已保存至 {args.output_json}")

    print("📊 正在计算 CodeBLEU...")
    # 注意：如果数据集中含有 HTML/JS 等多语言，这里统一按 python 计算会导致 AST 不准。
    # 但作为相对指标（比较模型差距），固定 lang="python" 是没问题的。
    result = calc_codebleu(references, predictions, lang="python", weights=(0.25, 0.25, 0.25, 0.25), tokenizer=None)

    print("\n" + "="*40)
    print(f"📉 评估模型: {os.path.basename(args.model_path)}")
    print(f"=== 总分 (CodeBLEU): {result['codebleu'] * 100:.2f} ===")
    print(f"- N-gram 匹配: {result['ngram_match_score'] * 100:.2f}")
    print(f"- 关键词匹配 : {result['weighted_ngram_match_score'] * 100:.2f}")
    print(f"- 语法树匹配 : {result['syntax_match_score'] * 100:.2f}")
    print(f"- 数据流匹配 : {result['dataflow_match_score'] * 100:.2f}")
    print("="*40 + "\n")

if __name__ == "__main__":
    main()