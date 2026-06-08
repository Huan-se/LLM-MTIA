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
    提取纯净代码，剔除 Markdown 标记以及模型多余的自然语言废话。
    """
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


def main():
    parser = argparse.ArgumentParser(description="Universal CodeBLEU Evaluator")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the merged model directory")
    parser.add_argument("--dataset_path", type=str, default="./datasets/Magicoder-OSS-Instruct", help="Path to the private test dataset")
    parser.add_argument("--output_json", type=str, default=None, help="Path to save or read generation results")
    parser.add_argument("--gpu_id", type=str, default="2", help="CUDA_VISIBLE_DEVICES index")
    args = parser.parse_args()

    predictions = []
    references = []

    # ==========================================
    # 1. 断点缓存直连逻辑：如果 JSON 存在，直接跳过模型加载！
    # ==========================================
    if args.output_json and os.path.exists(args.output_json):
        print(f"✅ 发现已有的生成结果文件 {args.output_json}，跳过模型推理，直接加载计算 CodeBLEU！")
        with open(args.output_json, "r", encoding="utf-8") as f:
            results_cache = json.load(f)
            
        for res in results_cache:
            predictions.append(res.get("pure_extracted_code", ""))
            # 💡 注意：读取缓存时，我们直接获取预先存好的 ground_truth
            references.append(res.get("ground_truth", ""))
            
        if len(predictions) == 0 or len(references) == 0:
            print("⚠️ 警告：读取到的缓存文件为空或格式不匹配。请删除旧 JSON 文件后重新运行。")
            return
            
    else:
        # ==========================================
        # 2. 常规推理流程：加载模型并生成
        # ==========================================
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        print(f"📦 正在加载模型和 Tokenizer: {args.model_path}")
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, fix_mistral_regex=True)
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

        model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, device_map="auto")
        model.eval()

        print(f"📝 加载测试数据: {args.dataset_path}")
        dataset = load_dataset(args.dataset_path, split="train")
        # 统一取最后的 200 条作为私有测试集
        test_dataset = dataset.select(range(len(dataset)-200, len(dataset))) 

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
                    "ground_truth": pure_reference,      # 💡 将纯净参考代码存入 JSON，保证读取时可用
                    "raw_generated": raw_generated_text, 
                    "pure_extracted_code": pure_code
                })

        # 落盘保存
        if args.output_json:
            os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
            with open(args.output_json, "w", encoding="utf-8") as f:
                json.dump(results_cache, f, ensure_ascii=False, indent=2)
            print(f"💾 详细生成结果已保存至 {args.output_json}")

    # ==========================================
    # 3. 详细分项输出 (还原你的原始习惯)
    # ==========================================
    print("📊 正在计算 CodeBLEU...")
    result = calc_codebleu(references, predictions, lang="python", weights=(0.25, 0.25, 0.25, 0.25), tokenizer=None)
    
    print("\n" + "="*50)
    print(f"📉 评估模型: {os.path.basename(args.model_path)}")
    print(f"=== 总分 (CodeBLEU): {result['codebleu'] * 100:.2f} ===")
    print(f"- N-gram 匹配: {result['ngram_match_score'] * 100:.2f}")
    print(f"- 关键词匹配 : {result['weighted_ngram_match_score'] * 100:.2f}")
    print(f"- 语法树匹配 : {result['syntax_match_score'] * 100:.2f}")
    print(f"- 数据流匹配 : {result['dataflow_match_score'] * 100:.2f}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()