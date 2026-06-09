import torch, json, os, re, argparse
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

def main():
    parser = argparse.ArgumentParser(description="Universal CodeBLEU Evaluator")
    parser.add_argument("--model_path", type=str, required=True)
    # 锁定私有验证集 OpenCodeInstruct
    parser.add_argument("--dataset_path", type=str, default="./datasets/OpenCodeInstruct")
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--gpu_id", type=str, default="2")
    args = parser.parse_args()

    predictions, references = [], []
    
    # 缓存直连
    if args.output_json and os.path.exists(args.output_json):
        print(f"✅ 发现生成结果缓存 {args.output_json}，直接计算 CodeBLEU！")
        with open(args.output_json, "r", encoding="utf-8") as f:
            results_cache = json.load(f)
        for res in results_cache:
            predictions.append(res.get("pure_extracted_code", ""))
            references.append(res.get("ground_truth", ""))
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

        model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, device_map="auto")
        model.eval()

        dataset = load_dataset(args.dataset_path, split="train")
        # 严格隔离：只从末尾 1000 条测试集中抽取最后的 200 条作为固定评估集
        test_dataset = dataset.select(range(len(dataset)-200, len(dataset))) 

        results_cache = []
        print("🚀 开始生成与提取 (防幻觉正则模式)...")
        for example in tqdm(test_dataset):
            q = example.get('instruction', example.get('problem', ''))
            a = example.get('output', example.get('solution', ''))
            if not q and 'messages' in example:
                msgs = example['messages']
                q = msgs[0]['content'] if len(msgs) > 0 else ''
                a = msgs[1]['content'] if len(msgs) > 1 else ''
            
            prompt_text = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
            inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
            
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
    print(f"📉 评估模型: {os.path.basename(args.model_path)}")
    print(f"=== 总分 (CodeBLEU): {result['codebleu'] * 100:.2f} ===")
    print(f"- N-gram 匹配: {result['ngram_match_score'] * 100:.2f}")
    print(f"- 关键词匹配 : {result['weighted_ngram_match_score'] * 100:.2f}")
    print(f"- 语法树匹配 : {result['syntax_match_score'] * 100:.2f}")
    print(f"- 数据流匹配 : {result['dataflow_match_score'] * 100:.2f}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()