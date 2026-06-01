import torch
import json
import os
import argparse
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from codebleu import calc_codebleu

def main():
    # 1. 命令行参数解析
    parser = argparse.ArgumentParser(description="Universal CodeBLEU Evaluator for MTIA Models")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the merged model directory")
    parser.add_argument("--dataset_path", type=str, default="./datasets/Evolcode", help="Path to the private test dataset")
    parser.add_argument("--output_json", type=str, default=None, help="Optional: Path to save generation results")
    parser.add_argument("--gpu_id", type=str, default="2", help="CUDA_VISIBLE_DEVICES index")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    print(f"📦 正在加载模型和 Tokenizer: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, fix_mistral_regex=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    # 使用 bfloat16 以匹配训练阶段的数值精度
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    model.eval()

    print(f"📝 加载测试数据: {args.dataset_path}")
    dataset = load_dataset(args.dataset_path, split="train")
    test_dataset = dataset.select(range(70000, 70200)) # 固定测试集范围以保证公平对比

    predictions = []
    references = []
    results_cache = []

    print("🚀 开始代码生成与防幻觉净化...")
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
                repetition_penalty=1.1, # 抑制复读机幻觉
                eos_token_id=[tokenizer.eos_token_id, im_end_id], # 遇终止符即停
                pad_token_id=tokenizer.eos_token_id
            )
        
        # 解码并执行后处理净化
        generated_code = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=False)
        generated_code = generated_code.split("<|im_end|>")[0].strip()
        generated_code = generated_code.replace("```python", "").replace("```", "").strip()
        
        predictions.append(generated_code)
        references.append(ground_truth_text)
        
        # 缓存用于保存的详细记录
        if args.output_json:
            results_cache.append({
                "instruction": instruction_text,
                "ground_truth": ground_truth_text,
                "generated": generated_code
            })

    # 将生成结果落盘
    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(results_cache, f, ensure_ascii=False, indent=2)
        print(f"💾 详细生成结果已保存至 {args.output_json}")

    print("📊 正在计算 CodeBLEU...")
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