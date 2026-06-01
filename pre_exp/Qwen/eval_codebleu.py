import torch
import json
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from codebleu import calc_codebleu
import os

# --- 配置 ---
MERGED_DIR = "./outputs/Oracle_Model_Merged"
DATASET_ID = "./datasets/Evolcode"
TEST_SAMPLES = 200
OUTPUT_JSON = "./outputs/test_predictions.json" # 保存中间结果的地方

print("📦 加载模型和 Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MERGED_DIR, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MERGED_DIR, torch_dtype=torch.bfloat16, device_map="auto")
model.eval()

print("📝 加载测试数据...")
dataset = load_dataset(DATASET_ID, split="train")
test_dataset = dataset.select(range(len(dataset)-TEST_SAMPLES, len(dataset)))

predictions = []
references = []

print("🚀 开始生成测试集代码 (具有断点保护)...")
# 如果之前已经生成过了，我们可以直接跳过这漫长的 40 分钟
if os.path.exists(OUTPUT_JSON):
    print(f"✅ 发现已有的生成结果文件 {OUTPUT_JSON}，直接加载用于算分！")
    with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
        predictions = data["predictions"]
        references = data["references"]
else:
    for example in tqdm(test_dataset):
        # 兼容不同数据集字段
        instruction_text = example.get('problem', example.get('instruction', ''))
        ground_truth_text = example.get('solution', example.get('output', ''))
        
        prompt_text = f"<|im_start|>user\n{instruction_text}<|im_end|>\n<|im_start|>assistant\n"
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=512, 
                do_sample=False, # 贪心解码，移除 temperature 以消除警告
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # 截取生成的纯代码部分
        generated = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        predictions.append(generated)
        references.append(ground_truth_text)
    
    # 【核心保护机制】：将结果落盘保存
    print("💾 生成完毕！正在将结果保存到本地...")
    os.makedirs("./outputs", exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"predictions": predictions, "references": references}, f, ensure_ascii=False, indent=2)

print("📊 正在计算 CodeBLEU...")
# 此时执行计算，安全可靠
result = calc_codebleu(references, predictions, lang="python", weights=(0.25, 0.25, 0.25, 0.25), tokenizer=None)

print("\n=== CodeBLEU 评估结果 ===")
print(f"总分 (CodeBLEU): {result['codebleu'] * 100:.2f}")
print(f"- N-gram 匹配: {result['ngram_match_score'] * 100:.2f}")
print(f"- 关键词匹配 : {result['weighted_ngram_match_score'] * 100:.2f}")
print(f"- 语法树匹配 : {result['syntax_match_score'] * 100:.2f}")
print(f"- 数据流匹配 : {result['dataflow_match_score'] * 100:.2f}")