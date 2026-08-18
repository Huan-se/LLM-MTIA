import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model

# 环境配置
# os.environ["CUDA_VISIBLE_DEVICES"] = "0" # 请根据实际情况修改
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 路径配置
MODEL_PATH = "./models/Qwen2.5-1.5B-Base"
DATASET_PATH = "./datasets/OpenCodeInstruct"
OUTPUT_DIR = "./outputs/Oracle_Checkpoints_Base"
MERGED_DIR = "./outputs/Oracle_Model_Merged_Base"

MAX_SEQ_LEN = 1024
BATCH_SIZE = 4            
GRAD_ACCUM_STEPS = 8      
LEARNING_RATE = 2e-5      
EPOCHS = 1
TRAIN_SIZE = 200000
TEST_SIZE = 1000

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MERGED_DIR, exist_ok=True)

print("📦 正在加载 Tokenizer 与 Base 模型...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
# 强制设定 pad_token，绝不改变词表大小
if tokenizer.pad_token is None: 
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto")

print("📝 处理私有数据集 (严格切分测试集防止数据泄露)...")
# 加载数据集
dataset = load_dataset(DATASET_PATH, split="train")

dataset = dataset.shuffle(seed=42)

train_dataset = dataset.select(range(TRAIN_SIZE))
test_dataset = dataset.select(
    range(TRAIN_SIZE, TRAIN_SIZE + TEST_SIZE)
)

def preprocess_function(example):
    # 新代码：终极万能字段提取
    q = example.get('instruction', example.get('problem', example.get('prompt', example.get('input', example.get('query', '')))))
    a = example.get('output', example.get('solution', example.get('response', '')))

    if not q and 'messages' in example:
        msgs = example['messages']
        q = msgs[0]['content'] if len(msgs) > 0 else ''
        a = msgs[1]['content'] if len(msgs) > 1 else ''

    # ⚠️ 加上这个安全锁，如果实在找不到列名，会在终端大声报警并打印所有的键名！
    if not q:
        print(f"\n🚨 严重警告: 无法在当前数据条目中找到指令字段！该数据的键名为: {list(example.keys())}")

    prompt_text = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
    response_text = f"{a}<|im_end|>\n"

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    response_ids = tokenizer(response_text, add_special_tokens=False).input_ids

    input_ids = (prompt_ids + response_ids)[:MAX_SEQ_LEN]
    labels = ([-100] * len(prompt_ids) + response_ids)[:MAX_SEQ_LEN]

    return {"input_ids": input_ids, "labels": labels, "attention_mask": [1] * len(input_ids)}

train_dataset = train_dataset.map(preprocess_function, remove_columns=dataset.column_names, num_proc=4)

print("💉 注入 LoRA 适配器...")
peft_config = LoraConfig(
    r=32, lora_alpha=64, 
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 
    bias="none", task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)

if hasattr(model, "enable_input_require_grads"): model.enable_input_require_grads()
else: model.get_input_embeddings().register_forward_hook(lambda m, i, o: o.requires_grad_(True))

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR, per_device_train_batch_size=BATCH_SIZE, gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    learning_rate=LEARNING_RATE, lr_scheduler_type="cosine", warmup_ratio=0.05, num_train_epochs=EPOCHS,
    bf16=True, gradient_checkpointing=True, max_grad_norm=0.5, logging_steps=20, save_strategy="no", report_to="none" 
)

trainer = Trainer(
    model=model, args=training_args, train_dataset=train_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)
)

print("🚀 开始第一阶段 Base 模型微调...")
trainer.train()

print("💾 保存完整的 Oracle 上限模型...")
del trainer
torch.cuda.empty_cache()
merged_model = model.merge_and_unload()
merged_model.save_pretrained(MERGED_DIR, safe_serialization=True)
tokenizer.save_pretrained(MERGED_DIR)
print("✅ 第一阶段完成，随时可进行 CodeBLEU 评估！")