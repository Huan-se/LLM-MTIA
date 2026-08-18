import os
import torch
from datasets import load_from_disk # 💡 [修改] 导入本地加载函数
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model

# 环境配置
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 💡 [修改] 路径配置：指向新的医学数据集与新的输出目录
MODEL_PATH = "./models/Qwen2.5-1.5B-Base"
DATASET_PATH = "./datasets/processed_medical_data/private_medical_o1_hf"  # 隐私医疗数据集
OUTPUT_DIR = "./outputs/Oracle_Checkpoints_Medical"
MERGED_DIR = "./outputs/Oracle_Model_Merged_Medical"

MAX_SEQ_LEN = 1024
BATCH_SIZE = 4            
GRAD_ACCUM_STEPS = 8      
LEARNING_RATE = 2e-5      
EPOCHS = 1
TEST_SIZE = 500           # 💡 [修改] 留 500 条作为测试集

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MERGED_DIR, exist_ok=True)

print("📦 正在加载 Tokenizer 与 Base 模型...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None: 
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto")

print("📝 处理私有医学数据集...")
# 💡 [修改] 从本地磁盘加载预处理好的 HF 格式数据
dataset = load_from_disk(DATASET_PATH)
dataset = dataset.shuffle(seed=42)

# 💡 [修改] 动态切分数据集，防止硬编码越界
TRAIN_SIZE = len(dataset) - TEST_SIZE
train_dataset = dataset.select(range(TRAIN_SIZE))
test_dataset = dataset.select(range(TRAIN_SIZE, len(dataset)))

def preprocess_function(example):
    # 💡 [修改] 适应医学纯文本格式的切分与 Mask 逻辑
    text = example["text"]
    
    # 按照格式化脚本中设定的标识符进行切分
    parts = text.split("### Doctor:\n")
    if len(parts) == 2:
        prompt_text = parts[0] + "### Doctor:\n"
        response_text = parts[1]
    else:
        # 万一有异常数据，作为 fallback
        prompt_text = text
        response_text = ""

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    response_ids = tokenizer(response_text, add_special_tokens=False).input_ids

    input_ids = (prompt_ids + response_ids)[:MAX_SEQ_LEN]
    # 对 prompt 部分屏蔽 loss (-100)
    labels = ([-100] * len(prompt_ids) + response_ids)[:MAX_SEQ_LEN]

    return {"input_ids": input_ids, "labels": labels, "attention_mask": [1] * len(input_ids)}

train_dataset = train_dataset.map(preprocess_function, remove_columns=dataset.column_names, num_proc=4)

print("💉 注入 LoRA 适配器...")
peft_config = LoraConfig(
    r=32, lora_alpha=64, 
    # 包含注意力层与FFN层
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

print("🚀 开始第一阶段 Oracle 目标模型微调...")
trainer.train()

print("💾 保存完整的 Oracle 上限模型...")
del trainer
torch.cuda.empty_cache()
merged_model = model.merge_and_unload()
merged_model.save_pretrained(MERGED_DIR, safe_serialization=True)
tokenizer.save_pretrained(MERGED_DIR)
print("✅ 第一阶段完成！")