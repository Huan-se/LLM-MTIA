import os
import argparse
import torch
import datasets
from datasets import load_dataset 
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
from tqdm import tqdm
import copy
from random import randint
import warnings


def prepare_dataset_dst(
    model,
    tokenizer,
    target_layer,
    dataset,
    dataset_text_field,
    max_seq_length,
    add_special_tokens=True,
    remove_unused_columns=True,
    dataset_batch_size=1,
    dataset_num_proc=None
):
    '''
    Inspired from trl.SFTTrainer._prepare_non_packed_dataloader
    '''
    # Inspired from: https://huggingface.co/learn/nlp-course/chapter7/6?fw=pt
    def tokenize(element):
        inputs = tokenizer(
            element[dataset_text_field],
            add_special_tokens=add_special_tokens,
            truncation=True,
            padding='max_length', # https://huggingface.co/docs/transformers/pad_truncation
            max_length=max_seq_length,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors='pt', 
        )
        if inputs['input_ids'].shape[-1] != max_seq_length:
            print(element, inputs['input_ids'].shape[-1], inputs)
        output = model(input_ids=inputs["input_ids"].to(model.device), attention_mask=inputs["input_ids"].to(model.device), output_hidden_states=True)['hidden_states'][target_layer].cpu()
    
        return {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"], 'hidden_states':output}

    signature_columns = ["input_ids", "labels", "hidden_states", "attention_mask"]

    if dataset.column_names is not None:  # None for IterableDataset
        extra_columns = list(set(dataset.column_names) - set(signature_columns))
    else:
        extra_columns = []

    if not remove_unused_columns and len(extra_columns) > 0:
        warnings.warn(
            "You passed `remove_unused_columns=False` on a non-packed dataset. This might create some issues with the default collator and yield to errors. If you want to "
            f"inspect dataset other columns (in this case {extra_columns}), you can subclass `DataCollatorForLanguageModeling` in case you used the default collator and create your own data collator in order to inspect the unused dataset columns."
        )

    map_kwargs = {
        "batched": True,
        "remove_columns": dataset.column_names if remove_unused_columns else None,
        "batch_size": dataset_batch_size,
    }
    if isinstance(dataset, datasets.Dataset):
        map_kwargs["num_proc"] = dataset_num_proc  # this arg is not available for IterableDataset
    tokenized_dataset = dataset.map(tokenize, **map_kwargs)

    return tokenized_dataset



# CUDA_VISIBLE_DEVICES=0 python prepare_replicate_dataset.py --target-model-path /llm/Meta-Llama-3.1-8B-Instruct --dataset-path ../data/processed-dataset/HuggingFaceH4-no_robots/train_dataset.json --access-layer-id 16 --max-seq-length 64 --save-path ../results/replic-dataset/

# CUDA_VISIBLE_DEVICES=0 python prepare_replicate_dataset.py --target-model-path /llm/ContactDoctor-Bio-Medical-Llama-3-8B --dataset-path ../data/processed-dataset/HuggingFaceH4-no_robots/train_dataset.json --access-layer-id 16 --max-seq-length 64 --save-path ../results/replic-dataset/



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Direct optimization.")
    parser.add_argument('--target-model-path', type=str, default='llama3.1-8binst', help='Path to loaded LLM')

    parser.add_argument('--dataset-path', type=str, default='', help='dataset_path, for replic mode dst')
    



    parser.add_argument('--save-path', type=str, default='', help='save_path')
    
    parser.add_argument('--access-layer-id', type=int, required=True, 
                        help='the list of the layer index from where the adversary has access')
    
    parser.add_argument('--max-seq-length', type=int, default=64, 
                        help='max sequence length')
    parser.add_argument('--loadin4bit', default=False, action="store_true", help='Load model in 4 Bit')


    args = parser.parse_args()

    if args.loadin4bit:
        quantization_config = {"load_in_4bit": True}
    else:
        quantization_config = None
    model = AutoModelForCausalLM.from_pretrained(
        args.target_model_path,
        torch_dtype=torch.float16,
        quantization_config=quantization_config,
        device_map="balanced"
    )
    model.eval()
    model.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(args.target_model_path)
    tokenizer.pad_token = tokenizer.eos_token

    train_dataset = load_dataset("json", data_files=args.dataset_path, split="train")

    # template dataset
    def template_dataset(examples):
        return{"text":  tokenizer.apply_chat_template(examples["messages"], tokenize=False)}

    train_dataset = train_dataset.map(template_dataset, remove_columns=["messages"])


    prepared_dataset = prepare_dataset_dst(
        model=model,
        tokenizer=tokenizer,
        target_layer=args.access_layer_id,
        dataset=train_dataset,
        dataset_text_field='text',
        max_seq_length=args.max_seq_length,
        add_special_tokens=True,
        remove_unused_columns=True,
        dataset_batch_size=1,
        dataset_num_proc=None)

   
    
    prepared_dataset.set_format("pt", columns=["input_ids", 'attention_mask', 'hidden_states'], output_all_columns=True)
    # set_format see https://discuss.huggingface.co/t/dataset-map-return-only-list-instead-torch-tensors/15767/5 and https://github.com/huggingface/datasets/issues/625

    
    for x in prepared_dataset['input_ids']:
        if len(x) != args.max_seq_length:
            print(x, len(x))
    del model


    save_folder_name = f'{args.replic_mode}-{args.target_model_path.split('/')[-1]}-msl{args.max_seq_length}-l{args.access_layer_id}'

    if 'train' in args.dataset_path:
        save_folder_name += f"-train"
    elif 'test' in args.dataset_path:
        save_folder_name += f"-test"

    prepared_dataset.save_to_disk(os.path.join(args.save_path, save_folder_name))
