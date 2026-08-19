import os
import json
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import warnings
import datasets


def prepare_dataset(
    model,
    tokenizer,
    invert_tokenizer,
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
        invert_inputs = invert_tokenizer(element[dataset_text_field],
            add_special_tokens=add_special_tokens,
            truncation=True,
            padding='max_length', # https://huggingface.co/docs/transformers/pad_truncation
            max_length=max_seq_length,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors='pt', )
        
        if inputs['input_ids'].shape[-1] != max_seq_length:
            print(element, inputs['input_ids'].shape[-1], inputs)
        output = model(input_ids=inputs["input_ids"].to(model.device), attention_mask=inputs["input_ids"].to(model.device), output_hidden_states=True)['hidden_states'][target_layer].cpu()

        return {"texts":element[dataset_text_field], "input_ids": invert_inputs["input_ids"], "attention_mask": invert_inputs["attention_mask"], 'hidden_states':output}

    signature_columns = [dataset_text_field, "input_ids", "labels", "hidden_states", "attention_mask"]

    if dataset.column_names is not None:
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
        map_kwargs["num_proc"] = dataset_num_proc
    tokenized_dataset = dataset.map(tokenize, **map_kwargs)

    return tokenized_dataset



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Direct optimization.")
    parser.add_argument('--model-name', type=str, default='Llama-3-8B-Instruct', help='Path to loaded LLM')
    parser.add_argument('--invert-model', type=str, default='t5-base', help='path to t5-base')
    parser.add_argument('--dataset-name', type=str, default='mentalhealth, evolcode, norobot-train, norobot-test')
    parser.add_argument('--access-layer-id', type=int, required=True, help='the layer index that the adversary can see')
    parser.add_argument('--maxseqlen', default=-1, type=int, help='maximum sequence length. -1 means no limitation.')
    parser.add_argument('--savefolder', default='', type=str, help='subfolder for saving')
    args = parser.parse_args()
    
    invert_tokenizer = AutoTokenizer.from_pretrained(args.invert_model)
    invert_tokenizer.pad_token = invert_tokenizer.eos_token
    unify_tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    unify_tokenizer.pad_token = unify_tokenizer.eos_token

    if args.dataset_name in ['mentalhealth', 'evolcode']:
        texts = json.load(open(os.path.join('../data/validation-set/long-context-datasets.json')))[args.dataset_name]
        texts = sorted(texts, key=lambda x: len(unify_tokenizer.encode(x, add_special_tokens=False)), reverse=True)[:100]
        system_prompt = [invert_tokenizer.encode(x, add_special_tokens=False) for x in texts]
        max_seqlength = max([len(x) for x in system_prompt])
        train_dataset = datasets.Dataset.from_dict({'texts': texts, 'system_prompt': system_prompt})
    elif args.dataset_name in ['norobot-train', 'norobot-test']:
        split = args.dataset_name.split('-')[1]
        train_dataset = datasets.load_dataset("json", data_files=f'../data/black-box-inversion/processed-dataset/HuggingFaceH4-no_robots/{split}_dataset.json', split="train")
        def template_dataset(examples):
            return{"text": examples["messages"][1]['content']}
        train_dataset = train_dataset.map(template_dataset, remove_columns=["messages"])
        texts = list(train_dataset['text'])
        system_prompt = [invert_tokenizer.encode(x, add_special_tokens=False) for x in texts]
        max_seqlength = max([len(x) for x in system_prompt])
        print(max_seqlength)
        train_dataset = datasets.Dataset.from_dict({'texts': texts, 'system_prompt': system_prompt})

    if args.maxseqlen > 0:
        max_seqlength = args.maxseqlen

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map="balanced"
    )
    model.eval()
    model.requires_grad_(False)
    

    prepared_dataset = prepare_dataset(model=model,
        tokenizer=unify_tokenizer,
        invert_tokenizer=invert_tokenizer,
        target_layer=args.access_layer_id,
        dataset=train_dataset,
        dataset_text_field='texts',
        max_seq_length=max_seqlength,
        add_special_tokens=True,
        remove_unused_columns=True,
        dataset_batch_size=1,
        dataset_num_proc=None)
    prepared_dataset.set_format("pt", columns=["texts", 'input_ids', 'attention_mask', 'hidden_states'], output_all_columns=True)

    prepared_dataset.save_to_disk(os.path.join('../data/black-box-inversion', args.savefolder, f"inversion_{args.dataset_name}_{args.invert_model.split('/')[-1]}_l{args.access_layer_id}_{args.model_name.split('/')[-1]}{f'_msl{args.maxseqlen}' if args.maxseqlen > 0 else ''}"))