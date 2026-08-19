
import argparse
from inversion import *
from datasets import load_from_disk
from transformers import AutoTokenizer
import json
from tqdm import tqdm



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Direct optimization.")
    parser.add_argument('--model-path', type=str, required=True, help='path to loaded LLM')
    parser.add_argument('--dataset-path', type=str, required=True, help='path to the saved dataset')
    parser.add_argument('--save-path', type=str,  required=True, help='path to store the results')
    args = parser.parse_args()


    test_ds = load_from_disk(args.dataset_path)

    with open('./config/llama3.1-8b.json') as f:
        config_dict = json.load(f)

    config = InversionConfig.from_dict(config_dict)
    training_args = args_from_config(TrainingArguments, config)
    model = HSInversionModel(config=config,)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
    print(config.model_name_or_path)

    training_args.output_dir = "./tmp"
    trainer = HS2OutputTrainer(model=model, args=training_args, data_collator=HS2OutputCollator())
    trainer.args.metric_for_best_model = None

    trainer._load_from_checkpoint(args.model_path)


    batchsize = 1
    generation_kwargs = {'early_stopping': False, 'num_beams': 1, 'do_sample': False, 'no_repeat_ngram_size': 0, 'max_length': 4096}
    dst_dict = {'inference': test_ds}
    result_dict = {}
    for k, ds in dst_dict.items():
        n_batches = len(ds) // batchsize
        
        reverttext = []
        for batch_id in tqdm(range(n_batches)):
            inputs = ds[batch_id * batchsize: (batch_id+1)*batchsize]
            inputs['hidden_states'] = inputs['hidden_states'].to(trainer.model.device)
            
            output = trainer.generate(inputs, generation_kwargs)
            reverttext.extend([x.replace('</s>', '').replace('<pad>', '').strip() for x in tokenizer.batch_decode(output)])
        result_dict[k] = reverttext
        torch.cuda.empty_cache()
    json.dump(result_dict, open(args.save_path, 'w'))