import torch
import numpy as np
import json
from transformers import AutoTokenizer
import datasets
import matplotlib.pyplot as plt

if __name__ == '__main__':
    tokenizerl3 = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct')

    mentalhealth = datasets.load_dataset("Amod/mental_health_counseling_conversations")["train"]['Context']
    aimedical = datasets.load_dataset('ruslanmv/ai-medical-chatbot')['train']['Patient']
    evolcode = datasets.load_dataset('nickrosh/Evol-Instruct-Code-80k-v1')['train']['instruction']
    codeparrotapps = datasets.load_dataset('codeparrot/apps')['test']['question']
    testdatasets = {'mentalhealth': mentalhealth, 'aimedical':aimedical, 'evolcode': evolcode, 'codeparrotapps': codeparrotapps}

    statistics = {}
    for k, dst in testdatasets.items():
        print(k)

        print(np.mean(lens), np.std(lens), np.max(lens), np.min(lens))
        lens = [len(tokenizerl3.encode(x, add_special_tokens=False)) for x in dst]
        statistics[k] = lens
        print(np.mean(lens), np.std(lens), np.max(lens), np.min(lens))
    output_dict = {}
    for k in testdatasets.keys():
        textlist = testdatasets[k]
        sortedlist = sorted(enumerate(statistics[k]), key=lambda x:x[1], reverse=True)[:1000]
        output_dict[k] = [textlist[x[0]] for x in sortedlist]

    json.dump(output_dict, open('data/validation-set/long-context-datasets.json', 'w'))

