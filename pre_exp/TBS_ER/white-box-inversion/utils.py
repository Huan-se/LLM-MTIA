import os
import json
import random
import time
from numpy.linalg import svd
from typing import Callable, Dict, List, Tuple, Union
import scipy
import nltk
import collections
import evaluate
import torch
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutput
import transformers
import copy
import numpy as np

from transformers import AutoTokenizer, AutoModel
import torch



DATA_PATH = '../data'
INPUTEMB_BASEVEC = '../data/tbs-basevector' # PATH TO STORE SVD BASIS MATRIX
RESULT_PATH = '../resultsx'

# It is recommended to download models and set a local path
EMBEDDING_PATH = 'BAAI/bge-large-en-v1.5' 
LLM_PATH = {
    'llama3-8binst': "meta-llama/Meta-Llama-3-8B-Instruct",
    'llama3.1-8binst': "meta-llama/Llama-3.1-8B-Instruct",
    'llama2-7bchat': "meta-llama/Llama-2-7b-chat-hf",
    'qwen2.5-7binst': 'Qwen/Qwen2.5-7B-Instruct',
    'qwen2.5coder-7binst': 'Qwen/Qwen2.5-Coder-7B-Instruct',
    't5-base': 't5-base'
}




def get_basis_vectors(embedding_weight, logger):
    logger.info("Producing basis vectors.")
    starttime = time.time()
    U, s, Vt = torch.svd(embedding_weight)
    duration = time.time() - starttime
    logger.info(f"Time: {duration}s.")
    return Vt


############
### DATA ###
############

def get_list_invert_text(dataset,):
    random.seed(0)
    print(dataset)
    if dataset == 'chat_2m':
        return json.load(open(os.path.join(DATA_PATH, 'validation-set/chat_2m_common.json'), 'r'))['text']
    elif dataset in ['mentalhealth', 'aimedical', 'evolcode', 'codeparrotapps']:
        data = json.load(open(os.path.join(DATA_PATH, 'validation-set/long-context-datasets.json')))[dataset]
        return data
    return []


############
### MAIN ###
############

def check_results(input_vector, embedding_weight, method='cosine', eps=1e-8):
    if len(input_vector.shape) == 2:
        input_vector = input_vector.unsqueeze(0)

    if method == 'cosine':

        # Compute the cosine similarity matrix
        cosine_similarity = torch.matmul(input_vector, embedding_weight.T)
        norm_input_vector = input_vector.norm(dim=2).unsqueeze(2).expand(input_vector.shape[0], input_vector.shape[1], embedding_weight.shape[0])
        norm_embedding = embedding_weight.norm(dim=1).reshape(1, 1, -1, ).expand(input_vector.shape[0], input_vector.shape[1], embedding_weight.shape[0])
        norms = norm_input_vector * norm_embedding
        cosine_similarity = cosine_similarity / torch.max(norms, eps * torch.ones_like(norms))
        # Find the index of the maximum cosine similarity for each row in B
        return cosine_similarity.argmax(dim=2)
    elif method == 'norm':
        return torch.cdist(input_vector, embedding_weight).argmin(dim=2)



def tbs_to_invert_vector_direct(optimized_var, opt_weight_matrix, fn_multiplier):
    return optimized_var @ opt_weight_matrix

def tbs_to_invert_vector_tanh(optimized_var, opt_weight_matrix, fn_multiplier):
    return (torch.tanh(optimized_var) * fn_multiplier) @ opt_weight_matrix

def tbs_to_invert_vector_atan(optimized_var, opt_weight_matrix, fn_multiplier):
    return (torch.atan(optimized_var) / torch.pi * fn_multiplier) @ opt_weight_matrix

CHANGE_VAR_FN_DICT = {'tanh': tbs_to_invert_vector_tanh, 'atan': tbs_to_invert_vector_atan, 'direct':tbs_to_invert_vector_direct}

def gen_invert_vector(attack_type, optimized_var=None, opt_weight_matrix=None, change_var_fn=None, fn_multiplier=1):

    if attack_type == 'ts':
        T = 1
        invert_vector = torch.softmax(optimized_var / T, dim=2) @ opt_weight_matrix
    elif attack_type == 'tbs':
        invert_vector = change_var_fn(optimized_var, opt_weight_matrix, fn_multiplier)
    else:
        invert_vector = optimized_var
    
    return invert_vector


##################
### Evaluation ###
##################

def embed(listsents, tokenizer, model, device='cuda'):
    if device == 'cuda':
        model.to(device)
        batches = len(listsents) // 128 + 1
        outputs = []
        with torch.no_grad():
            for batch in range(batches):
                text_list_batch = listsents[batch * 128 : (batch + 1) * 128]
                encoded_input = tokenizer(text_list_batch, padding=True, truncation=True, return_tensors='pt')
                encoded_input_cuda = {}
                for k, v in encoded_input.items():
                    encoded_input_cuda[k] = v.to(device)
                model_output = model(**encoded_input_cuda)
                # Perform pooling. In this case, cls pooling.
                sentence_embeddings = model_output[0][:, 0]
                sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
                outputs.append(sentence_embeddings.cpu())
        model.to('cpu')
        outputs = torch.cat(outputs)
        return outputs
    else:
        model.to('cpu')
        with torch.no_grad():
            
            encoded_input = tokenizer(listsents, padding=True, truncation=True, return_tensors='pt')

            model_output = model(**encoded_input)
            # Perform pooling. In this case, cls pooling.
            sentence_embeddings = model_output[0][:, 0]
            sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings


class EmbeddingCosineSimilarity:
    
    def __init__(self, embedding_model_path=EMBEDDING_PATH, device='cuda'):

        self.embtokenizer = AutoTokenizer.from_pretrained(embedding_model_path)
        self.embmodel = AutoModel.from_pretrained(embedding_model_path)
        self.embmodel.eval()
        self.device = device
    
    def __call__(self, s1: List[str], s2: List[str]) -> Dict[str, float]:
        try:
            e1 = embed(s1, self.embtokenizer, self.embmodel, device=self.device).to(torch.float32)
            e2 = embed(s2, self.embtokenizer, self.embmodel, device=self.device).to(torch.float32)
            sims = torch.nn.functional.cosine_similarity(e1, e2, dim=1)
            return {
                "bge_emb_cos_sim_mean": sims.mean().item(),
                "bge_emb_cos_sim_sem": scipy.stats.sem(sims.numpy()),
            }
        except Exception as e:
            print(e)
            print(f"Error getting {len(s1)} embeddings from OpenAI. Returning zeros.")
            return {
                "bge_emb_cos_sim_mean": 0.0,
                "bge_emb_cos_sim_sem": 0.0,
            }


def sem(L: List[float]) -> float:
    result = scipy.stats.sem(np.array(L))
    if isinstance(result, np.ndarray):
        return result.mean().item()
    return result


def mean(L: Union[List[int], List[float]]) -> float:
    return sum(L) / len(L)


def count_overlapping_ngrams(s1: str, s2: str, n: int) -> int:
    ngrams_1 = nltk.ngrams(s1, n)
    ngrams_2 = nltk.ngrams(s2, n)
    ngram_counts_1 = collections.Counter(ngrams_1)
    ngram_counts_2 = collections.Counter(ngrams_2)
    total = 0
    for ngram, count in ngram_counts_1.items():
        total += min(count, ngram_counts_2[ngram])
    return total



class Eval:
    additional_metrics: List[Callable[..., Dict[str, float]]]

    def __init__(self, tokenizer, embedding_model_path=EMBEDDING_PATH, device='cuda'):

        self.compute_metrics = self.compute_metrics_func
        self.metric_accuracy = evaluate.load("accuracy")
        self.metric_bleu = evaluate.load("sacrebleu")
        self.metric_rouge = evaluate.load("rouge")
        self.additional_metrics = [EmbeddingCosineSimilarity(embedding_model_path=embedding_model_path, device=device)]
        self.pad_token_id = tokenizer.pad_token_id
        self.bos_token_id = tokenizer.bos_token_id
        


    def compute_metrics_func(self, preds, labels):
        assert len(labels), "got empty labels for eval"
        assert (
            torch.tensor(preds).shape == torch.tensor(labels).shape
        ), f"preds.shape {preds.shape} / labels.shape {labels.shape}"

        # preds have the same shape as the labels.
        labels = labels.reshape(-1)
        preds = preds.reshape(-1)
        accuracy_result = self.metric_accuracy.compute(
            predictions=preds, references=labels
        )

        return {**accuracy_result}

    def _text_comparison_metrics(
        self,
        predictions_ids: List[List[int]],
        predictions_str: List[str],
        references_ids: List[List[int]],
        references_str: List[str],
    ) -> Dict[str, float]:
        assert len(predictions_ids) == len(references_ids)
        assert len(predictions_ids) == len(predictions_str)
        assert len(predictions_str) == len(references_str)
        num_preds = len(predictions_ids)
        if not num_preds:
            return {}

        ###########################################################

        # Compute token, precision, recall, and ngram-level metrics.
        precision_sum = 0.0
        recall_sum = 0.0
        num_overlapping_words = []
        num_overlapping_bigrams = []
        num_overlapping_trigrams = []
        num_true_words = []
        num_pred_words = []
        f1s = []
        for i in range(num_preds):
            true_words = nltk.tokenize.word_tokenize(references_str[i])
            pred_words = nltk.tokenize.word_tokenize(predictions_str[i])
            num_true_words.append(len(true_words))
            num_pred_words.append(len(pred_words))

            true_words_set = set(true_words)
            pred_words_set = set(pred_words)
            TP = len(true_words_set & pred_words_set)
            FP = len(true_words_set) - len(true_words_set & pred_words_set)
            FN = len(pred_words_set) - len(true_words_set & pred_words_set)

            precision = (TP) / (TP + FP + 1e-20)
            recall = (TP) / (TP + FN + 1e-20)

            try:
                f1 = (2 * precision * recall) / (precision + recall + 1e-20)
            except ZeroDivisionError:
                f1 = 0.0
            f1s.append(f1)

            precision_sum += precision
            recall_sum += recall

            ############################################################
            num_overlapping_words.append(
                count_overlapping_ngrams(true_words, pred_words, 1)
            )
            num_overlapping_bigrams.append(
                count_overlapping_ngrams(true_words, pred_words, 2)
            )
            num_overlapping_trigrams.append(
                count_overlapping_ngrams(true_words, pred_words, 3)
            )

        set_token_metrics = {
            "token_set_precision": (precision_sum / num_preds),
            "token_set_recall": (recall_sum / num_preds),
            "token_set_f1": mean(f1s),
            "token_set_f1_sem": sem(f1s),
            "n_ngrams_match_1": mean(num_overlapping_words),
            "n_ngrams_match_2": mean(num_overlapping_bigrams),
            "n_ngrams_match_3": mean(num_overlapping_trigrams),
            "num_true_words": mean(num_true_words),
            "num_pred_words": mean(num_pred_words),}
        ############################################################
        bleu_results = np.array(
            [
                self.metric_bleu.compute(predictions=[p], references=[r])["score"]
                for p, r in zip(predictions_str, references_str)
            ]
        )
        rouge_result = self.metric_rouge.compute(
            predictions=predictions_str, references=references_str
        )
        self.bleu_results = (
            bleu_results.tolist()
        )

        exact_matches = np.array(predictions_str) == np.array(references_str)
        gen_metrics = {
            "bleu_score": bleu_results.mean(),
            "bleu_score_sem": sem(bleu_results),
            "rouge1_score": rouge_result['rouge1'],
            "rouge2_score": rouge_result['rouge2'],
            "rougeL_score": rouge_result['rougeL'],
            "rougeLsum_score": rouge_result['rougeLsum'],
            "exact_match": mean(exact_matches),
            "exact_match_sem": sem(exact_matches),
        }


        all_metrics = {**set_token_metrics, **gen_metrics}
        for metric in self.additional_metrics:
            all_metrics.update(metric(references_str, predictions_str))

        return all_metrics

