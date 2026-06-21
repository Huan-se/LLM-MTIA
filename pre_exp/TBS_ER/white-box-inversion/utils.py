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
import pandas as pd  # 新增：用于读取 parquet 文件

from transformers import AutoTokenizer, AutoModel

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(CURRENT_DIR, '../data')
DATASET_PATH = os.path.join(DATA_PATH, 'my_datasets')
MODEL_PATH = os.path.join(DATA_PATH, 'models')
INPUTEMB_BASEVEC = os.path.join(DATA_PATH, 'tbs-basevector')
RESULT_PATH = os.path.join(CURRENT_DIR, '../resultsx')

EMBEDDING_PATH = 'BAAI/bge-large-en-v1.5' 

class DynamicModelPathResolver:
    def __init__(self, base_model_path):
        self.base_model_path = base_model_path
        self.preset_paths = {
            'llama3-8binst': "meta-llama/Meta-Llama-3-8B-Instruct",
            'llama3.1-8binst': "meta-llama/Llama-3.1-8B-Instruct",
            'llama2-7bchat': "meta-llama/Llama-2-7b-chat-hf",
            'qwen2.5-7binst': 'Qwen/Qwen2.5-7B-Instruct',
            'qwen2.5coder-7binst': 'Qwen/Qwen2.5-Coder-7B-Instruct',
            't5-base': 't5-base'
        }

    def __contains__(self, model_name):
        return True 

    def __getitem__(self, model_name):
        if model_name in self.preset_paths:
            return self.preset_paths[model_name]
        
        resolved_path = os.path.join(self.base_model_path, model_name)
        if not os.path.exists(resolved_path):
            print(f"[Warning] Model path may not exist: {resolved_path}")
        return resolved_path

LLM_PATH = DynamicModelPathResolver(MODEL_PATH)


def get_basis_vectors(embedding_weight, logger):
    logger.info("Producing basis vectors.")
    starttime = time.time()
    U, s, Vt = torch.svd(embedding_weight)
    duration = time.time() - starttime
    logger.info(f"Time: {duration}s.")
    return Vt

def get_list_invert_text(dataset):
    random.seed(0)
    # 兼容传入目录名或具体文件名
    dataset_base_path = os.path.join(DATASET_PATH, dataset)
    dataset_path_json = f"{dataset_base_path}.json"
    dataset_path_jsonl = f"{dataset_base_path}.jsonl"
    
    texts = []
    
    # 核心修改：如果是目录，则尝试寻找并读取 parquet 文件
    if os.path.isdir(dataset_base_path):
        parquet_files = sorted([os.path.join(dataset_base_path, f) for f in os.listdir(dataset_base_path) if f.endswith('.parquet')])
        if not parquet_files:
            raise FileNotFoundError(f"目录 {dataset_base_path} 中未找到任何 .parquet 文件。")
        
        # 读取第一个分片的数据（通常足够用于预实验提取几十条 top 长度文本）
        target_file = parquet_files[0]
        df = pd.read_parquet(target_file)
        
        possible_cols = ['instruction', 'prompt', 'text', 'problem', 'question', 'input']
        text_col = next((col for col in possible_cols if col in df.columns), None)
        
        if text_col:
            texts = df[text_col].dropna().astype(str).tolist()
        else:
             raise ValueError(f"Parquet 文件 {target_file} 未能找到有效的文本指令列。当前包含列: {list(df.columns)}")
             
    # 保留原有的 json/jsonl 读取逻辑
    elif os.path.exists(dataset_path_json) or os.path.exists(dataset_path_jsonl):
        target_path = dataset_path_json if os.path.exists(dataset_path_json) else dataset_path_jsonl
        if target_path.endswith('.jsonl'):
            with open(target_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip(): continue
                    data = json.loads(line)
                    text = data.get('instruction', data.get('prompt', data.get('text', data.get('problem', data.get('input', '')))))
                    if text:
                        texts.append(text)
        else:
            with open(target_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict) and 'text' in data:
                texts = data['text']
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                         text = item.get('instruction', item.get('prompt', item.get('text', item.get('problem', item.get('input', '')))))
                         if text: texts.append(text)
                    elif isinstance(item, str):
                         texts.append(item)
        
        if not texts:
            try:
                with open(target_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            first_obj = json.loads(line)
                            keys_info = list(first_obj.keys())
                            break
            except Exception:
                keys_info = "无法解析首行键名"
            raise ValueError(f"未能从 {target_path} 中解析出文本。首行数据的实际键名为: {keys_info}。请检查并修改 utils.py 的提取逻辑。")
    else:
        raise FileNotFoundError(f"未找到数据集: 尝试读取 {dataset_base_path} 目录及其 json 变体失败。")
        
    return texts

def check_results(input_vector, embedding_weight, method='cosine', eps=1e-8):
    if len(input_vector.shape) == 2:
        input_vector = input_vector.unsqueeze(0)

    if method == 'cosine':
        cosine_similarity = torch.matmul(input_vector, embedding_weight.T)
        norm_input_vector = input_vector.norm(dim=2).unsqueeze(2).expand(input_vector.shape[0], input_vector.shape[1], embedding_weight.shape[0])
        norm_embedding = embedding_weight.norm(dim=1).reshape(1, 1, -1, ).expand(input_vector.shape[0], input_vector.shape[1], embedding_weight.shape[0])
        norms = norm_input_vector * norm_embedding
        cosine_similarity = cosine_similarity / torch.max(norms, eps * torch.ones_like(norms))
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
                "bge_emb_cos_sim_mean": float(sims.mean().item()),
                "bge_emb_cos_sim_sem": 0.0 if len(sims) < 2 else float(scipy.stats.sem(sims.numpy())),
            }
        except Exception as e:
            print(e)
            print(f"Error getting {len(s1)} embeddings. Returning zeros.")
            return {
                "bge_emb_cos_sim_mean": 0.0,
                "bge_emb_cos_sim_sem": 0.0,
            }

def sem(L: List[float]) -> float:
    if len(L) < 2: return 0.0
    result = scipy.stats.sem(np.array(L))
    if isinstance(result, np.ndarray):
        return float(result.mean().item())
    return float(result)

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

            num_overlapping_words.append(count_overlapping_ngrams(true_words, pred_words, 1))
            num_overlapping_bigrams.append(count_overlapping_ngrams(true_words, pred_words, 2))
            num_overlapping_trigrams.append(count_overlapping_ngrams(true_words, pred_words, 3))

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

        bleu_results = np.array(
            [
                self.metric_bleu.compute(predictions=[p], references=[r])["score"]
                for p, r in zip(predictions_str, references_str)
            ]
        )
        rouge_result = self.metric_rouge.compute(
            predictions=predictions_str, references=references_str
        )
        self.bleu_results = bleu_results.tolist()

        exact_matches = np.array(predictions_str) == np.array(references_str)
        gen_metrics = {
            "bleu_score": bleu_results.mean(),
            "bleu_score_sem": sem(bleu_results.tolist()),
            "rouge1_score": rouge_result['rouge1'],
            "rouge2_score": rouge_result['rouge2'],
            "rougeL_score": rouge_result['rougeL'],
            "rougeLsum_score": rouge_result['rougeLsum'],
            "exact_match": mean(exact_matches.tolist()),
            "exact_match_sem": sem(exact_matches.tolist()),
        }

        all_metrics = {**set_token_metrics, **gen_metrics}
        for metric in self.additional_metrics:
            all_metrics.update(metric(references_str, predictions_str))

        return all_metrics