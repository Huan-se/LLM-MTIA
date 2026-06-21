import os
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import Cache, DynamicCache

from tqdm import trange
import logging
import utils
import shutil
import time
from scipy.linalg import svd

def init_weights_he_norm(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)

def init_weights_xav_norm(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal(m.weight)
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_normal(m.weight)
        
def init_weights_he_uni(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform(m.weight)
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_uniform(m.weight)

def init_weights_xav_uni(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform(m.weight)

init_dict = {
    'he_norm': init_weights_he_norm,
    'xav_norm': init_weights_xav_norm,
    'he_uni': init_weights_he_uni,
    'xav_uni': init_weights_xav_uni
    }

class MLP(nn.Module):
    def __init__(self, width=4096, num_classes=10):
        super(MLP, self).__init__()
        self.fc_1 = nn.Linear(width, width)
        self.fc_2 = nn.Linear(width, width)
        self.fc_3 = nn.Linear(width, num_classes)

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = F.relu(self.fc_1(out))
        out = F.relu(self.fc_2(out))
        out = self.fc_3(out)
        return out
    
    def embed(self, x):
        if len(x.shape) == 3:
            out = x.view(x.size(0), x.size(1), -1)
        elif len(x.shape) == 2:
            out = x.view(x.size(0), -1)
        out = F.relu(self.fc_1(out))
        out = F.relu(self.fc_2(out))
        return out

def get_network(width=4096, init='none'):
    net = MLP(width)
    if init != 'none':
        net.apply(init_dict[init])
    return net

def get_embeddings(last_hidden_states: torch.Tensor,
                 attention_mask: torch.Tensor) -> torch.Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        embeddings = last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        embeddings = last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
    return F.normalize(embeddings, p=2, dim=1)

def main():
    parser = argparse.ArgumentParser(description="Query-Free MTIA: Cross-Model TBS/ER Inversion.")

    parser.add_argument('--target-model', type=str, required=True, help='Name of the victim target model.')
    parser.add_argument('--surrogate-model', type=str, required=True, help='Name of the proxy surrogate model.')
    
    parser.add_argument('--embedding-model', action='store_true', default=False, help='Whether the models are embedding models.')
    parser.add_argument('--dataset', type=str, default='', help='Dataset name.')
    parser.add_argument('--range', type=str, default=None, help='range of dataset, in formt of start_id:end_id.')
    parser.add_argument('--folder', type=str, required=True, help='save folder under the path set in utils.RESULTS.')
    
    parser.add_argument('--attack', type=str, default=None, 
                        help='Advanced Attacks: ts, er, tbs')
    parser.add_argument('--tbs-changevar', type=str, default='atan:5', help='change of variable method for tbs')
    
    parser.add_argument('--access-layer-id', type=str, required=True, 
                        help='Layer index where ISs are extracted and optimized')

    parser.add_argument('--num-steps', type=int, default=10000, help='optimization steps')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    
    parser.add_argument('--wd-l1', type=float, default=0.0, help='Weight decay of L1 norm')
    parser.add_argument('--wd-l2', type=float, default=0.0, help='Weight decay of L2 norm')
    parser.add_argument('--w-dm', type=float, default=0, help='DM penalty.')
    parser.add_argument('--singular', action='store_true', help='whether to apply the original singular basis')
    parser.add_argument('--in-state-loss', choices=['mse', 'cos'], default='mse', help='loss function')
    parser.add_argument('--optim', type=str, default='AdamW', help='optimizer')
    parser.add_argument('--dtype', type=str, default='float32', help='Pytorch dtype for model.')
    parser.add_argument('--init', type=str, default=None, help='initialization for optimization')

    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--verbose', action='store_true', default=False, help='Print detailed output.')
    parser.add_argument('--subfolder-suffix', type=str, default=None, help='Additional description of saved folder.')
    
    args = parser.parse_args()

    #####################
    # Process arguments #
    #####################
    torch.manual_seed(0)
    savefolder = f'{args.range}-{args.attack}-l{args.access_layer_id}-{args.in_state_loss}-lr{args.lr}w1{args.wd_l1}w2{args.wd_l2}wd{args.w_dm}-ep{args.num_steps}'
    if args.attack == 'tbs':
        savefolder += f'-{args.tbs_changevar}'
    if args.init is not None:
        savefolder += f'-{args.init}'
        if args.init in ['randn', 'rand', 'ones', 'zeros']:
            init_fn = getattr(torch, args.init)
        else:
            init_fn = None
    else:
        init_fn = torch.ones if args.attack in ['ts', 'tbs'] else torch.zeros
    
    if args.dtype != 'float32':
        savefolder += f'-{args.dtype}'
    if args.singular:
        savefolder += '-osb'
    if args.subfolder_suffix is not None:
        savefolder += f'-{args.subfolder_suffix}'

    if args.embedding_model:
        savefolder = savefolder.replace(f'-l{args.access_layer_id}', '-embm')
        args.access_layer_id = [-1] 
    else:
        args.access_layer_id = list(map(int, args.access_layer_id.split(',')))
    
    savepath = os.path.join(utils.RESULT_PATH, args.folder, savefolder)
    os.makedirs(savepath, exist_ok=True)
    checkpoint_savepath = os.path.join(savepath, 'checkpoint')
    os.makedirs(checkpoint_savepath, exist_ok=True)
    shutil.copy(__file__, savepath)

    logging.basicConfig(
            filename=os.path.join(savepath, 'main.log'),
            filemode='w',
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            level=logging.DEBUG
        )
    logger = logging.getLogger(__name__)
    json.dump(args.__dict__, open(os.path.join(savepath, 'args.json'), 'w'))

    ######################
    # Load Models (Dual) #
    ######################
    assert args.dtype in ['float16', 'float32', 'bfloat16']
    assert args.target_model in utils.LLM_PATH and args.surrogate_model in utils.LLM_PATH
    
    target_llm_path = utils.LLM_PATH[args.target_model]
    surrogate_llm_path = utils.LLM_PATH[args.surrogate_model]

    modelclass = AutoModel if args.embedding_model else AutoModelForCausalLM
    
    logger.info(f"Loading Target Model from {target_llm_path}...")
    target_model = modelclass.from_pretrained(target_llm_path, torch_dtype=getattr(torch, args.dtype), device_map='auto')
    target_model.eval()
    target_model.requires_grad_(False)
    
    logger.info(f"Loading Surrogate Model from {surrogate_llm_path}...")
    surrogate_model = modelclass.from_pretrained(surrogate_llm_path, torch_dtype=getattr(torch, args.dtype), device_map='auto')
    surrogate_model.eval()
    surrogate_model.requires_grad_(False) 
    
    # 增加兼容 Mistral 架构分词器的安全加载逻辑
    try:
        tokenizer = AutoTokenizer.from_pretrained(target_llm_path, fix_mistral_regex=True)
    except TypeError:
        # 如果模型不是 Mistral 架构，不支持该参数，则回退到普通加载
        tokenizer = AutoTokenizer.from_pretrained(target_llm_path)
        
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    embedding_layer = surrogate_model.get_input_embeddings()
    embedding_cpu_weight = embedding_layer.weight.detach().cpu().clone()

    text_list = utils.get_list_invert_text(args.dataset)
    text_list = sorted(text_list, key=lambda x: len(tokenizer.encode(x, add_special_tokens=False)), reverse=True)

    start, end = args.range.split(':')
    target_texts = text_list[int(start): int(end)]

    inputdict = tokenizer(target_texts, 
                            padding=True ,truncation=True, max_length=128,
                            add_special_tokens=False, return_tensors="pt")
    attention_mask_device = inputdict['attention_mask'].to(args.device)
    testinput_emb_cpu = embedding_layer(inputdict['input_ids'].to(embedding_layer.weight.device)).detach().cpu()

    logger.info('\n'.join(target_texts))
    logger.info('Input Shape:\n')
    logger.info(testinput_emb_cpu.shape)

    ###############################
    # Generate Target State (ISs) #
    ###############################
    logger.info("Extracting Intermediate States (ISs) from Target Model...")
    with torch.no_grad():
        target_input_ids = inputdict['input_ids'].to(target_model.device)
        target_attention_mask = inputdict['attention_mask'].to(target_model.device)
        outputs_target = target_model(input_ids=target_input_ids, attention_mask=target_attention_mask, output_hidden_states=True)
        
        if args.embedding_model:
            target_state = get_embeddings(outputs_target.last_hidden_state, target_attention_mask).to(args.device)
        else:
            target_state = list(outputs_target['hidden_states'])
            for lid in args.access_layer_id:
                target_state[lid] = target_state[lid].detach().to(args.device)

    #####################
    # Initialize Attack #
    #####################

    if args.in_state_loss == 'mse':
        criterion = nn.MSELoss().to(args.device)
    elif args.in_state_loss == 'cos':
        criterion = nn.CosineSimilarity().to(args.device)
    
    try:
        optimizer_class = getattr(optim, args.optim)
    except AttributeError:
        logger.warning(f'No {args.optim}. Use AdamW.')
        optimizer_class = optim.AdamW
    
    torch_dtype = getattr(torch, args.dtype)
    change_var_fn, fn_multiplier = None, None
    
    if args.attack == 'er' or args.attack is None:
        opt_weight_matrix = None
        rweight = init_fn(testinput_emb_cpu.shape, requires_grad=True, device=args.device, dtype=torch_dtype)
        optimizer = optimizer_class([rweight], lr=args.lr, weight_decay=args.wd_l2)
        
    elif args.attack == 'ts':
        opt_weight_matrix = surrogate_model.get_input_embeddings().weight.to(device=args.device, dtype=torch_dtype)
        B = inputdict['input_ids'].shape[0]
        M = inputdict['input_ids'].shape[1]
        N = embedding_cpu_weight.shape[0]
        rweight = init_fn((B, M, N), requires_grad=True, device=args.device, dtype=torch_dtype)
        optimizer = optimizer_class([rweight], lr=args.lr, weight_decay=args.wd_l2)
        
    elif args.attack == 'tbs':
        basis_name = f'{args.surrogate_model}-inputembedding-basevector.pt'
        if basis_name in os.listdir(utils.INPUTEMB_BASEVEC):
            path_to_basisvector = os.path.join(utils.INPUTEMB_BASEVEC, basis_name)
            logger.info('Load Vts from path_to_basisvector.')
            opt_weight_matrix = torch.load(path_to_basisvector, map_location='cpu', weights_only=True)
        else:
            logger.info('Compute Vts for surrogate model.')
            U, s, Vts = svd(embedding_cpu_weight.to(torch.float32).numpy(), full_matrices=False)
            opt_weight_matrix = torch.tensor(Vts)
            os.makedirs(utils.INPUTEMB_BASEVEC, exist_ok=True)
            torch.save(opt_weight_matrix, os.path.join(utils.INPUTEMB_BASEVEC, basis_name))
            
        opt_weight_matrix = opt_weight_matrix.to(device=args.device, dtype=torch_dtype)
        
        if not args.singular:
            logger.info('Use unbiased basis vectors by transposing the singular matrix.')
            opt_weight_matrix = opt_weight_matrix.T
        
        B = inputdict['input_ids'].shape[0]
        M = inputdict['input_ids'].shape[1]
        N = opt_weight_matrix.shape[0]
        # ======== 核心修改：支持高级初始化分布 ========
        if args.init in ['randn', 'rand', 'ones', 'zeros']:
            rweight = getattr(torch, args.init)((B, M, N), device=args.device, dtype=torch_dtype)
        else:
            rweight = torch.ones((B, M, N), device=args.device, dtype=torch_dtype)
            if args.init == 'xav_uni':
                torch.nn.init.xavier_uniform_(rweight)
            elif args.init == 'xav_norm':
                torch.nn.init.xavier_normal_(rweight)
            elif args.init == 'he_norm':
                torch.nn.init.kaiming_normal_(rweight)
        
        # 恢复梯度追踪并进行基础缩放
        rweight.requires_grad = True
        rweight.data /= N
        # ==============================================
        optimizer = optimizer_class([rweight], lr=args.lr, weight_decay=args.wd_l2)

        change_var_fn_name, fn_multiplier = args.tbs_changevar.split(':')
        change_var_fn = utils.CHANGE_VAR_FN_DICT[change_var_fn_name]
        fn_multiplier = float(fn_multiplier)

    losses = []
    minloss = embedding_cpu_weight.shape[1]
    input_vector_cpu_detached_best = None
    torch.cuda.empty_cache()
    
    with trange(args.num_steps, desc=f'{savefolder}-{args.device}', leave=True) as t:
        for step in t:
            optimizer.zero_grad()
            
            input_vector = utils.gen_invert_vector(
                args.attack, optimized_var=rweight, 
                opt_weight_matrix=opt_weight_matrix if args.attack in ['tbs', 'ts'] else None,
                change_var_fn=change_var_fn,
                fn_multiplier=fn_multiplier)
       
            outputs = surrogate_model.model(inputs_embeds=input_vector, 
                            attention_mask=attention_mask_device, 
                            output_hidden_states=True)

            rec_loss = 0
            if args.embedding_model:
                outputs_emb = get_embeddings(outputs.last_hidden_state, attention_mask_device)
                rec_loss = criterion(outputs_emb, target_state)
            else:
                for lid in args.access_layer_id:
                    rec_loss += criterion(outputs['hidden_states'][lid], target_state[lid])

            if args.in_state_loss == 'cos':
                rec_loss = (1 - rec_loss).mean()

            l1loss = args.wd_l1 * torch.norm(rweight, p=1, dim=2).sum(dim=1).mean()
            rec_loss = rec_loss / len(target_texts)

            loss_dm = 0
            if args.w_dm != 0:
                net = get_network(width=input_vector.shape[2]).to(device=args.device, dtype=torch_dtype)
                net.train()
                surrogate_emb_gpu = surrogate_model.get_input_embeddings().weight
                loss_dm = args.w_dm * torch.nn.functional.mse_loss(
                    net.embed(input_vector[:, torch.randint(len(input_vector), (32,)), :]).mean(dim=0),
                    net.embed(surrogate_emb_gpu[torch.randint(len(surrogate_emb_gpu), (32,))]).mean(dim=0))

            all_loss = rec_loss + l1loss.to(rec_loss.device) + loss_dm
            all_loss.backward()
            grad_norm = rweight.grad.norm().item() / len(target_texts)
            optimizer.step()
            
            t1=time.time()
            if minloss > rec_loss.item():
                minloss = rec_loss.item()
                input_vector_cpu_detached_best = utils.gen_invert_vector(
                    attack_type=args.attack, 
                    optimized_var=rweight.detach(), 
                    opt_weight_matrix=opt_weight_matrix.detach() if args.attack in ['tbs', 'ts'] else None,
                    change_var_fn=change_var_fn,
                    fn_multiplier=fn_multiplier).cpu()
            
            losses.append(rec_loss.item())
            if (step + 1) % max(args.num_steps / 100, 1) == 0:
                logger.info(f'Epoch [{step+1}/{args.num_steps}], Loss:{rec_loss.item():.8e}. GN:{grad_norm:.8e}')
            
            if (step + 1) % max(args.num_steps / 10, 1) == 0:
                input_vector_cpu_detached = utils.gen_invert_vector(
                    attack_type=args.attack, 
                    optimized_var=rweight.detach(), 
                    opt_weight_matrix=opt_weight_matrix.detach() if args.attack in ['tbs', 'ts'] else None,
                    change_var_fn=change_var_fn,
                    fn_multiplier=fn_multiplier).cpu()
                recovered_ids = utils.check_results(input_vector_cpu_detached, embedding_cpu_weight)
                recovered_text = tokenizer.batch_decode(recovered_ids)
                torch.save({'invert_vector':input_vector_cpu_detached, 'invert_ids':recovered_ids, 'invert_text': recovered_text, 'rweight': rweight.detach().cpu() }, os.path.join(checkpoint_savepath, f'invert-{step+1}.pt'))
            
            t.set_postfix(loss=f'{rec_loss.item():.4e}', gn=f'{grad_norm:.4e}', time=f'{time.time()-t1:.2e}')

    ########################################
    # Evaluation and Padding Normalization #
    ########################################
    recovered_ids = utils.check_results(input_vector_cpu_detached_best, embedding_cpu_weight)
    
    cleaned_predictions_ids = []
    cleaned_predictions_str = []
    cleaned_references_ids = []
    cleaned_references_str = []
    
    for i in range(len(recovered_ids)):
        pred_ids_list = recovered_ids[i].tolist()
        if tokenizer.pad_token_id in pred_ids_list:
            pred_ids_list = pred_ids_list[:pred_ids_list.index(tokenizer.pad_token_id)]
        cleaned_predictions_ids.append(pred_ids_list)
        cleaned_predictions_str.append(tokenizer.decode(pred_ids_list, skip_special_tokens=True))
        
        ref_ids_list = inputdict['input_ids'][i].tolist()
        if tokenizer.pad_token_id in ref_ids_list:
            ref_ids_list = ref_ids_list[:ref_ids_list.index(tokenizer.pad_token_id)]
        cleaned_references_ids.append(ref_ids_list)
        cleaned_references_str.append(tokenizer.decode(ref_ids_list, skip_special_tokens=True))

    logger.info("=== Output Evaluation ===")
    logger.info(f"Target Text Sample: {cleaned_references_str[0]}")
    logger.info(f"Recovered Text Sample: {cleaned_predictions_str[0]}")

    evaluator = utils.Eval(tokenizer=tokenizer, device=args.device)
    metrics = evaluator._text_comparison_metrics(
        predictions_ids=cleaned_predictions_ids,
        predictions_str=cleaned_predictions_str,
        references_ids=cleaned_references_ids,
        references_str=cleaned_references_str
    )
    
    # 彻底阻断 JSON 序列化崩溃：主动拆解所有 NumPy 和 Tensor 类型
    for k, v in metrics.items():
        if hasattr(v, 'item'):
            metrics[k] = v.item()
        else:
            try:
                metrics[k] = float(v)
            except (TypeError, ValueError):
                pass
    
    logger.info(f"Metrics Report: \n{json.dumps(metrics, indent=4)}")

    torch.save({
        'L': torch.tensor(losses), 
        'invert_vector': input_vector_cpu_detached_best, 
        'invert_ids': recovered_ids, 
        'invert_text': cleaned_predictions_str, 
        'rweight': rweight.detach().cpu(),
        'evaluation_metrics': metrics
    }, os.path.join(savepath, 'invert-best.pt'))

if __name__ == "__main__":
    main()