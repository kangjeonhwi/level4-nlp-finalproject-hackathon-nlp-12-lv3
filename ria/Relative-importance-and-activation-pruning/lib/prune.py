import time 
import torch 
import torch.nn as nn 
import os
from .sparsegpt import SparseGPT 
from .layerwrapper import WrappedGPT
from .data import get_loaders 
import numpy as np
import torch.nn as nn
from torch import autocast
from tqdm import tqdm
from transformers.models.llama.modeling_llama import LlamaAttention
from pdb import set_trace as st 
from .quant import *
import numpy as np
from scipy.optimize import linear_sum_assignment
from torch.sparse import to_sparse_semi_structured, SparseSemiStructuredTensor
from torch.utils.data import DataLoader, TensorDataset
from datasets import Dataset, load_dataset

            
        
            


def lexsort(keys, dim=-1):
    idx = keys[0].argsort(dim=dim, stable=True)
    for k in keys[1:]:
        idx = idx.gather(dim, k.gather(dim, idx).argsort(dim=dim, stable=True))
    
    return idx


def maximize_total_value(matrix):
    # linear_sum_assignment
    row_indices, col_indices = linear_sum_assignment(matrix, maximize=True) 
    return col_indices


def find_layers(module, layers=[nn.Linear], name=''):
    """
    역할: 모델 내부의 특정 계층(nn.Linear)를 재귀적으로 탐색하여 찾는 함수이다.
    사용 목적: 모델 희소화를 적용할 때 대상 계층 (주로 선형 계층)을 특정하기 위해 사용한다.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def check_sparsity(args, model):
    '''
    역할: 모델의 희소성을 계산한다. 각 계층의 0으로 설정된 가중치 비율(희소성 비율)을 출력하고, 전체 모델의 희소도를 반환한다.
    주요 작업: 모든 계층의 가중치(weight.data)에서 0으로 설정된 요소의 비율을 계산하고, 모델의 모든 계층의 희소성을 종합하여 출력한다.
    '''
    use_cache = model.config.use_cache 
    model.config.use_cache = False 
    if "llama" in args.model:
        layers = model.model.layers
    elif "opt" in args.model:
        layers = model.model.decoder.layers
    count = 0 
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            if args.semi_sparse_acc:
                W = subset[name].mask
                
            else:
                W = subset[name].weight.data
            count += (W==0).sum().item()
            total_params += W.numel()

            sub_count += (W==0).sum().item()
            sub_params += W.numel()

        print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

    model.config.use_cache = use_cache 
    return float(count)/total_params 

def prepare_calibration_input(args, model, dataloader, device):
    print("prepare_calibration_input 시작")
    use_cache = model.config.use_cache
    model.config.use_cache = False
    if "llama" in args.model:
        layers = model.model.layers
        # dev = model.hf_device_map["model.embed_tokens"]
        if "model.embed_tokens" in model.hf_device_map:
            device = model.hf_device_map["model.embed_tokens"]
    elif "opt" in args.model:
        layers = model.model.decoder.layers


    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, args, module):
            super().__init__()
            self.module = module
            self.model = args.model
        def forward(self, inp, **kwargs):
            if cache['i'] < inps.size(0):
                inps[cache['i']] = inp[cache['i']]
                cache['i'] += 1
            if kwargs['attention_mask'] is not None:
                cache['attention_mask'] = kwargs['attention_mask']
            else: 
                cache['attention_mask'] = torch.ones(inps.size(1), inps.size(2), dtype=torch.int64, device=device)
            if "llama" in args.model and kwargs['position_ids'] is not None:
                cache['position_ids'] = kwargs['position_ids']
            else: 
                cache['position_ids'] = torch.arange(88, dtype=torch.long, device=device).unsqueeze(0)
            raise ValueError

    layers[0] = Catcher(args, layers[0])
    
    for b_idx, batch in enumerate(dataloader):
        try:
            data_list = batch["data"]
            data_tensor = torch.tensor(data_list, dtype=torch.long).to(device)
            model(data_tensor)
        except ValueError:
            pass
    
    layers[0] = layers[0].module
    # print(inps)
    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
   
    model.config.use_cache = use_cache
    if "llama" in args.model:
        position_ids = cache['position_ids']
        return inps, outs, attention_mask, position_ids 
    elif "opt" in args.model:
        return inps, outs, attention_mask


def prune_magnitude(args, model, tokenizer, device=None, prune_n=0, prune_m=0):
    '''
    역할: 가중치의 크기(magnitude)를 기반으로 한 unstructured pruning(비구조적 희소화)를 수행한다. 
    주요 작업:
        1. torch.abs()를 사용하여 가중치의 절대값을 계산한다.
        2. 기준 임계값(threshold) 이하의 가중치는 0으로 설정한다.
        3. 희소화 비율(sparsity_ratio)를 기반으로 구조적 pruning(n:m sparsity)을 적용한다.
    '''
    
    # device 설정
    if device is None: 
        if torch.cuda.is_available(): 
            device = torch.device("cuda")
            print("Using CUDA")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using MPS")
        else: 
            device = torch.device("cpu")
            print("Using CPU")
    
    # 모델 레이어 설정
    if "llama" in args.model:
        layers = model.model.layers
    elif "opt" in args.model:
        layers = model.model.decoder.layers
        
    per_outneuron = False

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        for name in subset:
            print(f"pruning layer {i} name {name}")
            W = subset[name].weight.data.clone().to(device)
            if args.prune_method == "magnitude":
                W_metric = torch.abs(W)
            elif args.prune_method == "ri":
                W_metric = torch.abs(W)/torch.sum(torch.abs(W), dim=0) + torch.abs(W)/torch.sum(torch.abs(W), dim=1).reshape(-1, 1)
            W_mask = torch.zeros_like(W_metric, dtype=torch.bool, device=device)
            if prune_n != 0:
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                if per_outneuron:
                    sort_res = torch.sort(W_metric, dim=-1, stable=True)
                    # unstructured pruning
                    indices = sort_res[1][:,:int(W_metric.shape[1]*args.sparsity_ratio)]
                    W_mask.scatter_(1, indices, True)
                else:
                    thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.shape[0]* W.shape[1]*args.sparsity_ratio)].cpu()
                    W_mask = (W_metric<=thresh)

            subset[name].weight.data[W_mask] = 0
            
def load_pt_data(pt_dir, num_files=None):
    """
    pt 데이터를 datasets 라이브러리를 사용해 로드하는 함수
    pt_dir: pt 데이터가 있는 디렉토리
    num_files: 불러올 파일의 개수 (None이면 모든 파일을 불러옴)

    return
        Hugging Face datasets.Dataset 객체
    """
    print(f"Loading .pt files from directory: {pt_dir}")
    pt_files = [f for f in os.listdir(pt_dir) if f.endswith('.pt')]

    if num_files:
        pt_files = pt_files[:num_files]

    # .pt 파일 데이터를 저장할 리스트
    all_data = []
    
    for pt_file in pt_files:
        file_path = os.path.join(pt_dir, pt_file)
        print(f"Loading file: {file_path}")
        data = torch.load(file_path)
        
        # 데이터가 Tensor일 경우 리스트로 변환
        if isinstance(data, torch.Tensor):
            data = data.tolist()
        all_data.extend(data)
    
    print(f"Loaded {len(all_data)} samples from .pt files.")

    # 데이터를 datasets 객체로 변환
    dataset = Dataset.from_dict({"data": all_data})
    return dataset


def prepare_inputs_from_pt(pt_data, device):
    '''
    .pt 데이터를 모델의 입력으로 변환한다.
    '''
    # 예제 데이터 변환: inps, outs, attention_mask 생성
    inps = pt_data[:, :, :-1].to(device) # 마지막 토큰 제외 (입력)
    outs = pt_data[:, :, 1:].to(device) # 첫 번째 토큰 제외 (출력)
    attention_mask = torch.ones(inps.shape[:2], dtype=torch.int64, device=device)  # Attention mask 생성
    
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
        
    return inps, outs, attention_mask

# Forward hook 함수 정의
def position_embeddings_hook(module, input, output):
    # position_embeddings은 일반적으로 내부적으로 계산되므로, 여기서 직접 접근하기는 어렵습니다.
    # 대신, 오류가 발생하기 전에 position_embeddings를 생성하거나 수정할 수 있는 방법을 모색해야 합니다.
    # 이 예시에서는 position_ids가 올바르게 전달되었는지 확인하고, 필요시 position_embeddings를 설정합니다.
    
    # position_ids는 input[2]에 위치할 수 있음 (input의 구조에 따라 다름)
    position_ids = input[2] if len(input) > 2 else None
    if position_ids is not None:
        # position_embeddings 생성 로직 강제 실행 (rotary_emb 사용)
        rotary_dim = module.rotary_emb.rotary_dim
        position_embeddings = module.rotary_emb(position_ids)
        if position_embeddings is None:
            # position_embeddings가 여전히 None인 경우, 기본값 설정
            position_embeddings = (torch.zeros_like(module.rotary_emb(freqs).cos), torch.zeros_like(module.rotary_emb(freqs).sin))
        
        # cos, sin 값을 반환하는 output을 수정
        return position_embeddings
    else:
        # position_ids가 없으면 기본값 설정
        rotary_dim = module.rotary_emb.rotary_dim
        position_embeddings = (torch.zeros((module.rotary_emb.freqs.shape[0], module.rotary_emb.freqs.shape[1]), device=module.rotary_emb.freqs.device),
                               torch.zeros((module.rotary_emb.freqs.shape[0], module.rotary_emb.freqs.shape[1]), device=module.rotary_emb.freqs.device))
        return position_embeddings
    

def prune_ria(args, model, tokenizer, device=None, prune_n=0, prune_m=0):
    '''
    역할: RIA(Relative Importance and Activations) 알고리즘을 사용해 모델의 가중치를 pruning한다.
    주요 특징: 
        상대적 중요도(Relative Importance): 가중치의 행 및 열의 합을 기준으로 중요도를 계산한다.
        활성화 값 (Activation): 활성화 스케일링(scaler_row)을 반영하여 pruning 기준을 계산한다.
        Heuristic Channel Reallocation(휴리스틱 채널 재할당) 및 Linear Sum Assignmenr(LSA)와 같은 고급 최적화 기법을 지원한다.
    '''
    inps = None
    
    # device 설정
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("Using CUDA")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using MPS on macOS")
        else:
            device = torch.device("cpu")
            print("Using CPU")
        
    if args.pt_files_dir:
        print(f"loading .pt files from {args.pt_files_dir}")
        data_loader = load_pt_data(args.pt_files_dir)
        print(f"Loaded {len(data_loader)} samples from .pt files")
    else:
        data_loader = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    
    with torch.no_grad():
        if "llama" in args.model:
            inps, outs, attention_mask, position_ids = prepare_calibration_input(args, model, data_loader, device)
        elif "opt" in args.model:
            inps, outs, attention_mask= prepare_calibration_input(args, model, data_loader, device)
    if "llama" in args.model:
        layers = model.model.layers
    elif "opt" in args.model:
        layers = model.model.decoder.layers
        
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        if "llama" in args.model:
            if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
                dev = model.hf_device_map[f"model.layers.{i}"]
                # inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)
                inps, outs, position_ids = inps.to(dev), outs.to(dev), position_ids.to(dev)
        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(args, subset[name], layer_name=name, reconstruct=args.reconstruction)
            if args.gptq:
                wrapped_layers[name].quantizer = Quantizer()
                wrapped_layers[name].quantizer.configure(
                        args.wbits, perchannel=True, sym=args.sym, mse=False
                    )

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for name, module in model.named_modules():
            if isinstance(module, LlamaAttention):
                module.register_forward_hook(position_embeddings_hook)
                
        if inps.dtype != torch.long and inps.dtype != torch.int:
            inps = inps.long()
                
        for j in range(args.nsamples):
            # 디버깅 메시지
            print(f"inps[{j}] size: {inps[j].size()}")
            print(f"attention_mask size: {attention_mask.size()}")
            print(f"position_ids size: {position_ids.size()}")
            
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(0).unsqueeze(1)  # [1, 1, 2048, 3072]
                attention_mask = attention_mask[:, :, :, :inps.size(1)]  # [1, 1, 1, seq_len]
                attention_mask = attention_mask.to(dtype=torch.float)
                print("Adjusted attention_mask size: ", attention_mask.size())
                
            if position_ids.size(0) != inps.size(0):  # 배치 차원 일치 확인
                position_ids = position_ids.expand(inps.size(0), -1)  # [batch_size, seq_len]
                print("Adjusted position_ids size: ", position_ids.size())
                
            attention_mask = attention_mask[:, :, :2048, :2048]  # Example for reducing size
            position_ids = position_ids[:, :2048]
            
            with torch.no_grad():
                if "llama" in args.model:
                    outputs = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
                    outs[j] = outputs[0]
                elif "opt" in args.model:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        for h in handles:
            h.remove()

        for name in subset:
            if args.gptq:
                print('Quantizing ...')
                wrapped_layers[name].fasterquant(
                    percdamp=args.percdamp, groupsize=args.groupsize, actorder=args.act_order, static_groups=args.static_groups
                )
            
            print(f"pruning layer {i} name {name}")
            W = subset[name].weight.data.clone()
            if args.prune_method == "wanda":
                W_metric = torch.abs(W) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
            elif args.prune_method == "ria":
                W_metric = (torch.abs(W)/torch.sum(torch.abs(W), dim=0) + torch.abs(W)/torch.sum(torch.abs(W), dim=1).reshape(-1, 1)) * (torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1))))**args.a
            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
            if prune_n != 0:
                # structured n:m sparsity
                if args.reallocation:
                    """
                        Using Heuristic Channel Reallocation
                    """
                    
                    # Try with directly N:M sparsity
                    for ii in range(W_metric.shape[1]):
                        if ii % prune_m == 0:
                            tmp = W_metric[:,ii:(ii+prune_m)].float()
                            W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
                    
                    pre_score = torch.sum(W_metric[W_mask==0].type(torch.float32)).item()
                    print("The total value before resort: ", pre_score)
                    
                    
                    # assign importance score to each columns
                    if args.importance_score == "sum":
                        # sum the total value of each column
                        sorted_idx = torch.sort(torch.sum(W_metric, dim=0))[1]
                    elif args.importance_score == "retained_degree_unstructured":
                        # try unstructured pruning
                        thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.shape[0]* W.shape[1]*args.sparsity_ratio)].cpu()
                        W_mask = (W_metric<=thresh)
                        keys = [torch.sum(W_mask, dim=0), torch.sum((W_mask==0)*W_metric, dim=0)]
                        sorted_idx = lexsort(keys)
                    elif args.importance_score == "retained_degree_per_outneuron":
                        # try unstructured pruning with per output neuron pruning
                        sort_res = torch.sort(W_metric, dim=-1, stable=True)
                        indices = sort_res[1][:,:int(W_metric.shape[1]*args.sparsity_ratio)]
                        W_mask = torch.zeros_like(W_metric)==1
                        W_mask.scatter_(1, indices, True)
                        
                        keys = [torch.sum(W_mask, dim=0), torch.sum((W_mask==0)*W_metric, dim=0)]
                        sorted_idx = lexsort(keys)
                    
                    # channel reallocation
                    index = torch.zeros_like(sorted_idx)
                    for ii in range(1, prune_m+1):
                        if ii % 2 == 1:
                            index[ii-1::prune_m] = sorted_idx[int(W_metric.shape[1]* (ii-1)/prune_m) :int(W_metric.shape[1]* ii/prune_m)]
                        else:
                            index[ii-1::prune_m] = sorted_idx[int(W_metric.shape[1]* (ii-1)/prune_m) :int(W_metric.shape[1]* ii/prune_m)].flip(0)
                        # index[ii-1::prune_m] = sorted_idx[int(W_metric.shape[1]* (ii-1)/prune_m) :int(W_metric.shape[1]* ii/prune_m)]
                    W_metric_resort = W_metric[:, index].clone()
                    
                    W_strip_value = torch.zeros(W_metric.shape[1]//prune_m).to(device)
                    W_mask_permute = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
                    for ii in range(W_metric.shape[1]):
                        if ii % prune_m == 0:
                            tmp = W_metric_resort[:,ii:(ii+prune_m)].float()
                            W_mask_permute.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
                            W_metric_strip = W_metric_resort[:, ii:(ii+prune_m)]
                            W_strip_value[ii//prune_m] = torch.sum(W_metric_strip[W_mask_permute[:, ii:(ii+prune_m)]==0])
                        
                    after_score = torch.sum(W_strip_value.type(torch.float32)).item()
                    print("The total value after heuristic channel reallocation: ", after_score)
                    
                    if args.lsa:
                        """
                            Using linear sum assignment to finetune the N:M
                        """
                        permutation_device = "cuda:7"
                        if args.fast:
                            print("Use Fast!!")
                            fast_name_list = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"]
                            if name in fast_name_list:
                                blocks = 4
                            elif "up_proj" in name or "gate_proj" in name:
                                blocks = 8
                            else:
                                blocks = 16
                        else:
                            blocks = 1
                        

                        shape = W_metric.shape[1]//prune_m//blocks
                        rows = torch.arange(shape).to(permutation_device)
                        lsa_columns = torch.arange(prune_m).to(permutation_device)
                        def lsa(W_metric, lsa_column, shape, rows, prune_n, prune_m, device):
                            W_metric = W_metric.to(device)
                            score_matrix = torch.zeros(shape, shape).to(device) # score matrix of LSA
                            num_parallel = 1 # How many parallel computation will be used.
                            
                            
                            for row in range(shape//num_parallel):
                                strip_idx = torch.zeros(num_parallel, shape, prune_m).long().to(device)
                                block_columns = torch.arange(prune_m).to(device)
                                columns_mask = block_columns != lsa_column
                                block_columns = block_columns[columns_mask]
                                
                                strip_idx[:, :, 0] = (rows * prune_m).reshape(1, -1) + lsa_column
                                strip_idx[:, :, 1:] = block_columns.reshape(1, 1, -1) + torch.arange(row*num_parallel, (row+1)*num_parallel).reshape(-1, 1, 1).to(device) * prune_m
                                
                                tmp = W_metric[:, strip_idx].transpose(1, 0).transpose(2, 1)
                                
                                W_mask = torch.zeros_like(tmp).to(device)
                                
                                
                                
                                tmp_index = torch.sort(tmp, dim=-1)[1]
                                W_mask.scatter_(dim=-1, index=tmp_index[:, :, :, :prune_n], value=1)
                    
                                score_matrix[:, row*num_parallel:(row+1)*num_parallel] = torch.sum(torch.sum((tmp*(W_mask==0)), dim=-1), dim=-1).transpose(1, 0)
                            
                            score_matrix = score_matrix.transpose(1, 0)
                            
                            col_indices = torch.LongTensor(maximize_total_value(score_matrix.cpu())).to(device)
                            idx = torch.arange(W_metric.shape[1]).long().to(device)
                            idx[rows* prune_m + lsa_column] = col_indices * prune_m + lsa_column
                            
                            return idx
                        
                        z = 0
                        for lsa_column in lsa_columns:
                            t1 = time.time()
                            for ii in range(blocks):
                                index_tmp = index[ii*len(index)//blocks:(ii+1)*len(index)//blocks]
                                permute_idx = lsa(W_metric[:, index_tmp], lsa_column, shape, rows, prune_n, prune_m, permutation_device)
                                permute_idx = permute_idx.to(index.device)

                                index[ii*len(index)//blocks:(ii+1)*len(index)//blocks] = index_tmp[permute_idx]
                            t2 = time.time()
                            W_metric_permute = W_metric[:, index]
                            
                            W_mask_permute = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
                            for ii in range(W_metric.shape[1]):
                                if ii % prune_m == 0:
                                    tmp = W_metric_permute[:,ii:(ii+prune_m)].float()
                                    W_mask_permute.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
                                    W_metric_strip = W_metric_permute[:, ii:(ii+prune_m)]
                                    W_strip_value[ii//prune_m] = torch.sum(W_metric_strip[W_mask_permute[:, ii:(ii+prune_m)]==0])
                            print("The total value after linear sum assignment round {}: {}, running time: {}s".format(z, torch.sum(W_strip_value.type(torch.float32)).item(), round(t2-t1, 2)))
                            
                            z += 1
                        
                        
                    W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
                    W_mask[:, index] = W_mask_permute
                    
                    if args.semi_sparse_acc and prune_n == 2 and prune_m == 4:
                        subset[name].weight = torch.nn.Parameter(to_sparse_semi_structured((W_mask_permute==0)*W[:, index].half()))
                        subset[name].mask = W_mask_permute==0
                    else:
                        subset[name].weight.data[W_mask] = 0

                        
                else:
                    # Directly N:M
                    W_mask = (torch.zeros_like(W_metric) == 1)
                    for ii in range(W_metric.shape[1]):
                        if ii % prune_m == 0:
                            tmp = W_metric[:,ii:(ii+prune_m)].float()
                            W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
                    
                    if args.semi_sparse_acc:
                        subset[name].weight = torch.nn.Parameter(to_sparse_semi_structured(((W_mask==0)*W)).half(), requires_grad=False)
                        subset[name].mask = W_mask==0
                    else:
                        subset[name].weight.data[W_mask] = 0
            else:
                if args.per_outneuron:
                    sort_res = torch.sort(W_metric, dim=-1, stable=True)
                    # unstructured pruning
                    indices = sort_res[1][:,:int(W_metric.shape[1]*args.sparsity_ratio)]
                    W_mask.scatter_(1, indices, True)
                else:
                    thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.shape[0]* W.shape[1]*args.sparsity_ratio)].cpu()
                    W_mask = (W_metric<=thresh)
                    
                if args.reconstruction:
                    wrapped_layers[name].fasterprune(args.sparsity_ratio, mask=W_mask)
                else:
                    subset[name].weight.data[W_mask] = 0  ## set weights to zero 
            wrapped_layers[name].free()

        for j in range(args.nsamples):
            with torch.no_grad():
                if "llama" in args.model:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
                elif "opt" in args.model:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()


@torch.no_grad()
def prune_sparsegpt(args, model, tokenizer, dev, prune_n=0, prune_m=0):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    '''
    역할: SparseGPT 알고리즘을 사용해 희소화 작업을 수행한다.
    주요 작업: 
        SparseGPT는 GPT 계열 모델에 특화된 pruning 알고리즘으로, 빠른 pruning을 지원
        활성화 값을 수집하고, 가중치를 평가하여 희소화를 적용
    '''
    
    
    print('Starting ...')
    dataloader, _ = get_loaders(args.calib_dataset, nsamples=args.nsamples,seed=args.seed,seqlen=args.seqlen,tokenizer=tokenizer)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    if "llama" in args.model:
        layers = model.model.layers
    elif "opt" in args.model:
        layers = model.model.decoder.layers

    if "model.embed_tokens" in model.hf_device_map:
        dev = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {
        'i': 0, 
        'attention_mask': torch.ones((1, 88), dtype=torch.long, device='cuda'), 
        'position_ids': torch.arange(88, dtype=torch.long, device='cuda').unsqueeze(0)
    }

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            if kwargs['attention_mask'] is not None:
                cache['attention_mask'] = kwargs['attention_mask']
            else:
                cache['attention_mask'] = torch.ones((1, 88), device=device)
            if "llama" in args.model and kwargs['position_ids'] is not None:
                cache['position_ids'] = kwargs['position_ids']
            else:
                cache['position_ids'] = torch.arange(88, dtype=torch.long, device=device).unsqueeze(0)
                
            print("cache in gpt: ", cache)
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    if cache['attention_mask'] is not None:
        attention_mask = cache['attention_mask']
    if cache['position_ids'] is not None:
        position_ids = cache['position_ids']

    print('Ready.')

    for i in range(len(layers)):
        layer = layers[i]
        if "llama" in args.model:
            if f"model.layers.{i}" in model.hf_device_map:
                dev = model.hf_device_map[f"model.layers.{i}"]
                print(f"layer {i} device {dev}")
                inps, outs, position_ids = inps.to(dev), outs.to(dev), position_ids.to(dev)

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = SparseGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
            
        # 모델 호출 (최상위 forward 사용)
        with torch.no_grad():
            inps = inps.to(torch.long)
            attention_mask = attention_mask.to(torch.long)
            position_ids = position_ids.to(torch.long)
            outputs = model(input_ids=inps, attention_mask=attention_mask, position_ids=position_ids)

        for h in handles:
            h.remove()

        for name in gpts:
            print('Pruning ...')
            if "norm" in args.model:
                gpts[name].fasterprune(args.sparsity_ratio, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128, norm=True)
            else:
                gpts[name].fasterprune(args.sparsity_ratio, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
            gpts[name].free()

        for j in range(args.nsamples):
            if "llama" in args.model:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            else:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        layers[i] = layer 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()