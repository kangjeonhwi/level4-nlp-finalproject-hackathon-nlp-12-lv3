import time 
import heapq 
import torch 
import glob
import os
import torch.nn as nn 
from tqdm import tqdm
from .sparsegpt import SparseGPT 
from .layerwrapper import WrappedGPT
from .data import get_loaders
from torch.utils.data import Dataset, DataLoader

from .ablate import AblateGPT

class PTPairDataset(Dataset):
    """
    두 개의 디렉토리에서 각각 .pt 파일들을 읽어, 
    입력(임베딩) 텐서와 attention_mask 텐서를 튜플로 반환하는 Dataset 클래스.
    """
    def __init__(self, inputs_dir, attn_dir):
        # 두 디렉토리 내의 .pt 파일 목록을 정렬하여 가져옵니다.
        self.input_files = sorted(glob.glob(os.path.join(inputs_dir, "*.pt")))
        self.attn_files = sorted(glob.glob(os.path.join(attn_dir, "*.pt")))

        if len(self.input_files) != len(self.attn_files):
            raise ValueError("입력 파일 수와 attention 파일 수가 일치하지 않습니다.")
    
    def __len__(self):
        return len(self.input_files)
    
    def __getitem__(self, idx):
        # 각 파일은 이미 저장된 텐서, 예를 들어 입력은 [1, seq_len, hidden_size] 
        # attention_mask는 [1, seq_len] 또는 [1, 1, seq_len] 등으로 저장되어 있다고 가정합니다.
        inp = torch.load(self.input_files[idx])
        attn = torch.load(self.attn_files[idx])
        return inp, attn

def custom_collate_fn(batch, pad_value=0):
    """
    batch: [(inp, attn), (inp, attn), ...]
      - 각 inp: [1, seq_len, hidden_size]
      - 각 attn: [1, seq_len] 또는 [1, 1, seq_len]
    
    모든 샘플의 시퀀스 길이를 동일하게 (max_seq_len) 패딩하여 배치로 만듭니다.
    """
    # 배치 내 각 샘플의 시퀀스 길이 추출 (여기서는 inp의 두 번째 차원이 시퀀스 길이)
    seq_lens = [item[0].shape[1] for item in batch]
    max_seq_len = 2048
    
    padded_inps = []
    padded_attns = []
    
    for inp, attn in batch:
        # inp의 shape: [1, seq_len, hidden_size]
        pad_len = max_seq_len - inp.shape[1]
        if pad_len > 0:
            # [1, pad_len, hidden_size]
            pad_tensor = torch.full((inp.shape[0], pad_len, inp.shape[2]), pad_value, dtype=inp.dtype, device=inp.device)
            inp = torch.cat([inp, pad_tensor], dim=1)
        padded_inps.append(inp)
        
        # attn의 shape: [1, seq_len] 또는 [1, 1, seq_len]
        if attn.dim() == 2:
            pad_attn = torch.full((attn.shape[0], pad_len), pad_value, dtype=attn.dtype, device=attn.device)
            attn = torch.cat([attn, pad_attn], dim=1)
        elif attn.dim() == 3:
            pad_attn = torch.full((attn.shape[0], attn.shape[1], pad_len), pad_value, dtype=attn.dtype, device=attn.device)
            attn = torch.cat([attn, pad_attn], dim=2)
        padded_attns.append(attn)
    
    # 배치 차원으로 concatenate: 각 inp는 [1, max_seq_len, hidden_size]
    batch_inps = torch.cat(padded_inps, dim=0)  # 최종 shape: [batch, max_seq_len, hidden_size]
    batch_attns = torch.cat(padded_attns, dim=0)  # shape은 attn에 따라 결정됨.
    
    return batch_inps, batch_attns

def get_rotary_embedding(seq_len, head_dim, base=10000, device='cuda:0'):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    positions = torch.arange(seq_len, device=device).unsqueeze(1).float()
    sinusoid_inp = positions * inv_freq
    sin = torch.sin(sinusoid_inp)
    cos = torch.cos(sinusoid_inp)  
    sin = torch.repeat_interleave(sin, 2, dim=-1)
    cos = torch.repeat_interleave(cos, 2, dim=-1)
    return cos.half().unsqueeze(0), sin.half().unsqueeze(0)

def load_samples_from_directory(directory, file_pattern="*.pt", nsamples=128, map_location="cuda", pad_token=0.0):
    file_paths = glob.glob(os.path.join(directory, file_pattern))
    file_paths.sort()
    file_paths = file_paths[:nsamples]
    
    samples = []
    max_seq_length = 0
    
    for file_path in file_paths:
        sample = torch.load(file_path, map_location=map_location)
        samples.append(sample)
        max_seq_length = max(max_seq_length, sample.shape[1])
        
    padded_samples = []
    for sample in samples: 
        seq_length = sample.shape[1]
        pad_size = max_seq_length - seq_length
        
        if pad_size > 0:
            if sample.ndim == 3:
                pad_tensor = torch.full((sample.shape[0], pad_size, sample.shape[2]), pad_token, dtype=sample.dtype, device=sample.device)
            elif sample.ndim == 2:
                pad_tensor = torch.full((sample.shape[0], pad_size), pad_token, dtype=sample.dtype, device=sample.device)
            else:
                raise ValueError("Unsupported tensor ndim: {}".format(sample.ndim))
                
            padded_sample = torch.cat([sample, pad_tensor], dim=1)
        else: 
            padded_sample = sample
        padded_samples.append(padded_sample)
        
    data = torch.stack(padded_samples, dim=0)
    return data

def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def check_sparsity(model):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    layers = model.model.layers
    count = 0 
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W==0).sum().item()
            total_params += W.numel()

            sub_count += (W==0).sum().item()
            sub_params += W.numel()

        print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

    model.config.use_cache = use_cache 
    return float(count)/total_params 

def prepare_calibration_input(model, dataloader, device):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    # dev = model.hf_device_map["model.embed_tokens"]
    if "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    first_batch = next(iter(dataloader))
    sample_inp, _ = first_batch
    seq_len = sample_inp.shape[2]
    
    inps = torch.zeros((128, 2048, model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None, 'position_embeddings': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inp = inp.squeeze(0).squeeze(0)
            inps[cache['i']] = inp
            cache['i'] += 1
            attn = kwargs.get('attention_mask')
            if attn is None:
                attn = torch.ones(
                    inp.shape[0], 
                    inp.shape[1], 
                    device=inp.device, 
                    dtype=torch.long)
            cache['attention_mask'] = attn
            cache['position_ids'] = kwargs.get(
                'position_ids', 
                torch.arange(inp.shape[1], device=inp.device)
                .unsqueeze(0)
                .expand(inp.shape[0], inp.shape[1])
            )
            cache['position_embeddings'] = kwargs.get('position_embeddings')
            raise ValueError
        
    original_first_layer = layers[0]
    layers[0] = Catcher(layers[0])
        
    for batch in dataloader:
        inp, attn = batch
        try:
            while inp.dim() > 2 and inp.size(0) == 1:
                inp = inp.squeeze(0)
            layers[0](inp.to(device), attention_mask=attn.to(device))
        except ValueError:
            pass
        if cache['i'] >= inps.shape[0]:
            break
    layers[0] = original_first_layer

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    position_embeddings = cache['position_embeddings']
    model.config.use_cache = use_cache
    
    if position_embeddings is None:
        rotary_dim = getattr(model.config, "rotary_dim", model.config.hidden_size // model.config.num_attention_heads)
        q_rotary_dim = 32
        k_rotary_dim = 8
        max_length = 2048
        cos, sin = get_rotary_embedding(max_length, rotary_dim, device=device)
        cos = cos.half()
        sin = sin.half()
        
        cos_q = cos[..., :q_rotary_dim]  # [1, 2048, 32]
        sin_q = sin[..., :q_rotary_dim]  # [1, 2048, 32]
        cos_k = cos[..., :k_rotary_dim]  # [1, 2048, 8]
        sin_k = sin[..., :k_rotary_dim]  # [1, 2048, 8]
        
        position_embeddings = ((cos_q, cos_k), (sin_q, sin_k))

    return inps, outs, attention_mask, position_ids, position_embeddings

    
        

def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    thres_cumsum = sum_before * alpha
    thres_cumsum_reshaped = thres_cumsum.reshape((-1, 1))
    sort_mask = tmp_metric <= thres_cumsum_reshaped
        
    thres = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True)-1)
    W_mask = (W_metric <= thres)
    cur_sparsity = (W_mask==True).sum() / W_mask.numel()
    
    return W_mask, cur_sparsity

def prune_magnitude(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    layers = model.model.layers 

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        for name in subset:
            W = subset[name].weight.data 
            W_metric = torch.abs(W)
            if prune_n != 0:
                W_mask = (torch.zeros_like(W)==1)
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.numel()*args.sparsity_ratio)].cpu()
                W_mask = (W_metric<=thresh)

            W[W_mask] = 0

def prune_wanda(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    inps_directory = "/data/home/psj/level4-nlp-finalproject-hackathon-nlp-12-lv3/wanda/wanda/datasets/output_inps_1"
    attn_directory = "/data/home/psj/level4-nlp-finalproject-hackathon-nlp-12-lv3/wanda/wanda/datasets/output_atts_1"
    
    dataset = PTPairDataset(inps_directory, attn_directory)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)
    
    inps, outs, attention_mask, position_ids, position_embeddings = prepare_calibration_input(model, dataloader, device)
    
    layers = model.model.layers
    for i in tqdm(range(len(layers)), desc="레이어: "):
        layer = layers[i]
        subset = find_layers(layer)

        if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]                
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
    
        batch_size, seq_len = attention_mask.shape 
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.expand(batch_size, 1, seq_len, seq_len)
        
        
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(
                inps[j].unsqueeze(0),
                attention_mask=extended_attention_mask.to(torch.bool),
                position_ids=position_ids,
                position_embeddings=position_embeddings
            )[0]
                
        for h in handles:
            h.remove()

        for name in subset:
            print(f"pruning layer {i} name {name}")
            
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
            if prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)

                if args.use_variant:
                    # wanda variant
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0., 0.8]
                    W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    while (torch.abs(cur_sparsity - args.sparsity_ratio)>0.001) and (alpha_hist[1]-alpha_hist[0]>=0.001):
                        if cur_sparsity > args.sparsity_ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha

                        alpha = alpha_new 
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                else:
                    # unstructured pruning
                    indices = sort_res[1][:,:int(W_metric.shape[1]*args.sparsity_ratio)]
                    W_mask.scatter_(1, indices, True)

            subset[name].weight.data[W_mask] = 0  ## set weights to zero 

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=extended_attention_mask.to(torch.bool), position_ids=position_ids, position_embeddings=position_embeddings)[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()


@torch.no_grad()
def prune_sparsegpt(args, model, tokenizer, dev, prune_n=0, prune_m=0):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    print('Starting ...')
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    if "model.embed_tokens" in model.hf_device_map:
        dev = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs.get('attention_mask') or torch.ones(inp.shape[:2], device=inp.device, dtype=torch.bool)
            cache['position_ids'] = kwargs.get('position_ids') or torch.arange(inp.shape[1], device=inp.device).unsqueeze(0).expand(inp.shape[0], inp.shape[1])
            cache['position_embeddings'] = kwargs.get('position_embeddings')
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
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')

    for i in range(len(layers)):
        layer = layers[i]
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            print(f"layer {i} device {dev}")
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

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

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')

            gpts[name].fasterprune(args.sparsity_ratio, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
            gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()



@torch.no_grad()
def prune_ablate(args, model, tokenizer, dev, prune_n=0, prune_m=0):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    print('Starting ...')
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    if "model.embed_tokens" in model.hf_device_map:
        dev = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
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
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')

    for i in range(len(layers)):
        layer = layers[i]
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            print(f"layer {i} device {dev}")
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = AblateGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')

            if args.prune_method == "ablate_wanda_seq":
                prune_mask = gpts[name].get_wanda_mask(args.sparsity_ratio, prune_n, prune_m)
            elif args.prune_method == "ablate_mag_seq":
                prune_mask = gpts[name].get_mag_mask(args.sparsity_ratio, prune_n, prune_m)
            elif "iter" in args.prune_method:
                prune_mask = None 

            gpts[name].fasterprune(args, args.sparsity_ratio, mask=prune_mask, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
            gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()