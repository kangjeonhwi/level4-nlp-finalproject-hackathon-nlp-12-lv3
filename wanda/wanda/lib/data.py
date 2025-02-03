# Code adapted from https://github.com/IST-DASLab/sparsegpt/blob/master/datautils.py

import numpy as np
import random
import torch
from datasets import load_dataset
from itertools import islice
from tqdm import tqdm 

# Set seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

# Wrapper for tokenized input IDs
class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids

# Load and process wikitext2 dataset
def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    # Load train and test datasets
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    # Encode datasets
    max_train_length = 16384
    train_text = " ".join(traindata['text'])
    trainenc = tokenizer(train_text, return_tensors='pt', truncation=True, max_length=max_train_length)
    
    eval_chunk_length = 16384
    test_chunks = []
    for text in testdata['text']:
        encoding = tokenizer(
            text, 
            return_tensors='pt', 
            truncation=True, 
            max_length=eval_chunk_length  # 개별 텍스트마다 최대 길이를 제한
        )
        input_ids = encoding.input_ids[0]  # 1D tensor
        # 만약 토큰 수가 eval_chunk_length보다 큰 경우, 청크로 분할
        if input_ids.size(0) > eval_chunk_length:
            for i in range(0, input_ids.size(0) - eval_chunk_length + 1, eval_chunk_length):
                chunk = input_ids[i:i+eval_chunk_length]
                test_chunks.append(chunk.unsqueeze(0))
        else:
            test_chunks.append(input_ids.unsqueeze(0))
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt', max_length=16384)

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

# Load and process c4 dataset
def get_c4(nsamples, seed, seqlen, tokenizer):
    # Load train and validation datasets
    nsamples = 100
    traindata_stream = load_dataset('allenai/c4', 'en', data_files={
        'train': 'en/c4-train.00000-of-01024.json.gz', 
        'validation': 'en/c4-train.00000-of-01024.json.gz'
    }, 
    split='train',
    streaming=True,
    verification_mode="no_checks"
    )
    traindata = list(islice(traindata_stream, nsamples))
    
    valdata_stream = load_dataset('allenai/c4', 'en', 
                           data_files={
                               'validation': 'en/c4-validation.00000-of-00008.json.gz'
                            }, 
                           split='validation',
                           streaming=True,
                           verification_mode="no_checks")
    valdata = list(islice(valdata_stream, nsamples))

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in tqdm(range(nsamples), desc="샘플을 수집: "):
        while True:
            seqlen = 2048
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    # Prepare validation dataset
    val_texts = [example['text'] for example in valdata[:1100]]
    valenc = tokenizer(' '.join(val_texts), truncation=True, max_length=1024, return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]
    valenc = TokenizerWrapper(valenc)
    return trainloader, valenc

# Function to select the appropriate loader based on dataset name
def get_loaders(name, nsamples=128, seed=0, seqlen=2048, tokenizer=None):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, tokenizer)
    if "c4" in name:
        return get_c4(nsamples, seed, seqlen, tokenizer)