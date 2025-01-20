import random
import numpy as np
import torch

from datasets import load_dataset
from torch.utils.data.dataset import Dataset
from salmonn_utils import load_model
from salmonn_utils import SALMONNTestDataset
from torch.utils.data import DataLoader

def get_salmonn_examples(datasets_config, llama_tokenizer, n_samples, seq_len):
    """
    SALMONN 모델의 LLM 파트를 위한 샘플 데이터를 생성합니다.

    Args:
        datasets_config: 데이터셋 설정 딕셔너리.
        llama_tokenizer: LLaMA 모델 토크나이저.
        n_samples: 생성할 샘플 수.
        seq_len: 샘플의 최대 시퀀스 길이.

    Returns:
        Dict: SALMONN 모델 입력 형식으로 구성된 데이터.
    """
    # 데이터 로더 설정
    dataset = SALMONNTestDataset(
        prefix=datasets_config['prefix'],
        ann_path=datasets_config['test_ann_path_asr'],  # ASR 태스크 기준 경로 사용
        whisper_path=datasets_config['whisper_path'],
    )
    dataloader = DataLoader(dataset, batch_size=n_samples, shuffle=True)

    # 샘플 데이터 가져오기
    batch = next(iter(dataloader))

    spectrogram = batch["spectrogram"].to(datasets_config['device'])
    raw_audio = batch["raw_wav"].to(datasets_config['device'])
    text = [t for t in batch.get("text", [])]

    # 텍스트 입력을 LLaMA 토크나이저로 처리
    tokenized_prompts = llama_tokenizer(
        text,
        return_tensors="pt",
        padding="longest",
        truncation=True,
        max_length=seq_len
    )

    return {
        "spectrogram": spectrogram,
        "raw_wav": raw_audio,
        "text": tokenized_prompts["input_ids"].to(datasets_config['device']),
        "attention_mask": tokenized_prompts["attention_mask"].to(datasets_config['device'])
    }

def get_c4(tokenizer, n_samples, seq_len):
    traindata = load_dataset(
        'allenai/c4', 'allenai--c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
    )
    
    tokenized_samples, history = [], []
    for _ in range(n_samples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            tokenized_sample = tokenizer(traindata[i]['text'], return_tensors='pt')
            if tokenized_sample.input_ids.shape[1] >= seq_len and i not in history:
                history.append(i)
                break
        i = random.randint(0, tokenized_sample.input_ids.shape[1] - seq_len )
        tokenized_samples.append(tokenized_sample.input_ids[:, i:i+seq_len])
    return torch.cat(tokenized_samples, dim=0)

def get_bookcorpus(tokenizer, n_samples, seq_len):
    traindata = load_dataset(
        'bookcorpus', split='train'
    )
    
    tokenized_samples, history = [], []
    for _ in range(n_samples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            tokenized_sample = tokenizer(traindata[i]['text'], return_tensors='pt')
            if tokenized_sample.input_ids.shape[1] >= seq_len and i not in history:
                history.append(i)
                break
        i = random.randint(0, tokenized_sample.input_ids.shape[1] - seq_len)
        tokenized_samples.append(tokenized_sample.input_ids[:, i:i+seq_len])
    return torch.cat(tokenized_samples, dim=0 )

def get_examples(dataset, tokenizer, n_samples, seq_len = 128):
    if dataset == 'c4':
        return get_c4(tokenizer, n_samples, seq_len)
    elif dataset == 'bookcorpus':
        return get_bookcorpus(tokenizer, n_samples, seq_len)
    else:
        raise NotImplementedError
