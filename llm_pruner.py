import os
import gc
import sys
import time
import json
import copy
import random
import argparse
import yaml
from typing import Tuple

import torch
import numpy as np
from transformers import AutoTokenizer, LlamaForCausalLM
from salmonn_utils import load_model
from LLMPruner.datasets.example_samples import get_salmonn_examples
from LLMPruner.torch_pruning import MetaPruner
from LLMPruner.pruner import hf_llama_pruner as llama_pruner
from LLMPruner.utils.logger import LoggerWithDepth
from LLMPruner.evaluator.ppl import PPLMetric

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main(args):
    set_random_seed(args.seed)

    # Config 파일 로드
    config = load_config(args.config)
    model_config = config['model']
    pruner_config = config['pruner']
    run_config = config['run']
    datasets_config = config['datasets']

    # Logger 설정
    logger = LoggerWithDepth(
        env_name="{}".format(run_config['save_ckpt_log_name']),
        config=args.__dict__,
        root_dir='prune_log',
        setup_sublogger=True
    )

    # SALMONN 모델 로드
    logger.log("Loading SALMONN model...")
    salmonn_model = load_model(model_config)
    llama_model = salmonn_model.llama_model  # LLM 파트만 대상으로 설정
    tokenizer = salmonn_model.llama_tokenizer

    llama_model.to(run_config['device'])
    logger.log("Model loaded successfully.")

    # Pruning 전에 성능 평가 (선택 사항)
    if run_config['test_before_train']:
        logger.log("\n================== Evaluation Before Pruning ==================\n")
        salmonn_model.eval()
        with torch.no_grad():
            sample_inputs = get_salmonn_examples(datasets_config, run_config['num_examples'], seq_len=run_config['max_seq_len'])
            speech_embeds, speech_atts = salmonn_model.encode_speech(
                sample_inputs["spectrogram"], raw_wav=sample_inputs["raw_wav"]
            )
            inputs_embeds, attention_mask = salmonn_model.prompt_wrap(
                speech_embeds, speech_atts, sample_inputs["text"]
            )
            outputs = llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=sample_inputs["text"]
            )
            logger.log("PPL before pruning: {:.4f}".format(outputs.loss.item()))

    # Pruning 설정
    logger.log("Configuring pruning...")
    pruner_type = pruner_config['pruner_type']
    assert pruner_type in ['random', 'l1', 'l2', 'taylor']

    if pruner_type == 'random':
        importance = llama_pruner.RandomImportance()
    elif pruner_type == 'l1':
        importance = llama_pruner.MagnitudeImportance(p=1)
    elif pruner_type == 'l2':
        importance = llama_pruner.MagnitudeImportance(p=2)
    elif pruner_type == 'taylor':
        importance = llama_pruner.TaylorImportance(
            group_reduction=pruner_config['grouping_strategy'],
            taylor=pruner_config['taylor']
        )
    else:
        raise NotImplementedError

    # Pruner 초기화
    forward_prompts = get_salmonn_examples(datasets_config, 1, seq_len=64)
    speech_embeds, speech_atts = salmonn_model.encode_speech(
        forward_prompts["spectrogram"], raw_wav=forward_prompts["raw_wav"]
    )
    inputs_embeds, _ = salmonn_model.prompt_wrap(
        speech_embeds, speech_atts, forward_prompts["text"]
    )

    pruner = MetaPruner(
        llama_model,
        inputs_embeds,
        importance=importance,
        global_pruning=pruner_config['global_pruning'],
        ch_sparsity=pruner_config['pruning_ratio'],
        iterative_steps=pruner_config['iterative_steps'],
        ignored_layers=[]
    )

    # Pruning 실행
    logger.log("Starting pruning...")
    llama_model.train()
    for step in range(pruner_config['iterative_steps']):
        sample_inputs = get_salmonn_examples(datasets_config, run_config['num_examples'], seq_len=run_config['max_seq_len'])
        speech_embeds, speech_atts = salmonn_model.encode_speech(
            sample_inputs["spectrogram"], raw_wav=sample_inputs["raw_wav"]
        )
        inputs_embeds, attention_mask = salmonn_model.prompt_wrap(
            speech_embeds, speech_atts, sample_inputs["text"]
        )
        outputs = llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=sample_inputs["text"]
        )
        loss = outputs.loss
        loss.backward()
        pruner.step()

        logger.log(
            "Step {}/{}: Loss = {:.4f}".format(step + 1, pruner_config['iterative_steps'], loss.item())
        )

    # Pruning 후 모델 저장 및 평가
    if run_config['save_model']:
        logger.log("Saving pruned model...")
        torch.save(
            {'model': llama_model.state_dict(), 'tokenizer': tokenizer},
            os.path.join('prune_log', '{}_pruned.pt'.format(run_config['save_ckpt_log_name']))
        )

    if run_config['test_after_train']:
        logger.log("\n================== Evaluation After Pruning ==================\n")
        llama_model.eval()
        with torch.no_grad():
            sample_inputs = get_salmonn_examples(datasets_config, run_config['num_examples'], seq_len=run_config['max_seq_len'])
            speech_embeds, speech_atts = salmonn_model.encode_speech(
                sample_inputs["spectrogram"], raw_wav=sample_inputs["raw_wav"]
            )
            inputs_embeds, attention_mask = salmonn_model.prompt_wrap(
                speech_embeds, speech_atts, sample_inputs["text"]
            )
            outputs = llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=sample_inputs["text"]
            )
            logger.log("PPL after pruning: {:.4f}".format(outputs.loss.item()))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pruning SALMONN LLM Part')

    parser.add_argument('--config', type=str, required=True, help='Path to the config file.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')

    args = parser.parse_args()
    main(args)
