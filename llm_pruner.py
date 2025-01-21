import os
import gc
import sys
import time
import json
import copy
import random
import argparse
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
from config import Config

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main(args):
    set_random_seed(args.seed)

    # Logger 설정
    logger = LoggerWithDepth(
        env_name="{}".format(args.save_ckpt_log_name),
        config=args.__dict__,
        root_dir='prune_log',
        setup_sublogger=True
    )

    # SALMONN 모델 로드
    logger.log("Loading SALMONN model...")
    salmonn_model = load_model(args.config.salmonn_preprocessor)
    llama_model = salmonn_model.llama_model  # LLM 파트만 대상으로 설정
    tokenizer = salmonn_model.llama_tokenizer

    llama_model.to(args.device)
    logger.log("Model loaded successfully.")

    # Pruning 전에 성능 평가 (선택 사항)
    if args.test_before_train:
        logger.log("\n================== Evaluation Before Pruning ==================\n")
        salmonn_model.eval()
        with torch.no_grad():
            sample_inputs = get_salmonn_examples(args.config, args.num_examples, seq_len=args.max_seq_len)
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
    pruner_type = args.pruner_type.lower()
    assert pruner_type in ['random', 'l1', 'l2', 'taylor']

    if pruner_type == 'random':
        importance = llama_pruner.RandomImportance()
    elif pruner_type == 'l1':
        importance = llama_pruner.MagnitudeImportance(p=1)
    elif pruner_type == 'l2':
        importance = llama_pruner.MagnitudeImportance(p=2)
    elif pruner_type == 'taylor':
        importance = llama_pruner.TaylorImportance(
            group_reduction=args.grouping_strategy,
            taylor=args.taylor
        )
    else:
        raise NotImplementedError

    # Pruner 초기화
    forward_prompts = get_salmonn_examples(args.config, 1, seq_len=64)
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
        global_pruning=args.global_pruning,
        ch_sparsity=args.pruning_ratio,
        iterative_steps=args.iterative_steps,
        ignored_layers=[]
    )

    # Pruning 실행
    logger.log("Starting pruning...")
    llama_model.train()
    for step in range(args.iterative_steps):
        sample_inputs = get_salmonn_examples(args.config, args.num_examples, seq_len=args.max_seq_len)
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
            "Step {}/{}: Loss = {:.4f}".format(step + 1, args.iterative_steps, loss.item())
        )

    # Pruning 후 모델 저장 및 평가
    if args.save_model:
        logger.log("Saving pruned model...")
        torch.save(
            {'model': llama_model.state_dict(), 'tokenizer': tokenizer},
            os.path.join('prune_log', '{}_pruned.pt'.format(args.save_ckpt_log_name))
        )

    if args.test_after_train:
        logger.log("\n================== Evaluation After Pruning ==================\n")
        llama_model.eval()
        with torch.no_grad():
            sample_inputs = get_salmonn_examples(args.config, args.num_examples, seq_len=args.max_seq_len)
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

    parser.add_argument('--config', type=str, default='llm_pruner_config.yaml', help='Path to the config file.')
    parser.add_argument('--pruner_type', type=str, default='l2', help='Pruner type [random, l1, l2, taylor].')
    parser.add_argument('--pruning_ratio', type=float, default=0.5, help='Pruning ratio.')
    parser.add_argument('--iterative_steps', type=int, default=1, help='Number of pruning iterations.')
    parser.add_argument('--grouping_strategy', type=str, default='sum', help='Grouping strategy for Taylor pruning.')
    parser.add_argument('--test_before_train', action='store_true', help='Evaluate model before pruning.')
    parser.add_argument('--test_after_train', action='store_true', help='Evaluate model after pruning.')
    parser.add_argument('--save_model', action='store_true', help='Save the pruned model.')
    parser.add_argument('--global_pruning', action='store_true', help='Enable global pruning.')
    parser.add_argument('--num_examples', type=int, default=10, help='Number of examples for pruning.')
    parser.add_argument('--max_seq_len', type=int, default=128, help='Maximum sequence length.')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for pruning.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--save_ckpt_log_name', type=str, default='salmonn_prune', help='Name for saving logs and checkpoints.')

    args = parser.parse_args()
    main(args)
