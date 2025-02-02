# Copyright (2024) Tsinghua University, Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import argparse
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import wandb
import gc

from utils import *
from config import Config
from dist_utils import get_rank, init_distributed_mode
from models import load_model
from dataset import SALMONNDataset
from runner import Runner


def parse_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument("--cfg-path", type=str, required=True, help='path to configuration file')
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    parser.add_argument("--dryrun", action='store_true', help='if True, use dummy model and skip forward/backward')
    parser.add_argument(
        "--stage", 
        type=str, 
        choices=["stage1", "stage2", "Merged"], 
        required=True, 
        help="Choose training stage: stage1, stage2, or Merged"
    )

    return parser.parse_args()


def setup_seeds(config):
    seed = config.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def main():
    # set before init_distributed_mode() to ensure the same job_id shared across all ranks.
    job_id = now()

    # load config
    args = parse_args()
    cfg = Config(args)
    run_config = cfg.config.run
    model_config = cfg.config.model
    data_config = cfg.config.datasets

    # initialize distributed training
    init_distributed_mode(run_config)
    setup_seeds(run_config)
    setup_logger() # set after init_distributed_mode() to only log on master.'''

    # Wandb logger        
    global_rank = int(os.environ.get("RANK", 0))
    if global_rank == 0:
        wandb.login()
        wandb.init(project="audio_lm", name=run_config.exp_name)

    # print config
    cfg.pretty_print()

    # build datasets
    datasets = {
        "train": SALMONNDataset(data_config.prefix, data_config.train_ann_path, data_config.whisper_path),
        "valid": SALMONNDataset(data_config.prefix, data_config.valid_ann_path, data_config.whisper_path),
        "test": SALMONNDataset(data_config.prefix, data_config.test_ann_path, data_config.whisper_path),
    }

    # build model

    if not args.dryrun:
        model = load_model(model_config)
    else: # load small dummy language model
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained("apple/OpenELM-270M-Instruct", trust_remote_code=True)

    # build runner
    output_dir = cfg.config.run.output_dir
    if args.stage == "stage1":
        cfg.config.run.output_dir = output_dir + "/stage1"
        runner = Runner(cfg, model, datasets, job_id, args.dryrun)
        logging.info("Starting Stage 1 Training...")
        runner.train()

    elif args.stage == "stage2":
        cfg.config.run.output_dir = output_dir + "/stage2"
        runner = Runner(cfg, model, datasets, job_id, args.dryrun)
        logging.info("Starting Stage 2 Training...")
        runner.train()

    elif args.stage == "Merged":
        logging.info("Starting Merged Training (Stage 1 + Stage 2)...")

        # Stage 1 Training
        cfg.config.run.output_dir = output_dir + "/stage1"
        runner = Runner(cfg, model, datasets, job_id, args.dryrun)
        logging.info("Training Stage 1...")
        runner.train()
        ckpt_path = runner.best_path

        # Clear memory after Stage 1
        logging.info("Clearing memory after Stage 1...")
        del runner  # Runner 객체 삭제
        del model   # 모델 삭제
        torch.cuda.empty_cache()  # GPU 메모리 해제
        gc.collect()  # CPU 메모리 정리

        # Load Stage 1 Model for Stage 2
        logging.info("Loading Stage 1 Model for Stage 2...")
        
        # Reload model and runner for Stage 2
        model = load_model(model_config)  # 모델 재생성
        cfg.config.model.ckpt = ckpt_path 
        cfg.config.run.output_dir = output_dir + "/stage2"
        runner = Runner(cfg, model, datasets, job_id, args.dryrun)

        # Stage 2 Training
        logging.info("Training Stage 2...")
        runner.train()

if __name__ == "__main__":
    main()