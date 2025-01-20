import os
import argparse
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from utils import now
from config import Config
from dist_utils import get_rank, init_distributed_mode
from models import load_model
from dataset import SALMONNDataset
from runner import Runner


def parse_args():
    parser = argparse.ArgumentParser(description='Extract embeddings parameters')
    parser.add_argument("--cfg-path", type=str, required=True, help='Path to configuration file')
    parser.add_argument(
        "--options",
        nargs="+",
        help="Override some settings in the config, the key-value pair in xxx=yyy format will be merged into config file",
    )
    return parser.parse_args()


def setup_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

def main():
    args = parse_args()
    cfg = Config(args)
    run_config = cfg.config.run
    model_config = cfg.config.model
    data_config = cfg.config.datasets
    
    init_distributed_mode(run_config)
    setup_seeds(run_config.seed)
    
    cfg.pretty_print()

    datasets = {
        "train": SALMONNDataset(data_config.prefix, data_config.train_ann_path, data_config.whisper_path),
        "valid": SALMONNDataset(data_config.prefix, data_config.valid_ann_path, data_config.whisper_path),
    }

    model = load_model(model_config)
    job_id = now()
    runner = Runner(cfg, model, datasets, job_id=job_id, dryrun=False)
    runner.extract_embeddings_for_dataset(runner.train_loader, "output_dir/train_embeds", "train")
    runner.extract_embeddings_for_dataset(runner.valid_loader, "output_dir/valid_embeds", "valid")


if __name__ == "__main__":
    main()

