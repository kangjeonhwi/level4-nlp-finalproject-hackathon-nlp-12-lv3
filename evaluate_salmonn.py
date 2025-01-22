import argparse
import json
import random
import sys
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import os
import time
# Add custom module path
# sys.path.append(str(Path(__file__).parent / "audiolm-trainer"))

# Custom modules
from salmonn_utils import SALMONNTestDataset, load_preprocessor, load_model
from config import Config
from utils import get_dataloader, prepare_sample
from metrics import compute_wer, compute_spider


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg-path", 
        type=str, 
        help='path to configuration file', 
        default='salmonn_eval_config.yaml'
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    # --- Deprecated options ---
    parser.add_argument("--task", type=str, default=None, 
                    help="(Deprecate) Task to evaluate. Use --mode instead. This option will be removed in a future version.", 
                    choices=['asr', 'aac'])

    parser.add_argument("--skip_scoring", action='store_false', default=True, 
                    help="(Deprecate) If True, skip scoring after inference. Use --mode instead. This option will be removed in a future version.")
    # --- Deprecated options end ---
    # --- New options ---
    parser.add_argument("--mode", type=str, default="valid_aac", 
                    help="Mode to evaluate. Supports submission and validation modes for ASR and AAC tasks.", 
                    choices=['submission_asr', 'submission_aac', 'valid_asr', 'valid_aac'])
    # --- New options end ---

    args = parser.parse_args()

    if args.mode is None:
        # --- For Previous Version ---
        if args.task is None:
            raise ValueError("Either --task or --mode must be provided")
        args.mode = convert_task_to_mode(args.task, args.skip_scoring)

    # --- Override Previous Version Args ---
    args.task = args.mode.split("_")[1]
    args.make_submission = args.mode.split("_")[0] == "submission"

    return args

def convert_task_to_mode(task, skip_scoring):
    if skip_scoring:
        if task == 'asr':
            return 'submission_asr'
        elif task == 'aac':
            return 'submission_aac'
    else:
        if task == 'asr':       
            return 'valid_asr'
        elif task == 'aac':
            return 'valid_aac'
    
    raise ValueError(f"Invalid task: {task} | {skip_scoring}")

def get_dataset(dataset_cfg, run_cfg, task):
    testset = SALMONNTestDataset(
        dataset_cfg.prefix, dataset_cfg.test_ann_path, dataset_cfg.whisper_path, task
    )

    test_loader = get_dataloader(testset, run_cfg, is_train=False, use_distributed=False)
    return test_loader

def replace_test_ann_path(cfg):
    if "test_ann_path" not in cfg.config.datasets.keys():
        if args.task == "asr":
            cfg.config.datasets.test_ann_path = cfg.config.datasets.test_ann_path_asr
        elif args.task == "aac":
            cfg.config.datasets.test_ann_path = cfg.config.datasets.test_ann_path_aac
    return cfg

def main(args):
    cfg = Config(args)
    cfg = replace_test_ann_path(cfg)
    # Load models
    salmonn_preprocessor = load_preprocessor(cfg)
    llama_model, tokenizer = load_model(salmonn_preprocessor)
    salmonn_preprocessor.llama_model = llama_model

    # Load data
    dataloader = get_dataset(cfg.config.datasets, cfg.config.run, args.task)

    # Evaluation
    testset_ids, hyps, refs = [], [], []
    start_time = time.time()
    for samples in tqdm(dataloader):
        testset_id = samples["testset_id"]
        testset_ids.extend(testset_id)

        # Preprocess
        samples = prepare_sample(samples, cuda_enabled=torch.cuda.is_available(), device=cfg.config.run.device)
        generate_cfg = cfg.config.generate
        results = salmonn_preprocessor.generate(samples, generate_cfg=generate_cfg)
        hyp = [result.split(generate_cfg.end_sym)[0].lower() for result in results]
        hyps.extend(hyp)

        if not args.make_submission:
            ref = samples["text"]
            refs.extend(ref)

    run_time = time.time() - start_time
    now_str = time.strftime('%Y-%m-%d %H:%M:%S')

    if args.make_submission:
        os.makedirs("submission_results", exist_ok=True)
        file_name = f"submission_results/{now_str}_{args.mode}.csv"
        with open(f"submission_results/{now_str}_{args.mode}.json", "w") as f:
            json.dump({"run_time": run_time}, f)
    else:
        if args.task == 'asr':
            compute_wer(hyps, refs)
            
        elif args.task == 'aac':
            compute_spider(hyps, refs)
        os.makedirs("valid_results", exist_ok=True)
        file_name = f"valid_results/{now_str}_{args.mode}.csv"
        with open(f"valid_results/{now_str}_{args.mode}.json", "w") as f:
            json.dump({"run_time": run_time}, f)

    result_df = pd.DataFrame({"testset_id": testset_ids, "text": hyps})
    result_df.to_csv(file_name, index=False)

if __name__ == '__main__':
    args = parse_args()

    random.seed(42)
    main(args)
