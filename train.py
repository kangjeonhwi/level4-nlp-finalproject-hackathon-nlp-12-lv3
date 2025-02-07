import os
import argparse
import torch
import wandb
import logging

from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import DatasetDict, Dataset
from config import Config
from models.salmonn import SALMONN
from dataset import SALMONNDataset

def parse_args():
    parser = argparse.ArgumentParser(description="train parameters")
    parser.add_argument("--cfg-path", type=str, required=True, help="path to configuration file")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override settings in the config, in key=value format.",
    )
    parser.add_argument("--dryrun", action="store_true", help="if True, use dummy model and skip forward/backward")
    return parser.parse_args()

def load_datasets(config):
    """ 기존 SALMONNDataset을 HF Trainer에서 사용할 수 있도록 변환 """
    datasets = {
        "train": SALMONNDataset(config.prefix, config.train_ann_path, config.whisper_path),
        "valid": SALMONNDataset(config.prefix, config.valid_ann_path, config.whisper_path),
        "test": SALMONNDataset(config.prefix, config.test_ann_path, config.whisper_path),
    }
    
    def preprocess(sample):
        """ 텍스트 토큰화 및 전처리 """
        return {
            "spectrogram": sample["spectrogram"].numpy(),
            "raw_wav": sample["raw_wav"],
            "text": sample["text"],
            "task": sample["task"],
            "Q": sample["Q"],
            "id": sample["id"],
        }
    
    datasets = DatasetDict({k: Dataset.from_list([preprocess(d) for d in v]) for k, v in datasets.items()})
    datasets.set_format("torch")
    return datasets

def main():
    args = parse_args()
    
    # Config 로드
    cfg = Config(args)
    run_config = cfg.config.run
    model_config = cfg.config.model
    data_config = cfg.config.datasets

    # WandB 설정
    wandb.init(project="hf_sft_trainer_salmonn", name=run_config.exp_name)

    # 데이터셋 로딩
    datasets = load_datasets(data_config)

    # 모델 로드
    model = SALMONN.from_config(model_config)

    # Hugging Face TrainingArguments 설정
    training_args = TrainingArguments(
        output_dir=run_config.output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=run_config.batch_size,
        per_device_eval_batch_size=run_config.batch_size,
        learning_rate=run_config.optims.init_lr,
        num_train_epochs=run_config.optims.max_epoch,
        logging_dir=run_config.logging_dir,
        report_to="wandb",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    # SFTTrainer 설정
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["valid"],
        dataset_text_field="text",  # Causal LM 훈련을 위한 텍스트 필드 지정
    )

    # 학습 실행
    trainer.train()
    trainer.save_model(run_config.output_dir)

    # 평가 실행
    eval_results = trainer.evaluate()
    print(f"Validation Results: {eval_results}")

    wandb.finish()

if __name__ == "__main__":
    main()
