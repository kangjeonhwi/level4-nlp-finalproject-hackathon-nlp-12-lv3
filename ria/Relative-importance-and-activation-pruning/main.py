import argparse
import os 
import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, GPT2Tokenizer
from importlib.metadata import version
from lib.eval import eval_ppl, eval_zero_shot
from airllm import AutoModel

# 환경 정보 출력
# CUDA 사용 불가능한 경우 False로 출력
print('cuda available:', torch.cuda.is_available())
print('mps available:', torch.backends.mps.is_available())  # MPS 사용 가능 여부 출력
print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))


def get_llm(model_name, cache_dir="llm_weights", seqlen=2048, pth_path=None):
    # Config 로드 및 rope_scaling 수정
    config = AutoConfig.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        use_auth_token=True,
        rope_scaling={"type": "linear", "factor": 32.0}
    )
    
    print(f"Modified rope_scaling: {config.rope_scaling}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        torch_dtype=torch.float16,
        cache_dir=cache_dir,
        low_cpu_mem_usage=True,
        device_map="auto" if torch.backends.mps.is_available() else None,  # MPS 환경 적용
        use_auth_token=True
    )
    
    # model = AutoModel.from_pretrained(model_name)
    
    
    # .pth 파일에서 가중치 로드
    if pth_path:
        print(f"Loading weights from .pth file: {pth_path}")
        state_dict = torch.load(pth_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        print("Weights loaded successfully!")

    model.seqlen = seqlen
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLaMA model')
    parser.add_argument('--pth_path', type=str, default=None,
                        help='Path to the .pth file containing model weights.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed for sampling the calibration data.')
    parser.add_argument('--calib_dataset', type=str, default="wikitext2", help='Calibration dataset')
    parser.add_argument('--nsamples', type=int, default=88,
                        help='Number of calibration samples.')
    parser.add_argument('--seqlen', type=int, default=2048,
                        help='Sequence length')
    parser.add_argument('--sparsity_ratio', type=float,
                        default=0.5, help='Sparsity level')
    parser.add_argument("--sparsity_type", type=str, default="unstructured",
                        help="Sparsity type, choose from unstructured, 4:8, 1:4, 2:4, 3:4.")
    parser.add_argument("--prune_method", type=str, choices=[
                        "svd_finetuned", "magnitude", "ri", "wanda", "svd_ri", "svd", "sparsegpt", "ria"])
    parser.add_argument("--cache_dir", default="llm_weights", type=str)
    parser.add_argument('--save', action="store_true")
    parser.add_argument('--save_model', type=str, default='./save_model',
                        help='Path to save the pruned model.')
    parser.add_argument('--semi_sparse_acc', action="store_true",
                        help="using pytorch semi sparse acceleration. Only when sparsity type is 2:4")
    parser.add_argument("--eval_zero_shot",
                        action="store_true", help="zero-shot performance")
    parser.add_argument("--a", type=float, default=0.5,
                        help="exponent of activation")
    parser.add_argument("--reconstruction", action="store_true",
                        help="remaining weight reconstruction based on sparsegpt")
    parser.add_argument("--reallocation", action="store_true",
                        help="Heuristic Channel Reallocation")
    parser.add_argument("--lsa", action="store_true",
                        help="Linear Sum Assignment")
    parser.add_argument("--importance_score", type=str,
                        default="sum", help="assign importance score for columns")
    parser.add_argument("--gptq", action="store_true", help="use gptq or not")
    parser.add_argument("--per_outneuron", action="store_true",
                        help="pruning per outneuron. Wanda's tactic.")
    parser.add_argument("--test_bs", type=int, default=1,
                        help="test batch size")
    parser.add_argument("--use_cusparselt", action="store_true")
    parser.add_argument("--layer_wise", action="store_true")
    parser.add_argument("--svd_threshold", type=float, default=1e-3)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--pt_files_dir", type=str, required=False, help="Directory containing .pt files for pruning.")
    args = parser.parse_args()

    # Reproducibility 설정
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    # MPS를 사용하거나 CPU를 사용할 수 있도록 설정
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    model_name = args.model.split("/")[-1]
    print(f"loading llm model {args.model}")
    model = get_llm(args.model, args.cache_dir, args.seqlen, pth_path='../pth_path/salmonn_3b_nota.pth')
    model.eval()
    if "opt" in args.model:
        tokenizer = GPT2Tokenizer.from_pretrained(args.model, use_fast=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    print(model)

    # Pruning 작업 수행
    if args.sparsity_ratio != 0:
        print("pruning starts")
        from lib.prune import prune_magnitude, prune_sparsegpt, prune_ria, check_sparsity
        if args.prune_method in ["wanda", "ria"]:
            prune_ria(args, model, tokenizer, device)
        elif args.prune_method == "magnitude":
            prune_magnitude(args, model, tokenizer, device)
        elif args.prune_method == "sparsegpt":
            prune_sparsegpt(args, model, tokenizer, device)
            
    # 모델 저장
    if args.save_model:
        model.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)

        # Sparsity 확인
        sparsity_ratio = check_sparsity(args, model)
        print(f"Sparsity sanity check: {sparsity_ratio:.4f}")

    # Perplexity 평가
    # ppl_test = eval_ppl(model, tokenizer, args.eval_dataset,
    #                     args.test_bs, device)
    # print(f"perplexity: {ppl_test}")


if __name__ == '__main__':
    main()
