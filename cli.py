import click
import json
import os
from pathlib import Path
import shutil
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

# 기본 경로 설정
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))  # /mnt/level4-nlp-finalproject-hackathon-nlp-12-lv3
CACHE_PATH = BASE_DIR.parent / "models" / "cache.json"  # ../models/cache.json
MODELS_DIR = BASE_DIR.parent / "models" / "models" # ../models
DATASETS_DIR = BASE_DIR.parent / "datasets"  # ../datasets

def load_cache():
    """캐시 파일 로드"""
    try:
        with open(CACHE_PATH, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        click.echo(f"캐시 파일을 찾을 수 없습니다: {CACHE_PATH}")
        return {}

def save_cache(cache_data):
    """캐시 파일 저장"""
    with open(CACHE_PATH, 'w') as f:
        json.dump(cache_data, f, indent=4)

def get_models_by_type(cache, model_type):
    """타입별 모델 필터링"""
    return {k: v for k, v in cache.items() if v['type'] == model_type}

def display_models_by_type(models, model_type):
    """타입별 모델 목록 표시"""
    click.echo(f"\n{model_type} 모델 목록:")
    for idx, name in enumerate(models.keys(), 1):
        click.echo(f"{idx}. {name}")
    if model_type == 'llm':
        click.echo("0. 새로운 Huggingface 모델 추가")

def reorganize_model_directory(org_name, model_id):
    """모델 디렉토리 구조 재구성"""
    src_dir = MODELS_DIR / f"models--{org_name}--{model_id}"
    org_dir = MODELS_DIR / org_name
    org_dir.mkdir(exist_ok=True)
    
    dst_dir = org_dir / model_id
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    shutil.move(str(src_dir), str(dst_dir))
    
    # snapshots 폴더 내 해시 디렉토리 찾기
    snapshot_dir = dst_dir / "snapshots"
    hash_dir = next(snapshot_dir.iterdir())
    return hash_dir

def select_model(cache, model_type):
    """모델 선택 프로세스"""
    filtered_models = get_models_by_type(cache, model_type)
    display_models_by_type(filtered_models, model_type)
    
    while True:
        choice = click.prompt(f"{model_type} 모델 번호를 선택하세요", type=int)
        
        if choice == 0 and model_type == 'llm':
            model_name = click.prompt("Huggingface 모델 경로를 입력하세요", type=str)
            try:
                # 모델과 토크나이저 다운로드
                model = AutoModelForCausalLM.from_pretrained(
                    model_name, 
                    cache_dir=str(MODELS_DIR),
                    trust_remote_code=True
                )
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name, 
                    cache_dir=str(MODELS_DIR),
                    legacy=False, 
                    use_fast=True,
                    trust_remote_code=True
                )
                assert isinstance(tokenizer, PreTrainedTokenizerBase), "Tokenizer 초기화 실패"
                click.echo("Model download complete.")
                # 디렉토리 재구성 및 경로 설정
                org_name, model_id = model_name.split('/')
                final_path = reorganize_model_directory(org_name, model_id)
                
                # LoRA 확인
                has_lora = any('lora' in name.lower() for name, _ in model.named_modules())
                
                # 캐시 업데이트
                model_key = model_name.replace('/', '-')
                cache[model_key] = {
                    "path": str(final_path.relative_to(MODELS_DIR)),
                    "LoRA": has_lora,
                    "type": "llm",
                    "ckpts": [],
                    "eos_token": tokenizer.eos_token
                }
                
                save_cache(cache)
                return model_key
                
            except Exception as e:
                click.echo(f"모델 다운로드 중 오류 발생: {str(e)}")
                continue
                
        elif 1 <= choice <= len(filtered_models):
            return list(filtered_models.keys())[choice-1]
        
        click.echo("잘못된 선택입니다. 다시 선택해주세요.")

@click.group()
def cli():
    """Audio Model Training CLI"""
    pass

@cli.command()
def train():
    """학습 모드"""
    # 프로젝트 이름 입력
    project_name = click.prompt("프로젝트 이름을 입력하세요", type=str)
    
    # 캐시 로드
    cache = load_cache()
    
    # 각 타입별로 순차적으로 모델 선택
    selected_models = {}
    for model_type in ["encoder_aac", "encoder_asr", "llm"]:
        click.echo(f"\n{model_type.upper()} 모델 선택")
        selected_model = select_model(cache, model_type)
        selected_models[model_type] = selected_model
    
    # 선택 결과 출력
    click.echo("\n선택된 설정:")
    click.echo(f"프로젝트 이름: {project_name}")
    for model_type, model_name in selected_models.items():
        click.echo(f"{model_type}: {model_name}")
        if model_name in cache:
            click.echo(f"  경로: {cache[model_name]['path']}")
            click.echo(f"  LoRA: {'사용' if cache[model_name]['LoRA'] else '미사용'}")

if __name__ == '__main__':
    cli()