from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 저장된 모델 경로
pruned_model_path = "out/deepseek/unstructured/wanda/"

# 프루닝된 모델 로드
model = AutoModelForCausalLM.from_pretrained(pruned_model_path, torch_dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(pruned_model_path)

# 테스트 문장 생성
input_text = "DeepSeek is an advanced AI model."
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

# 프루닝 후 모델 추론
with torch.no_grad():
    output = model.generate(**inputs, max_length=50)

# 출력 확인
print("Pruned Model Output:", tokenizer.decode(output[0], skip_special_tokens=True))