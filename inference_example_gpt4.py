import torch
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration
import os

def main():
    # 모델 저장 경로 설정
    model_cache_dir = "reference/models"
    os.makedirs(model_cache_dir, exist_ok=True)
    
    # Hugging Face 토큰 설정
    hf_token = ''
    if not hf_token:
        raise ValueError("HUGGINGFACE_TOKEN 환경 변수를 설정해주세요.")

    # 모델 이름과 토크나이저 및 모델 로드
    model_name = "google/gemma-3-4b-it"
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        token=hf_token,
        cache_dir=model_cache_dir
    )
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_name, 
        token=hf_token, 
        torch_dtype=torch.bfloat16,
        cache_dir=model_cache_dir
    )

    # GPU 사용 가능 시 모델을 GPU로 이동
    if torch.cuda.is_available():
        model = model.to("cuda")
        print("GPU를 사용하여 추론을 수행합니다.")
    else:
        print("GPU를 사용할 수 없어 CPU로 추론을 수행합니다.")

    # 질문 정의
    question = "Gemma 3 모델의 장점은 무엇입니까?"

    # 입력 텍스트 토큰화 및 텐서 변환
    inputs = tokenizer(question, return_tensors="pt")

    # GPU 사용 시 입력을 GPU로 이동
    if torch.cuda.is_available():
        inputs = {key: value.to("cuda") for key, value in inputs.items()}

    # 모델을 사용하여 텍스트 생성
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100)

    # 생성된 텍스트 디코딩 및 출력
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"질문: {question}")
    print(f"답변: {answer}")

if __name__ == "__main__":
    main()