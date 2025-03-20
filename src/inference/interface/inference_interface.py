import torch
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration
import os

class Gemma3Inference:
    def __init__(self, hf_token: str, model_cache_dir: str = "reference/models"):
        # 모델 저장 경로 설정
        self.model_cache_dir = model_cache_dir
        os.makedirs(self.model_cache_dir, exist_ok=True)
        
        # Hugging Face 토큰 설정
        self.hf_token = hf_token
        if not self.hf_token:
            raise ValueError("HUGGINGFACE_TOKEN 환경 변수를 설정해주세요.")

        # 모델 이름과 토크나이저 및 모델 로드
        self.model_name = "google/gemma-3-4b-it"
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, 
            token=self.hf_token,
            cache_dir=self.model_cache_dir
        )
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            self.model_name, 
            token=self.hf_token, 
            torch_dtype=torch.bfloat16,
            cache_dir=self.model_cache_dir
        )

        # GPU 사용 가능 시 모델을 GPU로 이동
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
            print("GPU를 사용하여 추론을 수행합니다.")
        else:
            print("GPU를 사용할 수 없어 CPU로 추론을 수행합니다.")

    def generate_response(self, question: str) -> str:
        # 입력 텍스트 토큰화 및 텐서 변환
        inputs = self.tokenizer(question, return_tensors="pt")

        # GPU 사용 시 입력을 GPU로 이동
        if torch.cuda.is_available():
            inputs = {key: value.to("cuda") for key, value in inputs.items()}

        # 모델을 사용하여 텍스트 생성
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=100)

        # 생성된 텍스트 디코딩 및 출력
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer