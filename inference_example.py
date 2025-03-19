import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

class GemmaInference:
    def __init__(self, model_path=None, cache_dir="models", token=None):
        """
        Gemma 모델을 초기화합니다.
        
        Args:
            model_path (str, optional): 파인튜닝된 모델의 경로. None이면 기본 Gemma 모델을 사용합니다.
            cache_dir (str): 모델과 토크나이저를 저장할 디렉토리
            token (str, optional): Hugging Face 토큰
        """
        self.model_name = "google/gemma-3-4b-it" # google/gemma-3-1b-it
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Hugging Face 토큰 설정
        if token:
            login(token=token)
            print(f"사용 모델: {self.model_name}")
            print(f"토큰으로 로그인 완료: {token[:8]}...")
        
        # 모델과 토크나이저 다운로드 및 저장
        print(f"모델 다운로드 중... (저장 위치: {cache_dir})")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=cache_dir,
            local_files_only=False,  # 로컬에 없으면 다운로드
            token=token  # 토큰 전달
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            cache_dir=cache_dir,
            local_files_only=False,  # 로컬에 없으면 다운로드
            trust_remote_code=True,
            token=token  # 토큰 전달
        )
        
        # GPU 사용 가능한 경우 GPU로 이동
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
            print("GPU 사용 중")
        else:
            print("CPU 사용 중")
        
        # 파인튜닝된 모델이 있다면 로드
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
            print(f"파인튜닝된 모델을 로드했습니다: {model_path}")

    def save_model(self, save_dir=None):
        """
        현재 모델과 토크나이저를 저장합니다.
        
        Args:
            save_dir (str, optional): 저장할 디렉토리. None이면 cache_dir을 사용합니다.
        """
        if save_dir is None:
            save_dir = self.cache_dir
            
        os.makedirs(save_dir, exist_ok=True)
        
        # 토크나이저 저장
        tokenizer_path = os.path.join(save_dir, "tokenizer")
        self.tokenizer.save_pretrained(tokenizer_path)
        print(f"토크나이저가 저장되었습니다: {tokenizer_path}")
        
        # 모델 저장
        model_path = os.path.join(save_dir, "model")
        self.model.save_pretrained(model_path)
        print(f"모델이 저장되었습니다: {model_path}")

    def load_local_model(self, model_dir):
        """
        로컬에 저장된 모델을 로드합니다.
        
        Args:
            model_dir (str): 모델이 저장된 디렉토리
        """
        if not os.path.exists(model_dir):
            raise ValueError(f"모델 디렉토리를 찾을 수 없습니다: {model_dir}")
            
        tokenizer_path = os.path.join(model_dir, "tokenizer")
        model_path = os.path.join(model_dir, "model")
        
        if not os.path.exists(tokenizer_path) or not os.path.exists(model_path):
            raise ValueError("토크나이저 또는 모델 파일이 없습니다.")
            
        print(f"로컬 모델 로드 중... ({model_dir})")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        # GPU 사용 가능한 경우 GPU로 이동
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
            print("GPU 사용 중")
        else:
            print("CPU 사용 중")
            
        print("로컬 모델 로드 완료")

    def generate_text(self, prompt, max_length=100, temperature=0.7, top_p=0.9):
        """
        주어진 프롬프트에 대한 텍스트를 생성합니다.
        
        Args:
            prompt (str): 입력 프롬프트
            max_length (int): 생성할 최대 토큰 수
            temperature (float): 생성 다양성 조절 (높을수록 더 다양한 출력)
            top_p (float): nucleus sampling 파라미터
            
        Returns:
            str: 생성된 텍스트
        """
        # 입력 텍스트 토큰화
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        # GPU 사용 가능한 경우 입력을 GPU로 이동
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        # 텍스트 생성
        outputs = self.model.generate(
            inputs["input_ids"],
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        # 생성된 텍스트 디코딩
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text

    def batch_generate(self, prompts, max_length=100, temperature=0.7, top_p=0.9):
        """
        여러 프롬프트에 대한 텍스트를 배치로 생성합니다.
        
        Args:
            prompts (list): 입력 프롬프트 리스트
            max_length (int): 생성할 최대 토큰 수
            temperature (float): 생성 다양성 조절
            top_p (float): nucleus sampling 파라미터
            
        Returns:
            list: 생성된 텍스트 리스트
        """
        generated_texts = []
        for prompt in prompts:
            generated_text = self.generate_text(
                prompt,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p
            )
            generated_texts.append(generated_text)
        return generated_texts

def main():
    # Hugging Face 토큰을 환경 변수에서 가져옴
    token = os.getenv('HUGGINGFACE_TOKEN')
    if not token:
        print("환경 변수 HUGGINGFACE_TOKEN이 설정되지 않았습니다.")
        print("다음 명령어로 환경 변수를 설정해주세요:")
        print("export HUGGINGFACE_TOKEN='your_token_here'")
        return
    
    print(f"현재 설정된 토큰: {token[:8]}...")  # 토큰의 처음 8자만 출력
    
    # 모델 저장 디렉토리 설정
    model_dir = "reference/models/gemma-3-4b-it"
    # google/gemma-3-1b-it
    
    # 모델 초기화 (처음 실행 시 다운로드)
    print("=== 모델 초기화 ===")
    gemma = GemmaInference(cache_dir=model_dir, token=token)
    
    # 모델 저장 (선택사항)
    print("\n=== 모델 저장 ===")
    gemma.save_model()
    
    # 저장된 모델 로드 (다음 실행 시 사용)
    print("\n=== 저장된 모델 로드 ===")
    gemma.load_local_model(model_dir)
    
    # 단일 프롬프트 예제
    prompt = "인공지능이란 무엇인가요?"
    print("\n=== 단일 프롬프트 예제 ===")
    print(f"프롬프트: {prompt}")
    print(f"생성된 텍스트: {gemma.generate_text(prompt)}")
    
    # 배치 프롬프트 예제
    prompts = [
        "딥러닝의 장점은 무엇인가요?",
        "머신러닝과 딥러닝의 차이점은 무엇인가요?",
        "강화학습의 응용 분야는 어떤 것들이 있나요?"
    ]
    print("\n=== 배치 프롬프트 예제 ===")
    generated_texts = gemma.batch_generate(prompts)
    for prompt, generated in zip(prompts, generated_texts):
        print(f"\n프롬프트: {prompt}")
        print(f"생성된 텍스트: {generated}")

if __name__ == "__main__":
    main() 