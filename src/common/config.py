import os
import getpass
from dotenv import load_dotenv


def load_environment():
    """
    현재 사용자 계정의 환경 변수 파일만 로드합니다.
    """
    # 프로젝트 루트 디렉토리 찾기
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    
    # 사용자별 환경 파일 로드
    current_user = getpass.getuser()
    user_env_path = os.path.join(project_root, f'.env.{current_user}')
    
    if os.path.exists(user_env_path):
        load_dotenv(user_env_path)
        print(f".env.{current_user} 파일을 로드했습니다.")
    else:
        print(f"Error: .env.{current_user} 파일을 찾을 수 없습니다.")
        print("환경 변수 파일이 없으면 프로그램이 정상적으로 동작하지 않을 수 있습니다.")
    
    return project_root 

def get_env_var(key, default=None):
    """
    환경 변수 값을 가져옵니다.
    
    Args:
        key (str): 환경 변수 키
        default: 키가 없을 경우 반환할 기본값
    
    Returns:
        환경 변수 값 또는 기본값
    """
    return os.getenv(key, default)