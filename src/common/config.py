import os
import getpass
from dotenv import load_dotenv


def find_project_root():
    """
    프로젝트 루트 디렉토리를 찾습니다.
    main.py 파일이 있는 디렉토리를 프로젝트 루트로 간주합니다.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    while current_dir:
        if os.path.exists(os.path.join(current_dir, 'main.py')):
            return current_dir
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:  # 루트 디렉토리에 도달
            break
        current_dir = parent_dir
    raise FileNotFoundError("프로젝트 루트 디렉토리(main.py가 있는 위치)를 찾을 수 없습니다.")

def load_environment():
    """
    현재 사용자 계정의 환경 변수 파일을 로드합니다.
    """
    # 프로젝트 루트 디렉토리 찾기
    project_root = find_project_root()
    
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