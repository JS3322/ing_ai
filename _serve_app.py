# main.py
# 가정: 해당 디렉토리에 venv가 있고 필요한 라이브러리(FastAPI 등) 설치됨

import time
import random
import concurrent.futures
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# 병렬 처리 풀(쓰레드 풀 또는 프로세스 풀)
# CPU 바운드 작업이라면 ProcessPoolExecutor 고려 가능
executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)

def main_task(data):
    # 시간이 걸리는 작업 시뮬레이션
    time.sleep(random.uniform(0.5, 5.0))
    result_value = data * data
    print(data)
    return result_value

class TaskRequest(BaseModel):
    data: int

@app.post("/execute")
def execute_task(request: TaskRequest):
    # 요청 받은 data를 main_task에 전달하고 비동기로 실행
    future = executor.submit(main_task, request.data)
    result = future.result()  # 여기서는 블록킹으로 결과 대기(비동기 응답 필요시 await 사용)
    return {"result": result}