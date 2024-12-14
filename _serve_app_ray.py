import time
import random
import ray
from fastapi import FastAPI
from pydantic import BaseModel

# Ray 초기화 (로컬에서 실행)
ray.init()

app = FastAPI()

# Ray 태스크로 함수 정의
@ray.remote
def compute_heavy_task(data):
    """
    오래 걸리는 계산 작업을 Ray 태스크로 정의.
    data: 입력 데이터
    """
    print(f"Processing {data} on worker: {ray.util.get_node_ip_address()}")
    time.sleep(random.uniform(2, 5))  # 가상으로 오래 걸리는 연산
    return data * data


class TaskRequest(BaseModel):
    data: list[int]  # 여러 작업을 병렬 처리 가능하도록 리스트로 변경


@app.post("/execute")
def execute_task(request: TaskRequest):
    """
    Ray를 사용하여 main_task를 병렬로 실행.
    """
    # 입력 데이터 리스트를 Ray 태스크로 병렬 실행
    futures = [compute_heavy_task.remote(d) for d in request.data]

    # 모든 태스크의 결과를 기다림
    results = ray.get(futures)

    return {"results": results}