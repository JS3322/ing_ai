import os
import time
import random
import concurrent.futures
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse

app = FastAPI()
# CPU 바운드 작업에서는 ProcessPoolExecutor 사용
executor = concurrent.futures.ProcessPoolExecutor(max_workers=5)

def main_task(data):
    # 시간이 걸리는 작업 시뮬레이션
    time.sleep(random.uniform(0.5, 5.0))
    result_value = data * data
    print(f"Processing: {data}")
    return result_value

class TaskRequest(BaseModel):
    data: int

@app.post("/execute")
def execute_task(request: TaskRequest):
    # 요청 받은 data를 main_task에 전달하고 비동기로 실행
    print(f"core : {os.cpu_count()}")
    future = executor.submit(main_task, request.data)
    return JSONResponse(content={"message": "Task submitted"})
    # result = future.result()  # 여기서는 블록킹으로 결과 대기
    # return {"result": result}
