from fastapi import FastAPI, Request, HTTPException
import re
import json

app = FastAPI()

# ❌ 차단할 특수문자 (SQL Injection 및 위험한 문자)
BLACKLISTED_PATTERN = re.compile(r"[;'\"\\--|/*<>]")

async def validate_input(data, parent_field="body"):
    """재귀적으로 모든 값 검사 (SQL Injection 위험 문자 필터링)"""
    if isinstance(data, dict):  # 딕셔너리 내부 값 검사
        for key, value in data.items():
            await validate_input(value, f"{parent_field}.{key}")
    elif isinstance(data, list):  # 리스트 내부 값 검사
        for index, item in enumerate(data):
            await validate_input(item, f"{parent_field}[{index}]")
    elif isinstance(data, (int, float)):  # 숫자는 허용
        return
    elif isinstance(data, str):  # 문자열이면 SQL Injection 위험 문자 검사
        if BLACKLISTED_PATTERN.search(data):  # 금지된 특수문자가 포함된 경우
            raise HTTPException(status_code=400, detail=f"'{parent_field}' 값에 SQL Injection 위험 특수문자가 포함되어 있습니다.")

@app.middleware("http")
async def sql_injection_middleware(request: Request, call_next):
    """모든 요청에서 SQL 인젝션 필터링"""
    # 1️⃣ Query Parameters 검사 (GET 요청)
    query_params = dict(request.query_params)
    await validate_input(query_params, "query")

    # 2️⃣ Request Body 검사 (POST, PUT, PATCH 요청)
    if request.method in ["POST", "PUT", "PATCH"]:
        try:
            body = await request.body()
            if body:  # Body가 존재하면 JSON 변환 후 검사
                body_json = json.loads(body.decode("utf-8"))
                await validate_input(body_json, "body")
        except json.JSONDecodeError:
            pass  # JSON이 아닌 경우 무시

    # 3️⃣ Headers 검사
    headers = dict(request.headers)
    await validate_input(headers, "headers")

    return await call_next(request)

@app.get("/items/")
async def get_items():
    return {"message": "GET 요청이 성공적으로 처리되었습니다!"}

@app.post("/items/")
async def create_item(data: dict):
    return {"message": "POST 요청이 성공적으로 처리되었습니다!", "data": data}