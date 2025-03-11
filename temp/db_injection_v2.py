from fastapi import FastAPI, Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response
import json
import re

app = FastAPI()

# 차단할 위험 문자 패턴 (SQL Injection 및 위험한 문자)
BLACKLISTED_PATTERN = re.compile(r"[;'\"\\]|--|/\*|\*/|<|>")

async def validate_input(data, parent_field="body"):
    """재귀적으로 모든 값 검사 (SQL Injection 위험 문자 필터링)"""
    if isinstance(data, dict):
        for key, value in data.items():
            await validate_input(value, f"{parent_field}.{key}")
    elif isinstance(data, list):
        for index, item in enumerate(data):
            await validate_input(item, f"{parent_field}[{index}]")
    elif isinstance(data, (int, float)):
        return  # 숫자는 허용
    elif isinstance(data, str):
        if BLACKLISTED_PATTERN.search(data):
            raise HTTPException(
                status_code=400,
                detail=f"'{parent_field}' 값에 SQL Injection 위험 특수문자가 포함되어 있습니다."
            )

class SQLInjectionMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # 1️⃣ Query Parameters 검사
        query_params = dict(request.query_params)
        await validate_input(query_params, "query")

        # 2️⃣ Request Body 검사 (POST, PUT, PATCH)
        if request.method in ["POST", "PUT", "PATCH"]:
            body_bytes = await request.body()
            if body_bytes:
                try:
                    body_json = json.loads(body_bytes.decode("utf-8"))
                    await validate_input(body_json, "body")
                except json.JSONDecodeError:
                    # JSON 형식이 아닐 경우 검사하지 않음
                    pass

        # 3️⃣ Headers 검사
        headers = dict(request.headers)
        await validate_input(headers, "headers")

        return await call_next(request)

# add_middleware를 통해 미들웨어 추가
app.add_middleware(SQLInjectionMiddleware)

@app.get("/items/")
async def get_items():
    return {"message": "GET 요청이 성공적으로 처리되었습니다!"}

@app.post("/items/")
async def create_item(data: dict):
    return {"message": "POST 요청이 성공적으로 처리되었습니다!", "data": data}