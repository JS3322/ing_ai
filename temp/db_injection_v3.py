from fastapi import FastAPI, Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response
import json
import re

app = FastAPI()

# ❌ 차단할 SQL Injection 위험 문자 패턴
BLACKLISTED_PATTERN = re.compile(r"[;'\"\\]|--|/\*|\*/|<|>")

async def validate_body(data, parent_field="body"):
    """Request Body 내부 값에서 SQL Injection 위험 문자 검사 (재귀적으로 탐색)"""
    if isinstance(data, dict):  # JSON 객체 내부 검사
        for key, value in data.items():
            await validate_body(value, f"{parent_field}.{key}")
    elif isinstance(data, list):  # 리스트 내부 값 검사
        for index, item in enumerate(data):
            await validate_body(item, f"{parent_field}[{index}]")
    elif isinstance(data, (int, float)):  # 숫자는 허용
        return
    elif isinstance(data, str):  # 문자열 검사
        if BLACKLISTED_PATTERN.search(data):
            raise HTTPException(
                status_code=400,
                detail=f"'{parent_field}' 값에 SQL Injection 위험 특수문자가 포함되어 있습니다."
            )

class SQLInjectionBodyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """POST, PUT, PATCH 요청의 Body에서 SQL Injection 위험 문자 검사"""
        if request.method in ["POST", "PUT", "PATCH"]:
            body_bytes = await request.body()
            if body_bytes:
                try:
                    body_json = json.loads(body_bytes.decode("utf-8"))
                    await validate_body(body_json, "body")
                except json.JSONDecodeError:
                    pass  # JSON 형식이 아닐 경우 검사하지 않음

        return await call_next(request)

# ✅ 미들웨어 등록 (Body만 검사)
app.add_middleware(SQLInjectionBodyMiddleware)

@app.post("/items/")
async def create_item(data: dict):
    return {"message": "POST 요청이 성공적으로 처리되었습니다!", "data": data}