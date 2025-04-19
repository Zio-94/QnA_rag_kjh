# FastAPI 프로젝트 설정 및 실행 가이드

## 필요 사항
- Python 3.8 이상
- uv (Python 패키지 관리자)

## 환경 설정

### 1. uv 설치
아직 uv가 설치되어 있지 않다면:
```
pip install uv
```

### 2. 가상환경 생성 및 활성화
```
uv venv
source .venv/bin/activate  # Linux/Mac
# 또는
.venv\Scripts\activate     # Windows
```

### 3. 의존성 설치
```
uv pip install -r requirements.txt
```

주요 패키지:
- fastapi
- uvicorn
- chromadb

## 애플리케이션 실행

### 개발 서버 실행
```
uv run -m uvicorn src.main:app --reload
```

### 프로덕션 서버 실행
```
uv run -m uvicorn src.main:app --host 0.0.0.0 --port 8000
```

## API 접근
서버 실행 후 브라우저에서 다음 URL에 접속:
- API 메인 페이지: [http://localhost:8000](http://localhost:8000)
- Swagger 문서: [http://localhost:8000/docs](http://localhost:8000/docs)

## 주의사항
- `.env` 파일에 필요한 환경 변수가 설정되어 있는지 확인하세요
- 벡터 데이터베이스 경로(VECTOR_DB_PATH)는 유효한 디렉토리 경로여야 합니다
