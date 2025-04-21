# FAQ 챗봇 서비스

이 프로젝트는 FastAPI와 Streamlit을 사용한 FAQ 챗봇 서비스입니다. 벡터 검색과 LLM을 활용하여 사용자 질문에 적절한 답변을 제공합니다.

## 시스템 요구사항

- Python 3.9+
- Redis 서버 (세션 관리 및 메모리 캐싱용)

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

## 애플리케이션 실행

### 1. Redis 서버 실행
Redis 서버는 애플리케이션 실행 전에 반드시 실행되어야 합니다:

```bash
# macOS (Homebrew)
brew services start redis

# Linux
sudo systemctl start redis

# Windows/기타
redis-server
```

### 2. 백엔드 API 실행
```
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Streamlit 프론트엔드 실행
```
# 별도 터미널에서 실행
streamlit run streamlit_app.py
```

## 데이터 초기화

애플리케이션은 시작 시(lifespan) 자동으로 데이터 초기화를 수행합니다. 데이터 초기화 옵션:

```python
# src/main.py의 lifespan 함수에서 설정 가능

# 기존 데이터 유지 모드
await ingestion_service.run_ingestion_pipeline(overwrite=False)

# 또는 데이터 초기화 모드 (기존 임베딩 삭제 후 재생성)
await ingestion_service.run_ingestion_pipeline(overwrite=True)
```

최초 실행 시에는 `overwrite=True`로 설정하여 임베딩 데이터를 생성해야 합니다.

## 주의사항

- Redis 서버가 실행되지 않으면 애플리케이션이 정상 작동하지 않습니다.
- 첫 실행 시 대용량 데이터 처리 및 임베딩 생성으로 시간이 소요될 수 있습니다.
- FastAPI와 Streamlit은 별도 프로세스로 실행되므로, 두 서버 모두 실행해야 완전한 기능을 사용할 수 있습니다.
