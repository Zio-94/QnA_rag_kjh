# src/streamlit_app/app.py
import streamlit as st
import requests
import json
import uuid
import logging

# --- 로깅 설정 (선택적) ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
# FastAPI 백엔드 서버 주소 (환경 변수나 설정 파일 사용 권장)
FASTAPI_BASE_URL = "http://localhost:8000" 
CHAT_ENDPOINT_URL = f"{FASTAPI_BASE_URL}/chat"

# --- Session State Initialization ---
# 앱이 재실행되어도 상태를 유지하기 위해 사용
if "conversation_id" not in st.session_state:
    # 각 새 세션마다 고유 ID 생성
    st.session_state.conversation_id = str(uuid.uuid4())
    logger.info(f"New conversation started with ID: {st.session_state.conversation_id}")
if "messages" not in st.session_state:
    # 채팅 기록 저장 리스트 초기화
    st.session_state.messages = [] # 형식: {"role": "user" or "assistant", "content": "..."}

# --- UI 구성 ---
st.title("스마트스토어 FAQ 챗봇 (Streamlit)")
st.caption(f"Conversation ID: {st.session_state.conversation_id}")

# 이전 대화 기록 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"]) # 마크다운 형식으로 내용 표시

# 사용자 입력 처리
if prompt := st.chat_input("스마트스토어에 대해 무엇이 궁금하신가요?"):
    logger.info(f"[{st.session_state.conversation_id}] User input: {prompt[:50]}...")
    # 1. 사용자 메시지를 세션 상태와 화면에 추가
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. 어시스턴트 응답 영역 준비 및 API 호출/스트리밍 처리
    with st.chat_message("assistant"):
        message_placeholder = st.empty() # 스트리밍 텍스트 표시될 영역
        full_response = "" # 전체 응답 저장용 변수

        try:
            # API 요청 데이터 준비
            request_data = {
                "message": prompt,
                "conversation_id": st.session_state.conversation_id,
            }
            logger.debug(f"[{st.session_state.conversation_id}] Sending request to API: {request_data}")

            # FastAPI 백엔드 스트리밍 API 호출
            # stream=True 필수, timeout 설정 권장
            response = requests.post(CHAT_ENDPOINT_URL, json=request_data, stream=True, timeout=180)
            response.raise_for_status() # 오류 발생 시 예외 발생 (4xx, 5xx)

            logger.info(f"[{st.session_state.conversation_id}] Receiving stream from API...")
            # SSE 스트림 처리
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith("data:"):
                        try:
                            data_str = decoded_line[len("data:"):].strip()
                            if data_str:
                                stream_data = json.loads(data_str)
                                token = stream_data.get("token")
                                is_final = stream_data.get("is_final", False)
                                error = stream_data.get("error", False)

                                if error:
                                    error_msg = f"오류 발생: {token}"
                                    logger.error(f"[{st.session_state.conversation_id}] Received error from API stream: {token}")
                                    st.error(error_msg)
                                    full_response += f"\n[{error_msg}]" # 오류도 기록
                                    break # 오류 시 스트림 중단

                                if is_final and token == "[DONE]":
                                    logger.info(f"[{st.session_state.conversation_id}] Stream finished signal received.")
                                    break # 종료 시그널 받으면 루프 종료

                                if token and token != "[DONE]":
                                    full_response += token
                                    # 스트리밍 효과: 받은 토큰 누적하여 표시 + 커서 효과
                                    message_placeholder.markdown(full_response + "▌")

                        except json.JSONDecodeError:
                            logger.warning(f"[{st.session_state.conversation_id}] Could not decode JSON from SSE event: {decoded_line}")
                        except Exception as parse_e:
                             logger.error(f"[{st.session_state.conversation_id}] Error parsing stream chunk: {parse_e}")
            # 스트림 완료 후 커서 제거
            message_placeholder.markdown(full_response)
            logger.info(f"[{st.session_state.conversation_id}] Finished processing stream. Full response length: {len(full_response)}")

        except requests.exceptions.ConnectionError as e:
            logger.error(f"[{st.session_state.conversation_id}] API Connection Error: {e}")
            st.error(f"API 서버 연결에 실패했습니다. FastAPI 서버가 실행 중인지 확인하세요: {e}")
            full_response = "[API 연결 오류]"
        except requests.exceptions.Timeout as e:
            logger.error(f"[{st.session_state.conversation_id}] API Timeout Error: {e}")
            st.error(f"API 요청 시간 초과: {e}")
            full_response = "[API 시간 초과 오류]"
        except requests.exceptions.RequestException as e: # 기타 requests 오류
            logger.error(f"[{st.session_state.conversation_id}] API Request Error: {e}")
            st.error(f"API 요청 중 오류 발생: {e}")
            full_response = "[API 요청 오류]"
        except Exception as e: # 예상치 못한 오류
            logger.exception(f"[{st.session_state.conversation_id}] An unexpected error occurred: {e}")
            st.error("예상치 못한 오류가 발생했습니다.")
            full_response = "[애플리케이션 오류]"

    # 3. 최종 어시스턴트 응답을 세션 상태에 추가 (오류 메시지도 포함)
    if full_response: # 내용이 있을 때만 추가
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        logger.debug(f"[{st.session_state.conversation_id}] Assistant response added to session state.")