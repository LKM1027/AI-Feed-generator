import os
from dotenv import load_dotenv
from google import genai

# .env 파일 강제 로드 (기존 캐시 무시)
load_dotenv(override=True)

# 클라이언트 생성 (환경 변수 GEMINI_API_KEY 자동 인식)
client = genai.Client()

print("🔍 현재 API 키로 사용 가능한 모델 목록을 조회합니다...\n")

try:
    # 사용 가능한 모든 모델 리스트를 가져와서 이름만 출력
    for model in client.models.list():
        print(f"- {model.name}")
    print("\n✅ 조회 완료!")
except Exception as e:
    print(f"❌ 오류 발생: {e}")