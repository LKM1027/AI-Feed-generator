"""사용 가능한 이미지 생성 관련 모델 목록 확인"""
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import os
from dotenv import load_dotenv
load_dotenv()

from google import genai

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

print("=== 이미지 생성 관련 모델 ===")
for m in client.models.list():
    name = m.name
    if any(kw in name.lower() for kw in ["image", "imagen", "flash", "gemini-2"]):
        methods = getattr(m, "supported_actions", None) or getattr(m, "supported_generation_methods", [])
        print(f"  {name}  |  methods: {methods}")
