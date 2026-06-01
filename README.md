# 도자기 공방 인스타그램 피드 자동 생성 AI 파이프라인 (MVP)

빠르게 LangGraph 상태 기반 파이프라인을 검증하기 위한 CLI MVP입니다.

## 준비

1. 가상환경 생성 및 활성화

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. 의존성 설치

```powershell
pip install -r requirements.txt
```

3. `.env` 설정 (예: `.env.example` 참고)

```text
GEMINI_API_KEY=your_api_key_here
```

## 사용법

```powershell
python main.py "path/to/input.jpg" warm
```

노드 중 일부는 Gemini 연동이 되지 않았을 경우 폴백 로직으로 동작합니다. `generate_image` 노드는 아직 더미 URL을 반환합니다.
