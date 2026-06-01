# 🚀 AI Feed Generator: 도자기 공방 맞춤형 SNS 인스타 피드 생성 에이전트

> **LangGraph 기반의 다중 모델 페일오버(Failover) 아키텍처를 적용한 프리미엄 감성 마케팅 콘텐츠 자동화 플랫폼**
> 
> 본 프로젝트는 사용자가 업로드한 도자기 그릇, 컵 등의 이미지를 멀티모달(Multimodal) AI로 정밀 분석하여, 공방 고유의 브랜드 감성(Warm, Minimal, Japanese)에 맞는 인스타그램용 이미지, 본문 캡션, 해시태그 칩셋을 원클릭으로 최종 완성하는 든든한 AI 마케팅 비서 시스템입니다.

---

## 📌 1. 개발 배경 & 서비스 목표
* **개발 배경:** 수공예 공방을 운영하는 1인 작가들은 작품 제작 외에 매일 SNS 콘텐츠 기획 및 감성 카피라이팅에 막대한 시간과 리소스를 소모합니다. 또한, 기존 LLM API 툴은 매번 톤앤매너가 달라지거나 무료 티어 쿼터(Rate Limit) 제한으로 파이프라인이 쉽게 크래시 나는 한계가 있습니다.
* **서비스 목표:** '도자기 공방' 도메인에 특화된 정갈한 무드 프리셋을 결합하여 고품질 피드를 출력하고, 어떠한 API 제한 속에서도 무중단으로 작동하는 철벽 백엔드를 구축합니다.

---

## 🏃‍♂️ 2. 시작하기 (Quick Start)

### 1) 저장소 클론 및 가상환경 설정
```powershell
# 저장소 클론
git clone [https://github.com/본인_깃허브_아이디/AI-Feed-generator.git](https://github.com/본인_깃허브_아이디/AI-Feed-generator.git)
cd AI-Feed-generator

# 가상환경 생성 및 활성화
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 의존성 패키지 설치
pip install -r requirements.txt

```

### 2) 환경 변수 설정

프로젝트 루트 디렉토리에 `.env` 파일을 생성하고 구글 제미나이 API 키를 입력합니다.

```text
GEMINI_API_KEY=your_actual_gemini_api_key_here

```

### 3) 사용법 (Usage)

#### 🔹 옵션 A. CLI 모드로 파이프라인 검증 (기본 MVP)

가장 빠르게 LangGraph 상태 기반 파이프라인을 검증하기 위한 CLI 모드입니다. 윈도우 인코딩 크래시 방지 설정을 켜준 뒤 실행합니다.

```powershell
$env:PYTHONUTF8=1
python main.py "input/도자기컵.jpeg" warm

```

* *참고:* 이미지 생성(`Imagen 4.0`) 모델이 무료 쿼터 제한으로 실패할 경우, 전체 크래시 대신 우아한 폴백(Failover) 로직이 작동하여 캡션과 해시태그 텍스트 자산을 끝까지 안전하게 뽑아냅니다.

#### 🔹 옵션 B. 프리미엄 웹 UI 모드로 실행 (추천)

FastAPI 백엔드와 실전 시각화 다크모드 웹 페이지를 연동하여 GUI 환경에서 테스트합니다.

```powershell
python server.py

```

서버 구동 후 브라우저에서 `http://localhost:8000`으로 접속하여 드래그 앤 드롭으로 시연할 수 있습니다.

#### 🌐 옥외 공유 (ngrok 로컬 터널링)

팀원 공유 및 외부 모바일 기기 테스트를 위해 터널을 개통합니다.

```powershell
npx ngrok http 8000
# 터미널에 생성된 [https://xxxx.ngrok-free.dev](https://xxxx.ngrok-free.dev) 주소로 어디서나 접속 가능합니다.

```

---

## 🏗️ 3. 아키텍처 구조도 & 기술 스택

### 📊 파이프라인 워크플로우 흐름도

```text
[사용자 이미지 업로드] 
       │
       ▼
┌────────────────────────────────────────────────────────────────────────┐
│  LangGraph Workflow Agent Pipeline                                     │
│                                                                        │
│  1. analyze_image  ──> 2. extract_mood  ──> 3. generate_prompt         │
│         │                      │                     │                 │
│         ▼                      ▼                     ▼                 │
│  [gemini-2.0-flash]    [무드 컨텍스트 결합]    [영문 가속 프롬프트 셋]      │
│                                                                        │
│                                                                        │
│  6. generate_hashtags <── 5. generate_caption <── 4. generate_image    │
│         │                         │                      │             │
│         ▼                         ▼                      ▼             │
│  [인스타 칩셋 바인딩]      [이모지 매칭 감성본문]    [Imagen 4.0 Failover]    │
└────────────────────────────────────────────────────────────────────────┘

```

### 🧠 LangGraph 활용 및 AI 모델 다중화 구성 (Core Logic)

본 프로젝트는 선형적인 시퀀셜 파이프라인의 한계를 극복하고 각 단계의 상태(State)를 안전하게 전사 관리하기 위해 **LangGraph**를 도입했습니다. 특히, 무료 API 환경의 불안정성을 원천 차단하기 위해 **3단계 레이어 다중화(Failover) 체계**를 자체 구축했습니다.

1. **텍스트 노드 다중화 (Text Multi-Quota):** 일일 20회 제한이 있는 `gemini-2.5-flash` 대신, 1일 1,500회 호출이 가능한 `gemini-2.0-flash-lite`를 주 모델로 설정하고 한도 초과 시 독립 버킷인 `gemini-2.0-flash`로 즉시 스위칭합니다.
2. **이미지 생성 노드 폴백 (Image Failover):** 무료 쿼터 차단에 대응하여 `gemini-2.5-flash-image` ➡️ `gemini-3.1-flash-image` ➡️ `imagen-4.0-fast-generate-001` 순으로 지능형 백오프 재시도 및 순차 낙하 시도를 처리합니다.
3. **우아한 열화 (Graceful Degradation):** 이미지 생성 모델이 모두 차단되더라도 전체 파이프라인이 터지지 않고 빈 경로를 반환하여, **인스타 캡션과 해시태그 칩셋은 끝까지 무사히 완성**되도록 견고하게 방어합니다.

### 💻 기술 스택 요약

* **Backend:** Python 3.11+, FastAPI, LangGraph, Pydantic, `google-genai` SDK
* **Frontend:** Vanilla JS, Fetch API, Tailwind CSS (Premium Dark Mode UI)
* **DevOps / Infra:** Git (Git Flow 브랜치 전략), `ngrok` Local Tunneling