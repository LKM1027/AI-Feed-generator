from typing import Dict, Any, List
import os
import json
import re
import time
import datetime
from io import BytesIO
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from PIL import Image

import urllib.request
import urllib.parse

from state import FeedGenerationState

# google-genai imports (modern SDK)
from google import genai
from google.genai import types

load_dotenv(override=True)

# ========== Pydantic Schemas (Structured Outputs) ==========
class ImageAnalysisResult(BaseModel):
    subject: str = Field(description="사진에 등장하는 주요 객체 (예: 백자 달항아리, 분청사기 기물 등)")
    technique: str = Field(description="도자기 제작 기법이나 현재 상황 (예: 물레 성형 중, 가마 소성 후, 시유 작업 등)")
    texture: str = Field(description="도자기나 흙의 표면 질감 표현 (예: 매트하고 거친 느낌, 은은한 유약의 광택 등)")
    lighting: str = Field(description="사진의 조명 상태 (예: 창가에서 들어오는 은은한 오후 자연광, 선명한 실내 형광등)")


class MoodExtractionResult(BaseModel):
    mood_keywords: List[str] = Field(description="감성 무드 키워드 3~5개 (한국어)")
    mood_summary: str = Field(description="종합 감성 설명 (한국어, 1~2문장)")


class PromptGenerationResult(BaseModel):
    prompt: str = Field(description="영어로 작성된 이미지 생성 프롬프트 (Imagen/DALL-E 호환)")


class CaptionGenerationResult(BaseModel):
    caption: str = Field(description="인스타그램 캡션 (한국어, 이모지 포함, 2~3문장)")


class HashtagGenerationResult(BaseModel):
    hashtags: List[str] = Field(description="인스타그램 해시태그 5~7개")


# ========== Helper utilities ==========
def get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY가 .env 파일에 없거나 로드되지 않았습니다.")
    print(f"[DEBUG] 현재 주입된 API 키 끝 4자리: {api_key[-4:]}")
    return genai.Client(api_key=api_key)


def _log(step: str) -> None:
    print(f"[*] {step} 진행 중...")


def _call_with_retry(fn, max_retries: int = 3):
    """429 Rate Limit 발생 시 자동 재시도.
    - 일일 한도(PerDay) 소진 시는 즉시 raise(재시도 소용없음)
    - 분당 한도(PerMinute) 시는 API 권장 대기 후 재시도
    """
    last_exc = None
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            err_str = str(e)
            if "429" not in err_str:
                raise  # 429 아님 → 즉시 raise
            if "PerDay" in err_str:
                raise  # 일일 한도 → 재시도해도 소용없음
            if attempt == max_retries - 1:
                raise
            # 분당 한도 → API 권장 retryDelay만큼 대기
            match = re.search(r"retryDelay.*?(\d+(?:\.\d+)?)s", err_str)
            wait = float(match.group(1)) if match else 15.0 * (attempt + 1)
            wait = min(wait + 2, 90)  # 최대 90초
            print(f"[!] Rate limit 감지 — {wait:.0f}초 후 재시도 ({attempt + 1}/{max_retries - 1})...")
            time.sleep(wait)
    raise last_exc


# 텍스트 노드용 모델 우선순위 (독립적인 quota 버킷)
TEXT_MODELS = ["gemini-2.5-flash", "gemini-2.0-flash"]


def _generate_text(client, contents, config) -> Any:
    """TEXT_MODELS 순서로 시도하는 텍스트 생성 헬퍼.
    일일 한도 소진 시 다음 모델로 자동 전환.
    """
    last_exc: Exception = RuntimeError("모델 목록 비어있음")
    for model in TEXT_MODELS:
        try:
            return _call_with_retry(
                lambda m=model: client.models.generate_content(
                    model=m, contents=contents, config=config
                )
            )
        except Exception as e:
            last_exc = e
            err_str = str(e)
            if "429" in err_str and "PerDay" in err_str:
                print(f"[!] {model} 일일 한도 소진 → 다음 모델 시도...")
                continue
            raise  # 다른 오류는 즉시 raise
    raise last_exc


PRESETS_DIR = os.path.join(os.path.dirname(__file__), "prompts")


def load_style_preset(style: str) -> Dict[str, str]:
    preset_name = style if style else "warm"
    preset_path = os.path.join(PRESETS_DIR, f"{preset_name}.txt")
    fallback_path = os.path.join(PRESETS_DIR, "warm.txt")
    try:
        with open(preset_path, "r", encoding="utf-8") as f:
            return {"preset_text": f.read(), "error_message": ""}
    except FileNotFoundError:
        try:
            with open(fallback_path, "r", encoding="utf-8") as f:
                return {"preset_text": f.read(), "error_message": f"preset file not found for style '{preset_name}', fallback to warm.txt."}
        except Exception as e:
            return {"preset_text": "A warm, cozy pottery workshop scene.", "error_message": f"failed to load fallback preset: {str(e)}"}
    except Exception as e:
        try:
            with open(fallback_path, "r", encoding="utf-8") as f:
                return {"preset_text": f.read(), "error_message": f"failed to load preset '{preset_name}': {str(e)}, fallback to warm.txt."}
        except Exception as fallback_error:
            return {"preset_text": "A warm, cozy pottery workshop scene.", "error_message": f"failed to load preset '{preset_name}' and fallback preset: {str(fallback_error)}"}


# ========== Node implementations (return partial state dicts) ==========
def analyze_image(state: FeedGenerationState) -> Dict[str, Any]:
    _log("이미지 분석")
    try:
        client = get_gemini_client()

        img_path = state.get("original_image_path")
        if not img_path:
            raise ValueError("original_image_path is empty")

        with open(img_path, "rb") as f:
            image_bytes = f.read()

        prompt = (
            "이 사진은 도자기 공방 사장님이 인스타그램에 올리기 위해 찍은 사진입니다. "
            "사진에 등장하는 주요 객체(subject), 제작 기법이나 상황(technique), 전체적인 질감(texture), "
            "그리고 조명 상태(lighting)를 분석해서 한국어로 핵심만 요약해 주세요."
        )

        response = _generate_text(
            client,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                prompt,
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ImageAnalysisResult,
            ),
        )

        result = ImageAnalysisResult.model_validate_json(response.text)
        return {"image_analysis": result.model_dump(), "current_status": "image_analysis updated"}
    except Exception as e:
        return {"error_message": f"image_analysis failed: {str(e)}", "current_status": "image_analysis_failed"}


def extract_mood(state: FeedGenerationState) -> Dict[str, Any]:
    _log("무드 추출")
    try:
        client = get_gemini_client()

        analysis = state.get("image_analysis", {})
        style = state.get("selected_style", "")

        prompt = (
            f"아래 이미지 분석과 스타일을 참고하여 인스타그램 피드에 어울리는 감성 무드를 추출하세요.\n"
            f"이미지 분석: {json.dumps(analysis, ensure_ascii=False) if isinstance(analysis, dict) else str(analysis)}\n"
            f"선택된 스타일: {style}\n"
            f"출력은 JSON 포맷으로, 필드 'mood_keywords' (한국어 리스트)와 'mood_summary' (한국어 짧은 문장)를 포함해주세요."
        )

        response = _generate_text(
            client,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=MoodExtractionResult,
            ),
        )

        result = MoodExtractionResult.model_validate_json(response.text)
        return {"mood": result.mood_summary, "current_status": "mood extracted"}
    except Exception as e:
        return {"error_message": f"mood_extraction failed: {str(e)}", "current_status": "mood_extraction_failed"}


def generate_prompt(state: FeedGenerationState) -> Dict[str, Any]:
    _log("프롬프트 생성")
    preset_error = ""
    try:
        client = get_gemini_client()

        mood = state.get("mood", "")
        selected_style = state.get("selected_style", "warm")
        analysis = state.get("image_analysis", {})

        preset = load_style_preset(selected_style)
        preset_text = preset.get("preset_text", "")
        preset_error = preset.get("error_message", "")

        prompt = (
            f"Using the following style preset and image analysis, create an English image-generation prompt for a casual, realistic Instagram snapshot.\n"
            f"Style preset:\n{preset_text}\n\n"
            f"Mood: {mood}\n"
            f"Image Details: {json.dumps(analysis, ensure_ascii=False) if isinstance(analysis, dict) else str(analysis)}\n\n"
            f"[CRITICAL RULES FOR REALISM]\n"
            f"1. VIBE: 'Shot on iPhone', candid amateur photography, natural window light. STRICTLY AVOID cinematic, dramatic, or 3D rendered looks.\n"
            f"2. TEXTURE: Keep the pottery surface simple, natural, and slightly imperfect. STRICTLY AVOID generating intricate, unnatural, or alien-like carved patterns."
        )

        response = _generate_text(
            client,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=PromptGenerationResult,
            ),
        )

        result = PromptGenerationResult.model_validate_json(response.text)
        output: Dict[str, Any] = {"generation_prompt": result.prompt, "current_status": "generation_prompt created"}
        if preset_error:
            output["error_message"] = preset_error
        return output
    except Exception as e:
        return {"error_message": f"prompt_generation failed: {str(e)}" + (f" | {preset_error}" if preset_error else ""), "current_status": "generation_prompt_failed"}


def _save_image_bytes(image_bytes: bytes) -> str:
    """이미지 바이트를 output/ 폴더에 저장하고 경로 반환"""
    pil_image = Image.open(BytesIO(image_bytes))
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"generated_{timestamp}.png")
    pil_image.save(output_path)
    return output_path

'''
def generate_image(state: FeedGenerationState) -> Dict[str, Any]:
    _log("이미지 생성")

    generation_prompt = state.get("generation_prompt", "")
    if not generation_prompt:
        return {
            "error_message": "generation_prompt가 비어 있습니다.",
            "current_status": "image_generation_failed",
        }

    client = get_gemini_client()
    errors: List[str] = []

    # ── 시도 1: gemini-2.5-flash-image (generateContent + IMAGE modality) ──
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=generation_prompt,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"],
            ),
        )
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                return {
                    "generated_image_url": _save_image_bytes(part.inline_data.data),
                    "current_status": "generated_image_url created",
                }
        errors.append("gemini-2.5-flash-image: 이미지 파트 없음")
    except Exception as e:
        errors.append(f"gemini-2.5-flash-image: {e}")
        print(f"[!] gemini-2.5-flash-image 실패 → 다음 모델 시도")

    # ── 시도 2: gemini-3.1-flash-image (generateContent + IMAGE modality) ──
    try:
        response = client.models.generate_content(
            model="gemini-3.1-flash-image",
            contents=generation_prompt,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"],
            ),
        )
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                return {
                    "generated_image_url": _save_image_bytes(part.inline_data.data),
                    "current_status": "generated_image_url created",
                }
        errors.append("gemini-3.1-flash-image: 이미지 파트 없음")
    except Exception as e:
        errors.append(f"gemini-3.1-flash-image: {e}")
        print(f"[!] gemini-3.1-flash-image 실패 → 다음 모델 시도")

    # ── 시도 3: imagen-4.0-fast-generate-001 (generate_images, 유료 전용) ──
    try:
        response = client.models.generate_images(
            model="imagen-4.0-fast-generate-001",
            prompt=generation_prompt,
            config=types.GenerateImagesConfig(
                number_of_images=1,
                aspect_ratio="1:1",
            ),
        )
        if response.generated_images:
            return {
                "generated_image_url": _save_image_bytes(
                    response.generated_images[0].image.image_bytes
                ),
                "current_status": "generated_image_url created",
            }
        errors.append("imagen-4.0-fast-generate-001: 이미지 없음")
    except Exception as e:
        errors.append(f"imagen-4.0-fast-generate-001: {e}")
        print(f"[!] imagen-4.0-fast-generate-001 실패")

    # ── 전체 실패 → 파이프라인 계속 진행 (크래시 방지) ──
    error_summary = " | ".join(errors)
    print(f"[!] 이미지 생성 전체 실패: {error_summary}")
    return {
        "generated_image_url": "",
        "error_message": f"image_generation failed: {error_summary}",
        "current_status": "image_generation_failed",
    }
'''

def generate_image(state: FeedGenerationState) -> Dict[str, Any]:
    _log("이미지 생성 (무료 API 우회 중)")

    generation_prompt = state.get("generation_prompt", "")
    if not generation_prompt:
        return {
            "error_message": "generation_prompt가 비어 있습니다.",
            "current_status": "image_generation_failed",
        }

    try:
        # 1. 영어 프롬프트를 URL에 넣을 수 있도록 인코딩
        encoded_prompt = urllib.parse.quote(generation_prompt)
        
        # 2. Pollinations.ai에 '그리지 말아야 할 것' 추가
        negative_words = "cinematic, 3D render, over-polished, alien patterns, artificial, plastic, intricate carvings, dramatic lighting, studio setup"
        encoded_negative = urllib.parse.quote(negative_words)
        
        # 2. Pollinations.ai 무료 API URL 구성 (가로세로 1024px, 로고 제거)
        url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1080&height=1350&nologo=true&negative_prompt={encoded_negative}"
        
        # 3. API에 요청을 보내서 이미지 바이트 데이터 받아오기
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            image_bytes = response.read()

        # 4. 사장님이 기존에 만들어둔 저장 함수를 그대로 호출하여 파일로 저장
        saved_path = _save_image_bytes(image_bytes)

        return {
            "generated_image_url": saved_path,
            "current_status": "generated_image_url created",
        }

    except Exception as e:
        print(f"[!] 무료 이미지 API 실패: {str(e)}")
        return {
            "generated_image_url": "",
            "error_message": f"image_generation failed: {str(e)}",
            "current_status": "image_generation_failed",
        }



def generate_caption(state: FeedGenerationState) -> Dict[str, Any]:
    _log("캡션 생성")
    try:
        client = get_gemini_client()

        mood     = state.get("mood", "")
        analysis = state.get("image_analysis", {})
        user_text = (state.get("user_text") or "").strip()

        # user_text 존재 여부에 따라 프롬프트 분기
        if user_text:
            user_text_instruction = (
                f"\n\n[추가 정보 반영 규칙]\n"
                f"사용자가 다음 추가 정보를 입력했습니다: \"{user_text}\"\n"
                f"공방 무드와 이미지 감성을 유지하면서, "
                f"위 내용(이벤트·공지·날짜 등)이 인스타그램 본문 카피 안에 자연스럽게 녹아들도록 작성하세요. "
                f"광고처럼 딱딱하지 않고, 감성적인 흐름 속에 정보가 스며들어야 합니다."
            )
        else:
            user_text_instruction = (
                "\n\n[추가 정보 없음]\n"
                "이미지와 무드 중심의 감성 문구만 작성하세요."
            )

        prompt = (
            f"인스타그램에 바로 올릴 수 있는 친근하고 감성적인 한국어 캡션을 작성하세요.\n"
            f"요구사항: 2~3문장, 이모지 포함 (예: 🍯, 🏺, ✨), 따뜻하고 정갈한 톤.\n"
            f"무드: {mood}\n"
            f"이미지 상세: {json.dumps(analysis, ensure_ascii=False) if isinstance(analysis, dict) else str(analysis)}"
            f"{user_text_instruction}"
        )

        response = _generate_text(
            client,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=CaptionGenerationResult,
            ),
        )

        result = CaptionGenerationResult.model_validate_json(response.text)
        return {"caption": result.caption, "current_status": "caption generated"}
    except Exception as e:
        return {"error_message": f"caption_generation failed: {str(e)}", "current_status": "caption_failed"}



def generate_hashtags(state: FeedGenerationState) -> Dict[str, Any]:
    _log("해시태그 생성")
    try:
        client = get_gemini_client()

        caption = state.get("caption", "")
        mood = state.get("mood", "")

        prompt = (
            f"생성된 캡션과 무드를 참고하여 인스타그램 노출에 유리한 해시태그 5~7개를 한국어/영어 혼합 형태로 추출하세요.\n"
            f"캡션: {caption}\n"
            f"무드: {mood}\n"
            f"출력은 JSON 포맷으로, 필드 'hashtags'를 리스트로 반환해주세요."
        )

        response = _generate_text(
            client,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=HashtagGenerationResult,
            ),
        )

        result = HashtagGenerationResult.model_validate_json(response.text)
        return {"hashtags": result.hashtags, "current_status": "hashtags generated"}
    except Exception as e:
        return {"error_message": f"hashtags_generation failed: {str(e)}", "current_status": "hashtags_failed"}
