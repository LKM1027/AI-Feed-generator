import argparse
import os
import sys

# Windows cp949 콘솔에서 이모지/한글 출력 오류 방지
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv

from graph import build_graph
from state import FeedGenerationState


def build_initial_state(image_path: str, style: str) -> FeedGenerationState:
    return {
        "original_image_path": image_path,
        "selected_style": style,
        "image_analysis": {},
        "mood": "",
        "generation_prompt": "",
        "generated_image_url": "",
        "caption": "",
        "hashtags": [],
        "current_status": "initialized",
        "retry_count": 0,
        "error_message": None,
    }


def print_final_state(state: FeedGenerationState) -> None:
    print("\n=== 최종 상태 ===")
    print(f"original_image_path: {state['original_image_path']}")
    print(f"selected_style: {state['selected_style']}")
    print(f"image_analysis: {state['image_analysis']}")
    print(f"mood: {state['mood']}")
    print(f"generation_prompt: {state['generation_prompt']}")
    print(f"generated_image_path: {state['generated_image_url']}")
    print(f"caption: {state['caption']}")
    print(f"hashtags: {state['hashtags']}")
    print(f"current_status: {state['current_status']}")
    print(f"retry_count: {state['retry_count']}")
    print(f"error_message: {state['error_message']}")


def main() -> None:
    # load environment early so nodes can access GEMINI_API_KEY
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="도자기 공방 인스타그램 피드 자동 생성 AI 파이프라인 MVP"
    )
    parser.add_argument("image_path", help="입력 이미지 경로")
    parser.add_argument("style", help="선택한 스타일 예: warm, minimal")
    args = parser.parse_args()

    app = build_graph()
    state = build_initial_state(args.image_path, args.style)

    print("[*] 파이프라인 실행 시작")
    final_state = app.invoke(state)
    print("[*] 파이프라인 실행 완료")

    print_final_state(final_state)


if __name__ == "__main__":
    main()
