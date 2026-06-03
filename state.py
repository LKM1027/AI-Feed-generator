from typing import TypedDict, Optional, List, Dict


class FeedGenerationState(TypedDict):
    original_image_path: str
    selected_style: str
    user_text: Optional[str]          # 사용자 추가 입력 (이벤트, 공지, 날짜 등)
    image_analysis: Dict[str, str]
    mood: str
    generation_prompt: str
    generated_image_url: str
    caption: str
    hashtags: List[str]
    current_status: str
    retry_count: int
    error_message: Optional[str]
