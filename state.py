from typing import TypedDict, Optional, List, Dict


class FeedGenerationState(TypedDict):
    original_image_path: str
    selected_style: str
    image_analysis: Dict[str, str]
    mood: str
    generation_prompt: str
    generated_image_url: str
    caption: str
    hashtags: List[str]
    current_status: str
    retry_count: int
    error_message: Optional[str]
