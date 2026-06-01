from langgraph.graph import StateGraph, START, END

from state import FeedGenerationState
from nodes import (
    analyze_image,
    extract_mood,
    generate_prompt,
    generate_image,
    generate_caption,
    generate_hashtags,
)


def build_graph():
    graph = StateGraph(FeedGenerationState)

    graph.add_node("analyze_image", analyze_image)
    graph.add_node("extract_mood", extract_mood)
    graph.add_node("generate_prompt", generate_prompt)
    graph.add_node("generate_image", generate_image)
    graph.add_node("generate_caption", generate_caption)
    graph.add_node("generate_hashtags", generate_hashtags)

    graph.add_edge(START, "analyze_image")
    graph.add_edge("analyze_image", "extract_mood")
    graph.add_edge("extract_mood", "generate_prompt")
    graph.add_edge("generate_prompt", "generate_image")
    graph.add_edge("generate_image", "generate_caption")
    graph.add_edge("generate_caption", "generate_hashtags")
    graph.add_edge("generate_hashtags", END)

    app = graph.compile()
    return app
