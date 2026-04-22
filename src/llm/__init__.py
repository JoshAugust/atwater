"""
src.llm — LLM integration layer for Atwater.

Exports:
- LMStudioClient: HTTP client for LM Studio's OpenAI-compatible API.
- SYSTEM_PROMPTS: Role → system prompt mapping.
- build_director_prompt, build_creator_prompt, build_grader_prompt,
  build_consolidator_prompt: Prompt builders per agent role.
"""

from .client import LMStudioClient
from .prompts import (
    SYSTEM_PROMPTS,
    build_consolidator_prompt,
    build_creator_prompt,
    build_director_prompt,
    build_grader_prompt,
)

__all__ = [
    "LMStudioClient",
    "SYSTEM_PROMPTS",
    "build_director_prompt",
    "build_creator_prompt",
    "build_grader_prompt",
    "build_consolidator_prompt",
]
