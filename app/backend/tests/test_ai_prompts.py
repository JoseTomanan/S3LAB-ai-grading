import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from logic.ai_interface import (
    ANSWER_RUBRIC_PROMPT,
    ANSWER_MULTI_RUBRIC_PROMPT,
    COMPARE_EXPECTED_FINAL_ANSWER_PROMPT,
    PROMPT_CUSTOM_INSTRUCTIONS,
)

GRADED_PROMPTS = [
    ("ANSWER_RUBRIC_PROMPT", ANSWER_RUBRIC_PROMPT),
    ("ANSWER_MULTI_RUBRIC_PROMPT", ANSWER_MULTI_RUBRIC_PROMPT),
    ("COMPARE_EXPECTED_FINAL_ANSWER_PROMPT", COMPARE_EXPECTED_FINAL_ANSWER_PROMPT),
]


@pytest.mark.parametrize("name,prompt", GRADED_PROMPTS)
def test_prompt_has_grade_level_context(name, prompt):
    assert "Grade 1" in prompt, f"{name} is missing grade-level context"


@pytest.mark.parametrize("name,prompt", GRADED_PROMPTS)
def test_prompt_embeds_custom_instructions(name, prompt):
    assert PROMPT_CUSTOM_INSTRUCTIONS in prompt, (
        f"{name} does not embed PROMPT_CUSTOM_INSTRUCTIONS verbatim"
    )


def test_custom_instructions_is_nonempty():
    assert PROMPT_CUSTOM_INSTRUCTIONS.strip(), "PROMPT_CUSTOM_INSTRUCTIONS must not be empty"


@pytest.mark.parametrize("keyword", [
    "spelling",
    "capitalization",
    "handwriting",
    "sentence",
    "digit",
])
def test_custom_instructions_covers_leniency_category(keyword):
    assert keyword.lower() in PROMPT_CUSTOM_INSTRUCTIONS.lower(), (
        f"PROMPT_CUSTOM_INSTRUCTIONS missing leniency category: '{keyword}'"
    )
