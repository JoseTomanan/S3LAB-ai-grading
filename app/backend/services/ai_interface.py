import os
from dotenv import load_dotenv
import cv2
import numpy as np
from google import genai
from google.genai import types

import logging
import json
logger = logging.getLogger(__name__)



# ================================
#region   Class
class AIAnswerEvaluator:
    def __init__(self, flash: bool = False):
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=api_key)
        self.version = "gemini-2.5-flash" if flash else "gemini-2.5-flash-lite"

    def get_item_number(self, image_bytes: bytes):
        return self._send_image_prompt(image_bytes, FIND_ITEM_NUMBER_PROMPT)
    
    def evaluate_expected_answer(self, image_bytes: bytes, question: str, answer: str):
        return self._send_image_prompt(
                        image_bytes,
                        f"{COMPARE_EXPECTED_FINAL_ANSWER_PROMPT}\nQUESTION: {question}\nANSWER: {answer}"
                        )
        
    def evaluate_rubric(self, image_bytes: bytes, question: str, rubric: str):
        return self._send_image_prompt(
                        image_bytes,
                        f"{ANSWER_RUBRIC_PROMPT}\nQUESTION: {question}\nPROMPT: \"{rubric}\": can this be said about the answer?"
                        )
    
    def find_four_points(self, image_bytes: bytes) -> list[tuple[int, int]] | None:
        response = self._send_image_prompt(image_bytes, DETECT_CORNERS_PROMPT)
        
        try:
            assert isinstance(response, str)
            corners = json.loads(response)
            return corners
        
        except (json.JSONDecodeError, TypeError):
            logger.error(f"Failed to parse corner coordinates: {response}")
            return None

    def _send_image_prompt(self, image_bytes: bytes, prompt: str) -> str | None:
        """
        Send a chat completion request with the image input
        """
        image_encoded = types.Part.from_bytes(
                            data=image_bytes,
                            mime_type='image/jpeg'
                            )

        response = self.client.models.generate_content(
                    model=self.version,
                    contents=[image_encoded, prompt]
                    )

        return response.text
#endregion
# ================================
    


# ================================
#region   Prompts
ANSWER_RUBRIC_PROMPT: str = "You are given an image of a student's handwritten work in response to a math problem. Included in this prompt, preceded by `QUESTION:` is the problem the student is answering. Your task is to answer a question/questions (in a new line, preceded by `PROMPT:`) based solely on the visual content of the student's work. Your answer should be clear and concise, and directly relate to the image. If a question can be answered with a yes or no, only generate your answer as `YES` or `NO`. Otherwise, generate your answer as raw text, with no prefixes or sentences (e.g. `3x+2=8` or `5`). If multiple questions are given in the prompt, separate your answers for each with `; `."


COMPARE_EXPECTED_FINAL_ANSWER_PROMPT: str = "You are given an image of a student's handwritten work in response to a math problem. Included in this prompt, preceded by `QUESTION:` is the problem the student is answering. Your task is to evaluate if the student's final answer is the same as the expected final answer (in a new line, preceded by `ANSWER:`). Your answer should be clear and concise, and generated as only `YES` or `NO`. If the student does not have a clear final answer, generate your answer as `UNCLEAR`."


DETECT_CORNERS_PROMPT = "You are analyzing an image of a document/page that may be skewed or rotated.  Identify the 4 corners of the document's rectangular boundary.\nReturn ONLY a JSON array with 4 coordinate pairs in this EXACT order: [[x_top_left, y_top_left], [x_top_right, y_top_right], [x_bottom_right, y_bottom_right], [x_bottom_left, y_bottom_left]].\nWhere (0,0) is the top-left of the image. Use integers only. \nExample: [[10, 15], [590, 12], [595, 470], [5, 475]]"


FIND_ITEM_NUMBER_PROMPT: str = """You are identifying an encircled item number in the TOP-LEFT corner of a student's answer sheet.

IMPORTANT RULES:
1. Look ONLY at the upper-left corner region
2. The number is encircled (has a circle around it)
3. Common numbers are: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
4. Handwriting may be poor - look carefully at the shape
5. Do NOT confuse similar numbers (1 vs 7, 2 vs 3, 5 vs 6)

EXAMPLES:
- A single vertical line = 1
- A curve with horizontal base = 2
- Two curves stacked = 3
- Vertical line with horizontal cross = 4

Generate ONLY the number (e.g., `2`) or `NONE` if no number exists.
Do not add any explanation or text."""

#endregion
# ================================
