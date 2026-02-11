import os

from google import genai
from google.genai import types



# ================================
#   Class
# ================================
class AIAnswerEvaluator:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=api_key)

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
                        f"{ANSWER_RUBRIC_PROMPT}\nQUESTION: {question}\nPROMPT: {rubric}"
                        )

    def _send_image_prompt(self, image_bytes: bytes, prompt: str) -> str | None:
        """
        Send a chat completion request with the image input
        """
        image_encoded = types.Part.from_bytes(
                data=image_bytes,
                mime_type='image/jpeg'
                )

        response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[image_encoded, prompt]
                )

        return response.text
    


# ================================
#   Prompts
# ================================
ANSWER_RUBRIC_PROMPT: str = "You are given an image of a student's handwritten work in response to a math problem. Included in this prompt, preceded by `QUESTION:` is the problem the student is answering. Your task is to answer a question/questions (in a new line, preceded by `PROMPT:`) based solely on the visual content of the student's work. Your answer should be clear and concise, and directly relate to the image. If a question can be answered with a yes or no, only generate your answer as `YES` or `NO`. Otherwise, generate your answer as raw text, with no prefixes or sentences (e.g. `3x+2=8` or `5`). If multiple questions are given in the prompt, separate your answers for each with `; `."

COMPARE_EXPECTED_FINAL_ANSWER_PROMPT: str = "You are given an image of a student's handwritten work in response to a math problem. Included in this prompt, preceded by `QUESTION:` is the problem the student is answering. Your task is to evaluate if the student's final answer is the same as the expected final answer (in a new line, preceded by `ANSWER:`). Your answer should be clear and concise, and generated as only `YES` or `NO`. If the student does not have a clear final answer, generate your answer as `UNCLEAR`."

FIND_ITEM_NUMBER_PROMPT: str = "There is an item number indicated on the upper left corner of the image attached. It is encircled and may be written with poor handwriting, and/or have gridlines underneath them. Identify what it is; generate your answer simply (for example, just `2`). If there is no number, generate your answer as `NONE`."
