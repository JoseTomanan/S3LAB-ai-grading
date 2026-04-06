import os
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types



# ================================
#region   Class
class AIAnswerEvaluator:
    def __init__(self):
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=api_key)
        self.pro_version = "gemini-2.5-pro"
        self.flash_version = "gemini-2.5-flash"

    def get_item_number(self, image_bytes: bytes):
        return self._send_image_prompt(image_bytes, FIND_ITEM_NUMBER_PROMPT)

    def get_nearest_item_number(self, image_bytes: bytes, label_choices: list[str]):
        return self._send_image_prompt(image_bytes, FIND_NEAREST_ITEM_NUMBER_PROMPT(label_choices))
    
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

    def evaluate_multi_rubric(self, image_bytes: bytes, question: str, rubrics: list[str]):
        rubrics_formatted = json.dumps(rubrics)
        return self._send_image_prompt(
                        image_bytes,
                        f"{ANSWER_MULTI_RUBRIC_PROMPT}\nQUESTION: {question}\nRUBRICS: {rubrics_formatted}"
                        )
    
    def find_four_points(self, image_bytes: bytes) -> list[tuple[int, int]] | None:
        """[UNUSED] AI-based fallback for corner detection. CV pipeline handles this instead."""
        response = self._send_image_prompt(image_bytes, DETECT_CORNERS_PROMPT)
        
        try:
            assert isinstance(response, str)
            corners = json.loads(response)
            return corners
        
        except (json.JSONDecodeError, TypeError):
            print(f"BACKEND:\tFailed to parse corner coordinates: {response}")
            return None

    def _send_image_prompt(self, image_bytes: bytes, prompt: str, is_flash: bool = True) -> str | None:
        """
        Send a chat completion request with the image input
        """
        image_encoded = types.Part.from_bytes(
                            data=image_bytes,
                            mime_type='image/jpeg'
                            )

        try:
            response = self.client.models.generate_content(
                        model=self.flash_version if is_flash else self.pro_version,
                        contents=[image_encoded, prompt]
                        )
            return response.text
        except Exception as e:
            print(f"BACKEND:\tAPI call failed: {e}")
            return None
#endregion
# ================================
    


# ================================
#region   Prompts
ANSWER_RUBRIC_PROMPT: str = """You are given an image of a student's handwritten work in response to a math problem.
Included in this prompt, preceded by `QUESTION:` is the problem the student is answering.
Your task is to answer a question/questions (in a new line, preceded by `PROMPT:`) based solely on the visual content of the student's work.
Your answer should be clear and concise, and directly relate to the image.
If a question can be answered with a yes or no, only generate your answer as `YES` or `NO`.
Otherwise, generate your answer as raw text, with no prefixes or sentences (e.g. `3x+2=8` or `5`).
If multiple questions are given in the prompt, separate your answers for each with `; `."""


ANSWER_MULTI_RUBRIC_PROMPT: str = """You are given an image of a student's handwritten work in response to a math problem.
Included in this prompt, preceded by `QUESTION:`, is the math problem the student is answering.
You will also receive a list of grading criteria, preceded by `RUBRICS:` and formatted as a list of strings in brackets (e.g., `["Correct equation setup", "Calculated correct final answer"]`). Each rubric item is a declarative phrase or statement.
Your task is to evaluate whether each rubric statement accurately describes the visual content of the student's work.
For each rubric item, output strictly `YES` if the statement is true/present, or `NO` if it is false/missing.
Format your final output as a single string of `YES` or `NO` responses, separated by semicolons with no spaces, corresponding to the exact order of the given rubrics (e.g., `YES;YES;NO`). Do not generate any additional text, sentences, prefixes, or explanations."""


COMPARE_EXPECTED_FINAL_ANSWER_PROMPT: str = """You are given an image of a student's handwritten work in response to a math problem.
Included in this prompt, preceded by `QUESTION:` is the problem the student is answering.
Your task is to evaluate if the student's final answer is the same as the expected final answer (in a new line, preceded by `ANSWER:`).
Your answer should be clear and concise, and generated as only `YES` or `NO`.
If the student does not have a clear final answer, generate your answer as `UNCLEAR`."""


DETECT_CORNERS_PROMPT = """You are analyzing an image of a document/page that may be skewed or rotated.
Identify the 4 corners of the document's rectangular boundary.
Return ONLY a JSON array with 4 coordinate pairs in this EXACT order:
[[x_top_left, y_top_left], [x_top_right, y_top_right], [x_bottom_right, y_bottom_right], [x_bottom_left, y_bottom_left]]

Where (0,0) is the top-left of the image. Use integers only.
Example: [[10, 15], [590, 12], [595, 470], [5, 475]]"""


FIND_ITEM_NUMBER_PROMPT: str = """Identify the encircled label in the TOP-LEFT corner.

Rules:
1. Return ONLY the label you see, or `UNKNOWN` if illegible
2. Handwriting may be poor—interpret generously (b may look like 6, a may look like 9, lowercase/uppercase vary)
3. Examine shape carefully (vertical line=1, curve+base=2, stacked curves=3, cross=4)
4. NO explanations, NO other text—output the label only"""


FIND_NEAREST_ITEM_NUMBER_PROMPT = lambda l : f"""Identify the encircled label in the TOP-LEFT corner.

Valid choices: {l}

Rules:
1. Return ONLY a value from the list above, or `UNKNOWN` if illegible
2. ALWAYS match to the closest choice—interpret generously (b may look like 6, a may look like 9, lowercase/uppercase vary)
3. Prioritize matching over strict reading—if ambiguous, pick the most similar valid choice
4. Examine shape carefully (vertical line=1, curve+base=2, stacked curves=3, cross=4)
5. NO explanations, NO other text—output the matched label only"""
#endregion
# ================================
