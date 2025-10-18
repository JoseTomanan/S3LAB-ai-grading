"""
Crude application for evaluating answers using AI. Powered by Google Gemini API.
"""
import csv
import os

from all_prompts import *
from google import genai
from google.genai import types
from dotenv import load_dotenv

from io import BytesIO
from PIL import Image, ImageEnhance


IMAGE_PATH = "dataset/2.jpeg"
QUESTION_PATH = "dataset/2.csv"
RUBRIC_QUESTION = "What is the student's final answer? What is the expected answer for the question? Are they the same?"


class ImagePreprocessor:
	def __init__(self):
		self.buffer = BytesIO()

	def load_image(self, image_path: str) -> Image.Image:
		return Image.open(image_path)

	def increase_visibility(self, image: Image.Image, brighten_val: float=1.3, contrast_val: float=1.5) -> Image.Image:
		brightened = ImageEnhance.Brightness(image)\
				.enhance(brighten_val)
		contrasted = ImageEnhance.Contrast(brightened)\
				.enhance(contrast_val)
		
		return contrasted
	
	def encode_to_bytes(self, image: Image.Image) -> bytes:
		image.save(self.buffer, format="JPEG")
		image_bytes = self.buffer.getvalue()

		return image_bytes


class CSVProcessor:
	def get_context(self, question_path: str) -> list[str]:
		"""
		Get first problem-answer pair from CSV file. Assumed structure is PROBLEM|ANSWER
		"""
		with open(question_path, "r") as csv_file:
			reader = csv.reader(csv_file, delimiter="|")
			return next(reader)


class AIAnswerEvaluator:
	def __init__(self):
		api_key = os.getenv("GEMINI_API_KEY")
		self.client = genai.Client(api_key=api_key)
		self.imager = ImagePreprocessor()

	def get_response(self, image_bytes: bytes, system_prompt: str, user_prompt: str):
		"""
		Send a chat completion request with the image input
		"""
		image_encoded = types.Part.from_bytes(
				data=image_bytes,
				mime_type='image/jpeg'
			)

		response = self.client.models.generate_content(
				model="gemini-2.5-pro",
				contents=[
					image_encoded,
					f"{system_prompt}\n{user_prompt}"
				]
			)

		return response.text


if __name__ == "__main__":
	load_dotenv()

	image_preprocessor = ImagePreprocessor()
	contexter = CSVProcessor()
	ai_evaluator = AIAnswerEvaluator()

	system_prompt = ANSWER_RUBRIC_PROMPT
	rubric_question = RUBRIC_QUESTION
	image_path = IMAGE_PATH
	question_path = QUESTION_PATH

	context = contexter.get_context(question_path)
	context_question, expected_answer = context
	
	image = image_preprocessor.load_image(image_path)
	image = image_preprocessor.increase_visibility(image, brighten_val=1.2, contrast_val=1.7)
	image_bytes = image_preprocessor.encode_to_bytes(image)

	user_prompt = f"CONTEXT:{context_question}\nPROMPT:{rubric_question}"

	item_number = ai_evaluator.get_response(image_bytes, FIND_ITEM_NUMBER_PROMPT, "")
	print("ITEM NUMBER:", item_number)

	response = ai_evaluator.get_response(image_bytes, system_prompt, user_prompt)
	print("RESPONSE:", response)
