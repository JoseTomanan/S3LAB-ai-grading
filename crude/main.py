"""
Crude application for evaluating answers using AI. Powered by Google Gemini API.
"""
import csv
import os

from all_prompts import *

from google import genai
from google.genai import types
from dotenv import load_dotenv


class ImagePreprocessor:
	...

class CSVProcessor:
	def get_context(self, question_path: str) -> list[str]:
		"""Get first problem-answer pair from CSV file. Assumed structure is PROBLEM|ANSWER"""
		with open(question_path, "r") as csv_file:
			reader = csv.reader(csv_file, delimiter="|")
			return next(reader)

class AIAnswerEvaluator:
	def __init__(self):
		api_key = os.getenv("GEMINI_API_KEY")
		self.client = genai.Client(api_key=api_key)
		
	def get_image(self, image_path: str) -> bytes:
		"""Get unencoded image"""
		with open(image_path, "rb") as image_file:
			return image_file.read()

	def get_response(self, image_path: str, system_prompt: str, user_prompt: str):
		"""Send a chat completion request with the image input"""
		image_bytes = self.get_image(image_path)
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

	system_prompt = ANSWER_RUBRIC_PROMPT
	image_path = "dataset/2.jpeg"
	question_path = "dataset/2.csv"

	context = CSVProcessor().get_context(question_path)
	context_question, expected_answer = context
	
	print("CONTEXT: ", context_question, expected_answer)

	rubric_question = "What is the student's final answer? What is the expected answer for the question? Are they the same?"
	
	user_prompt = f"CONTEXT:{context_question}\nPROMPT:{rubric_question}"
	
	ai_evaluator = AIAnswerEvaluator()
	response = ai_evaluator.get_response(image_path, system_prompt, user_prompt)

	print(f"RESPONSE: {response}")
