"""
Crude application for evaluating answers using AI. Powered by Google Gemini API.
"""
import csv
import os

from all_prompts import *

from google import genai
from google.genai import types

# from google import generativeai
# from google.generativeai import types

from dotenv import load_dotenv

import numpy as np # Required for byte-to-array conversion in brighten
import cv2

class ImagePreprocessor:
	def load_image(self, image_path: str) -> bytes:
		image = cv2.imread(image_path, cv2.IMREAD_COLOR)
		if image is None:
			raise ValueError(f"Could not load image from {image_path}")
		
		ret, buffer = cv2.imencode('.jpg', image)
		if not ret:
			raise ValueError("Failed to encode image")
		return buffer.tobytes()
	
	def brighten(self, image_bytes: bytes, amount: float = 0.1) -> bytes:
		"""
        Brighten the image by scaling pixel values with (1 + amount).
        - Input: JPEG bytes
        - Output: Brightened JPEG bytes
        - amount > 0 increases brightness; < 0 decreases it.
        """
        # Decode bytes to BGR uint8 array
		nparr = np.frombuffer(image_bytes, np.uint8)
		image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
		if image is None:
			raise ValueError("Failed to decode image bytes")
        
        # Apply brightness
		brightened = cv2.convertScaleAbs(image, alpha=(1 + amount), beta=0)
        
        # Encode back to JPEG bytes
		ret, buffer = cv2.imencode('.jpg', brightened)
		if not ret:
			raise ValueError("Failed to encode brightened image")
		return buffer.tobytes()

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
		"""Image Preprocessing"""
		# preprocessor = ImagePreprocessor()
		# image_bytes = preprocessor.load_image(image_path)
		# brightened_bytes = preprocessor.brighten(image_bytes, amount=0.2)

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
