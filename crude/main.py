"""
Crude application for evaluating answers using AI. Powered by Google Gemini API.
"""
import os

from all_prompts import ANSWER_RUBRIC_PROMPT

from google import genai
from google.genai import types

# from google import generativeai
# from google.generativeai import types

from dotenv import load_dotenv

import numpy as np
import matplotlib.pyplot as plt
import cv2

class ImagePreprocessor:
	def load_image(self, image_path: str) -> np.ndarray:
		image = cv2.imread(image_path, cv2.IMREAD_COLOR)
		if image is None:
			raise ValueError(f"Could not load image from {image_path}")
		return image
	
	def brighten(self, image: np.ndarray, amount: float = 0.1) -> np.ndarray:
		"""
        Brighten the image by scaling pixel values with (1 + amount).
        - amount > 0 increases brightness; < 0 decreases it.
        - Works directly on uint8 BGR arrays; preserves dtype/shape.
        """
        # Use cv2.convertScaleAbs for efficient scaling (alpha = scale factor)
		brightened = cv2.convertScaleAbs(image, alpha=(1 + amount), beta=0)
		return brightened

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
					f"{system_prompt} {user_prompt}"
				]
			)

		return response.text


if __name__ == "__main__":
	load_dotenv()

	system_prompt = ANSWER_RUBRIC_PROMPT
	
	image_path = "dataset/2.jpeg"
	expected_answer = "2"
	
	user_prompt = f"Can the final answer (boxed) be interpreted as '{expected_answer}'?"
	
	ai_evaluator = AIAnswerEvaluator()
	response = ai_evaluator.get_response(image_path, system_prompt, user_prompt)

	print(f"RESPONSE: {response}")
