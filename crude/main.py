"""
Crude application for evaluating answers using AI.
"""
import google.generativeai as genai

from base64 import b64encode
from PIL import Image


class AIAnswerEvaluator:
	def __init__(self):
		api_key = ...

		self.model = "gemini-1.5-pro"

		genai.configure(api_key=api_key)
		self.client = genai.GenerativeModel(self.model)

	def encode_image(self, image_path):
		"""Encode image in base64"""
		with open(image_path, "rb") as image_file:
			encoded_file = b64encode(image_file.read())
			return encoded_file.decode("utf-8")

	def get_response(self, image_path, system_prompt, user_prompt):
		"""Send a chat completion request with the image input"""
		...


if __name__ == "__main__":
	image_path = "../dataset/2.jpeg"

	system_prompt = """
		You are given an image of a student's handwritten work in response to a math problem.
		The student's work is shown on the right side of the image, and the problem is displayed on the left side for context.
		Your task is to answer a question based solely on the visual content of the student's work, present on the right side of the image.
		Your answer should be clear and concise, and directly relate to the image presented on the right side of the given image. 
		For example, given question "what is the equation shown in the image?", generate your answer as: "3x+2=8".
		"""
	
	expected_answer = ""
	user_prompt = f"Is the final answer the same as '{expected_answer}'?"
	
	ai_evaluator = AIAnswerEvaluator()
	response = ai_evaluator.get_response(image_path, system_prompt, user_prompt)

	print(response)
