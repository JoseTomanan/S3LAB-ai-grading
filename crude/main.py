"""
Crude application for evaluating answers using AI.
"""
from base64 import b64encode
from openai import OpenAI


class OpenAIAnswerEvaluator:
	def __init__(self):
		api_key = ...
		self.client = OpenAI(api_key=api_key)

	def encode_image(self, image_path):
		"""Encode image in base64"""
		with open(image_path, "rb") as image_file:
			encoded_file = b64encode(image_file.read())
			return encoded_file.decode("utf-8")

	def get_response(self, image_path, system_prompt, user_prompt=None):
		"""Send to OpenAI """
		...


if __name__ == "__main__":
	image_path = ...

	prompt = f"""
		You are given an image of a student's handwritten work in response to a math problem. 
		The student's work is shown on the right side of the image, and the problem is displayed on the left side for context.
		Your task is to answer a question based solely on the visual content of the student's handwritten work,
		which is present on the right side of the image. Your answer should be clear and concise, and directly relate to the image presented
		on the right side of the given image. 
		For example, given the question: 
		"What is the equation shown in the image?"
		Generate your answer as: "3x + 2 = 8"
		"""
	
	ai_evaluator = OpenAIAnswerEvaluator()
	response = ai_evaluator.generate()
