"""
Crude application for evaluating answers using AI. Powered by Google Gemini API.
"""
import os

from google import genai
from google.genai import types

from dotenv import load_dotenv


class AIAnswerEvaluator:
	def __init__(self):
		api_key = os.getenv("GEMINI_API_KEY")
		self.client = genai.Client(api_key=api_key)
		
	def get_image(self, image_path: str) -> bytes:
		"""Get unencoded image"""
		with open(image_path, "rb") as image_file:
			return image_file.read()

	def get_response(self, image_path: str, system_prompt: str, user_prompt: str):
		"""Send a chat completion request with the image input. Uses inline image for now"""
		image_bytes = self.get_image(image_path)
		
		image_encoded = types.Part.from_bytes(
				data=image_bytes,
				mime_type='image_jpeg'
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

	system_prompt = """
		You are given an image of a student's handwritten work in response to a math problem.
		The student's work is shown on the right side of the image, and the problem is displayed on the left side for context.
		Your task is to answer a question based solely on the visual content of the student's work, present on the right side of the image.
		Your answer should be clear and concise, and directly relate to the image presented on the right side of the given image. 
		For example, given question "what is the equation shown in the image?", generate your answer as: "3x+2=8".
		"""
	
	image_path = "../dataset/2.jpeg"
	expected_answer = "2"
	
	user_prompt = f"Can the final answer (boxed) be interpreted as '{expected_answer}'?"
	
	ai_evaluator = AIAnswerEvaluator()
	response = ai_evaluator.get_response(image_path, system_prompt, user_prompt)

	print(response)
