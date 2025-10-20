"""
Crude application for evaluating answers using AI. Powered by Google Gemini API.
"""
import csv
import os

from all_prompts import *
from google import genai
from google.genai import types
from dotenv import load_dotenv
from PIL import Image, ImageEnhance
from io import BytesIO

import numpy as np
import cv2


IMAGE_PATH = "dataset/2.jpeg"
QUESTION_PATH = "dataset/2.csv"
RUBRIC_QUESTION = "What is the student's final answer? What is the expected answer for the question? Are they the same?"


class PILImagePreprocessor:
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

class CVImagePreprocessor:
	def load_image(self, image_path: str) -> bytes:
		"""
		Load image (unencoded) and return as bytes
		"""
		image = cv2.imread(image_path, cv2.IMREAD_COLOR)
		if image is None:
			raise ValueError(f"Could not load image from {image_path}")
		
		ret, buffer = cv2.imencode('.jpg', image)
		if not ret:
			raise ValueError("Failed to encode image")
		return buffer.tobytes()

	def brighten(self, image_bytes: bytes, amount: float) -> bytes:
		"""
        Brighten the image by scaling pixel values with (1 + amount).
        - Input: JPEG bytes
        - Output: Brightened JPEG bytes
        - amount > 0 increases brightness; < 0 decreases it.
        """
		image = self._decode_bytes(image_bytes)
        
        # Apply brightness
		brightened = cv2.convertScaleAbs(image, alpha=1, beta=amount)
        
        # Encode back to JPEG bytes
		ret, buffer = cv2.imencode('.jpg', brightened)
		if not ret:
			raise ValueError("Failed to encode brightened image")
		return buffer.tobytes()
	
	def adjust_contrast(self, image_bytes: bytes, amount: float) -> bytes:
		"""
		Increase/decrease contrast by given alpha
		"""
		image = self._decode_bytes(image_bytes)
        
		contrasted = cv2.convertScaleAbs(image, alpha=amount, beta=128*(1 - amount))
        
        # Encode back to JPEG bytes
		ret, buffer = cv2.imencode('.jpg', contrasted)
		if not ret:
			raise ValueError("Failed to encode contrasted image")
		return buffer.tobytes()
	
	def save_image(self, image_bytes: bytes, save_path: str) -> None:
		"""
		Save the processed image (in JPEG format) to the specified path.
		- image_bytes: The image in byte form, after any preprocessing (brightened, cropped, etc.)
		- save_path: The path where the image will be saved, including the filename and .jpeg extension.
		"""

		with open(save_path, "wb") as f:
			ret = f.write(image_bytes)
		if not ret:
			raise ValueError(f"Failed to save image to {save_path}")
		print(f"Image saved to {save_path}")


	def _decode_bytes(self, image_bytes: bytes):
		"""
		Decode bytes into BGR uint8 array
		"""
		nparr = np.frombuffer(image_bytes, np.uint8)
		image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

		if image is None:
			raise ValueError("Failed to decode image bytes")
		
		return image


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
		self.imager = CVImagePreprocessor()

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



def extract_csv_column_to_txt(csv_path: str, column_name: str, output_path: str) -> None:
	if not os.path.exists(csv_path):
		raise FileNotFoundError(f"CSV file not found: {csv_path}")

	# Ensure output directory exists
	os.makedirs(os.path.dirname(output_path), exist_ok=True)

	extracted_entries = []

	# Read the CSV file
	with open(csv_path, mode="r", encoding="utf-8", newline="") as csvfile:
		reader = csv.DictReader(csvfile)
		if column_name not in reader.fieldnames:
			raise ValueError(f"Column '{column_name}' not found in CSV headers: {reader.fieldnames}")
	
		for row in reader:
			value = row[column_name].strip() if row[column_name] is not None else ""
			extracted_entries.append(value)
		
	# Write entries to .txt file (one per line)
	with open(output_path, mode="w", encoding="utf-8") as txtfile:
		for entry in extracted_entries:
			txtfile.write(entry + "\n")

	print(f"Extracted column {column_name} from '{csv_path}' saved to '{output_path}'")


if __name__ == "__main__":
	load_dotenv()

	image_preprocessor = CVImagePreprocessor()
	contexter = CSVProcessor()
	ai_evaluator = AIAnswerEvaluator()

	system_prompt = ANSWER_RUBRIC_PROMPT
	rubric_question = RUBRIC_QUESTION
	image_path = IMAGE_PATH
	question_path = QUESTION_PATH

	context = contexter.get_context(question_path)
	context_question, expected_answer = context
	# print("CONTEXT:", context_question, expected_answer)
	
	image_bytes = image_preprocessor.load_image(image_path)
	image_bytes = image_preprocessor.brighten(image_bytes, amount=0.2)
	image_bytes = image_preprocessor.adjust_contrast(image_bytes, amount=1.2)

	user_prompt = f"CONTEXT:{context_question}\nPROMPT:{rubric_question}"

	item_number = ai_evaluator.get_response(image_bytes, FIND_ITEM_NUMBER_PROMPT, "")
	response = ai_evaluator.get_response(image_bytes, system_prompt, user_prompt)

	print(f"ITEM NUMBER: {item_number}\nRESPONSE: {response}")
