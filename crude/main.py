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


IMAGE_PATH = "dataset/2_test_good.jpg"
QUESTION_PATH = "dataset/2.csv"
RUBRIC_QUESTION = "Based only on the visual content of the student's work, what is the student's final answer? What is the correct mathematical answer to the problem? Are they the same?"



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
        
		brightened = cv2.convertScaleAbs(image, alpha=1, beta=amount)
        
		return self._encode_to_bytes(brightened)
	
	def adjust_contrast(self, image_bytes: bytes, amount: float) -> bytes:
		"""
		Increase/decrease contrast by given alpha
		"""
		image = self._decode_bytes(image_bytes)
        
		contrasted = cv2.convertScaleAbs(image, alpha=amount, beta=128*(1 - amount))
        
		return self._encode_to_bytes(contrasted)
	
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


	def _decode_bytes(self, image_bytes: bytes) -> cv2.typing.MatLike:
		"""
		Decode bytes into BGR uint8 array
		"""
		nparr = np.frombuffer(image_bytes, np.uint8)
		image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

		if image is None:
			raise ValueError("Failed to decode image bytes")
		
		return image
	
	def _encode_to_bytes(self, image_matrix: cv2.typing.MatLike) -> bytes:
		"""
		Encode BGR uint8 array back to JPEG bytes
		"""
		ret, buffer = cv2.imencode('.jpg', image_matrix)
		if not ret:
			raise ValueError("Failed to encode image")
		return buffer.tobytes()

	# Experimental Additions
	def detect_paper_type(self, image_bytes: bytes, line_threshold: int = 5, min_line_length: int = 100) -> str:
		"""
        Detect if image is lined (Group L) or blank (Group B) paper.
        - line_threshold: Min number of horizontal lines to classify as lined.
        - Returns: 'lined' or 'blank'.
        """
		image = self._decode_bytes(image_bytes)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Detect lines using probabilistic Hough Transform
		lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=min_line_length, maxLineGap=10)
        
		horizontal_lines = 0
		if lines is not None:
			for line in lines:
				assert isinstance(line, np.ndarray) and line.ndim == 2
				x1, y1, x2, y2 = line[0]
				# Check if mostly horizontal (small angle)
				if abs(y2 - y1) < 10:  # Vertical tolerance for "horizontal"
					horizontal_lines += 1
        
		paper_type = 'lined' if horizontal_lines >= line_threshold else 'blank'
		print(f"Detected {horizontal_lines} horizontal lines → Paper type: {paper_type}")
		return paper_type
		
	def remove_horizontal_lines(self, image_bytes: bytes, kernel_size: int = 3) -> bytes:
		"""
        Remove horizontal lines from lined paper using morphology.
        - kernel_size: Width of horizontal structuring element.
        """
		image = self._decode_bytes(image_bytes)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
				
		# Horizontal kernel for line detection
		horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size * 20, 1))  # Wide for horizontals
		horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel)  # Extract lines
		vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_size * 20))  # For any vertical noise
		vertical_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, vertical_kernel)
				
		# Combine and mask
		lines_mask = cv2.addWeighted(horizontal_lines, 1, vertical_lines, 1, 0)
		lines_mask = cv2.dilate(lines_mask, horizontal_kernel, iterations=1)  # Thicken mask
				
		# Inpaint: Replace lines with surrounding pixels (seamless clone for better results)
		mask_inv = cv2.bitwise_not(lines_mask)
		result = cv2.inpaint(image, lines_mask, 1, cv2.INPAINT_TELEA)  # Telea algorithm for smooth fill
				
		return self._encode_to_bytes(result)

	def deskew_for_blank(self, image_bytes: bytes, angle_threshold: float = 10) -> bytes:
		"""
		Deskew (straighten) for blank paper to handle uneven layouts.
		"""
		image = self._decode_bytes(image_bytes)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
				
		# Edge detection and Hough for skew angle
		edges = cv2.Canny(gray, 50, 150, apertureSize=3)
		lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
				
		if lines is not None:
			angles = []
			for rho, theta in lines[:, 0]:
				angle = (theta * 180 / np.pi) - 90
				if 0 < abs(angle) < angle_threshold:  # Small angles only
					angles.append(angle)
            
			if angles:
				median_angle = np.median(angles)
				if abs(median_angle) > 0.5:  # Worth correcting
					(h, w) = image.shape[:2]
					center = (w // 2, h // 2)
					assert isinstance(median_angle, float)

					M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
					deskewed = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

					return self._encode_to_bytes(deskewed)
        
		return image_bytes  # No skew detected



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
	print("CONTEXT:", context_question, expected_answer)
	print("PROMPT:", rubric_question)
	
	image_bytes = image_preprocessor.load_image(image_path)
	image_bytes = image_preprocessor.brighten(image_bytes, amount=0.2)
	image_bytes = image_preprocessor.adjust_contrast(image_bytes, amount=1.2)

	user_prompt = f"CONTEXT:{context_question}\nPROMPT:{rubric_question}"

	item_number = ai_evaluator.get_response(image_bytes, FIND_ITEM_NUMBER_PROMPT, "")
	print(f"ITEM NUMBER: {item_number}")

	response = ai_evaluator.get_response(image_bytes, system_prompt, user_prompt)
	print(f"RESPONSE: {response}")
