"""
All object classes
"""
import csv
import os

from all_prompts import *
from find_box import *

from google import genai
from google.genai import types

import numpy as np
import cv2



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
	def _detect_paper_type(self, image_bytes: bytes, line_threshold: int = 5, min_line_length: int = 100) -> str:
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
		
	def _remove_horizontal_lines(self, image_bytes: bytes, kernel_size: int = 3) -> bytes:
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

	def _deskew_for_blank(self, image_bytes: bytes, angle_threshold: float = 10) -> bytes:
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
	
	def _find_endpoints(self, h: cv2.typing.MatLike):
		"""Internal function; find endpoints of sheet. Ordered clockwise, starting from upper left."""
		h = h.reshape((4,2))
		hnew = np.zeros((4,2),dtype = np.float32)

		add = h.sum(1)
		hnew[0] = h[np.argmin(add)]
		hnew[2] = h[np.argmax(add)]

		diff = np.diff(h,axis = 1)
		hnew[1] = h[np.argmin(diff)]
		hnew[3] = h[np.argmax(diff)]

		return hnew

	def find_first_box(self, image_bytes: bytes) -> bytes:
		"""Find first box within image. SOURCE: https://github.com/AdityaPai2398/CamScanner-In-Python"""
		image_matlike = self._decode_bytes(image_bytes)
		return self._encode_to_bytes(find_first_box(image_matlike))



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
