import os
from dotenv import load_dotenv
import cv2
import numpy as np
from google import genai
from google.genai import types

import logging
logger = logging.getLogger(__name__)


# ================================
#   Class
# ================================
class AIAnswerEvaluator:
    def __init__(self, flash: bool = False):
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=api_key)
        self.version = "gemini-2.5-flash" if flash else "gemini-2.5-flash-lite"

    def get_item_number(self, image_bytes: bytes):
        return self._send_image_prompt(image_bytes, FIND_ITEM_NUMBER_PROMPT)
    
    def evaluate_expected_answer(self, image_bytes: bytes, question: str, answer: str):
        return self._send_image_prompt(
                        image_bytes,
                        f"{COMPARE_EXPECTED_FINAL_ANSWER_PROMPT}\nQUESTION: {question}\nANSWER: {answer}"
                        )
        
    def evaluate_rubric(self, image_bytes: bytes, question: str, rubric: str):
        return self._send_image_prompt(
                        image_bytes,
                        f"{ANSWER_RUBRIC_PROMPT}\nQUESTION: {question}\nPROMPT: \"{rubric}\": can this be said about the answer?"
                        )

    def _send_image_prompt(self, image_bytes: bytes, prompt: str) -> str | None:
        """
        Send a chat completion request with the image input
        """
        image_encoded = types.Part.from_bytes(
                            data=image_bytes,
                            mime_type='image/jpeg'
                            )

        response = self.client.models.generate_content(
                    model=self.version,
                    contents=[image_encoded, prompt]
                    )

        return response.text
    


    # OCR Item Number Label - Hamdi
    def get_item_number_ocr(self, image_bytes: bytes) -> str:
        """
        Extract item number using traditional OpenCV.
        Skips PaddleOCR to avoid oneDNN compatibility issues on Windows.
        """
        try:
            # Load image
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                logger.warning("❌ Failed to decode image for item number detection")
                return "UNKNOWN"
            
            h, w = image.shape[:2]
            logger.info(f"✓ Image loaded: {w}x{h}px")
            
            # Crop top-left corner (where encircled number typically is)
            # ADJUSTED: Larger crop area to ensure number is captured
            crop_h = int(h * 0.40)  # Was 0.30
            crop_w = int(w * 0.40)  # Was 0.35
            cropped = image[0:crop_h, 0:crop_w]
            
            # DEBUG: Save cropped region for inspection
            cv2.imwrite("debug_crop.jpg", cropped)
            logger.info(f"✓ Saved debug_crop.jpg (crop: {crop_w}x{crop_h})")
            
            # Convert to grayscale
            gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Threshold using Otsu's method (automatic threshold)
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # DEBUG: Save thresholded image
            cv2.imwrite("debug_thresh.jpg", thresh)
            
            # Morphological operations to connect broken strokes
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            dilated = cv2.dilate(thresh, kernel, iterations=2)
            eroded = cv2.erode(dilated, kernel, iterations=1)
            
            # DEBUG: Save morphological result
            cv2.imwrite("debug_morph.jpg", eroded)
            
            # Find contours
            contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            logger.info(f"✓ Found {len(contours)} total contours")
            
            # Filter contours - MORE PERMISSIVE THRESHOLDS
            candidates = []
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                x, y, cw, ch = cv2.boundingRect(contour)
                
                # Log ALL contours for debugging
                if area > 50:  # Log anything visible
                    aspect = cw / ch if ch > 0 else 0
                    logger.debug(f"  Contour #{i}: area={area:.0f}, aspect={aspect:.2f}, bbox=({x},{y},{cw},{ch})")
                
                # ⚠️ MORE PERMISSIVE FILTERS (adjust based on your images)
                if 100 < area < 60000:  # Was 300 < area < 40000
                    aspect_ratio = cw / ch if ch > 0 else 0
                    if 0.2 < aspect_ratio < 2.0:  # Was 0.3 < aspect < 1.5
                        candidates.append((x, y, cw, ch, contour, area))
            
            logger.info(f"✓ Found {len(candidates)} candidate digit contours (passed filters)")
            
            # DEBUG: Save all contours visualization
            if not candidates:
                debug_img = cropped.copy()
                cv2.drawContours(debug_img, contours, -1, (0, 255, 0), 2)
                cv2.imwrite("debug_all_contours.jpg", debug_img)
                logger.warning("⚠ No candidate contours passed filters - see debug_all_contours.jpg")
            
            if not candidates:
                return "UNKNOWN"
            
            # Sort by area (largest first - likely the encircled number)
            candidates.sort(key=lambda k: k[5], reverse=True)
            x, y, cw, ch, contour, area = candidates[0]
            
            logger.info(f"✓ Top candidate: area={area:.0f}, bbox=({x},{y},{cw},{ch})")
            
            # Extract ROI
            digit_roi = gray[y:y+ch, x:x+cw]
            cv2.imwrite("debug_digit_roi.jpg", digit_roi)
            
            # Count holes (for classification)
            roi_thresh = thresh[y:y+ch, x:x+cw]
            contour_hierarchy = cv2.findContours(
                roi_thresh,
                cv2.RETR_TREE,
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            hole_count = 0
            if len(contour_hierarchy) > 1 and contour_hierarchy[1] is not None:
                hierarchy = contour_hierarchy[1][0]
                for i, h in enumerate(hierarchy):
                    if h[3] != -1:
                        hole_count += 1
            
            logger.info(f"✓ Hole count: {hole_count}, Aspect ratio: {cw/ch:.2f}")
            
            # Classification
            aspect = cw / ch if ch > 0 else 0
            
            if aspect < 0.4:
                result = "1"
            elif hole_count >= 2:
                result = "8"
            elif hole_count == 1:
                result = "0" if aspect > 0.9 else "6"
            else:
                result = "2"  # Default fallback
            
            logger.info(f"✓ Classified as: '{result}'")
            return result
            
        except Exception as e:
            logger.error(f"✗ CV extraction failed: {e}", exc_info=True)
            return "UNKNOWN"

# ================================
#   Prompts
# ================================
ANSWER_RUBRIC_PROMPT: str = "You are given an image of a student's handwritten work in response to a math problem. Included in this prompt, preceded by `QUESTION:` is the problem the student is answering. Your task is to answer a question/questions (in a new line, preceded by `PROMPT:`) based solely on the visual content of the student's work. Your answer should be clear and concise, and directly relate to the image. If a question can be answered with a yes or no, only generate your answer as `YES` or `NO`. Otherwise, generate your answer as raw text, with no prefixes or sentences (e.g. `3x+2=8` or `5`). If multiple questions are given in the prompt, separate your answers for each with `; `."

COMPARE_EXPECTED_FINAL_ANSWER_PROMPT: str = "You are given an image of a student's handwritten work in response to a math problem. Included in this prompt, preceded by `QUESTION:` is the problem the student is answering. Your task is to evaluate if the student's final answer is the same as the expected final answer (in a new line, preceded by `ANSWER:`). Your answer should be clear and concise, and generated as only `YES` or `NO`. If the student does not have a clear final answer, generate your answer as `UNCLEAR`."

FIND_ITEM_NUMBER_PROMPT: str = """You are identifying an encircled item number in the TOP-LEFT corner of a student's answer sheet.

IMPORTANT RULES:
1. Look ONLY at the upper-left corner region
2. The number is encircled (has a circle around it)
3. Common numbers are: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
4. Handwriting may be poor - look carefully at the shape
5. Do NOT confuse similar numbers (1 vs 7, 2 vs 3, 5 vs 6)

EXAMPLES:
- A single vertical line = 1
- A curve with horizontal base = 2
- Two curves stacked = 3
- Vertical line with horizontal cross = 4

Generate ONLY the number (e.g., `2`) or `NONE` if no number exists.
Do not add any explanation or text."""
