import cv2
import numpy as np
import sys
from pathlib import Path

# Add backend to path for imports
script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir))

from ai_interface import AIAnswerEvaluator

# Test with your actual image
image_path = "C:/Users/Hamdi/Documents/School/CS_198/S3LAB-ai-grading/dataset/3-Rizal_Seatwork-1_202160151_1_39587ba38d7547e2be4911c9afba0424_2.jpg"
image_bytes = cv2.imencode('.jpg', cv2.imread(image_path))[1].tobytes()

evaluator = AIAnswerEvaluator()
result = evaluator.get_item_number_ocr(image_bytes)

print(f"Detected item number: {result}")